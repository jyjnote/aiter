# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Generator
import json
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import logging
import inflect # "123" → "one hundred twenty three" 와 같은 곳에 사용됨
try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    logging.info("failed to import ttsfrd, use wetext instead")
    from wetext import Normalizer as ZhNormalizer
    from wetext import Normalizer as EnNormalizer
    use_ttsfrd = False
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()# 사용할 토큰나이저 정의
        self.feat_extractor = feat_extractor # 멜 스펙트로그램 등의 특성 추출 함수
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1 #멀티스레드 연산을 제한해 재현성 및 안정성 보장

        ## Camp+ 화자 임베딩 모델
        # ONNX Runtime 세션 객체, 화자 임베딩(Speaker Embedding) 모델 이름
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        logging.info(f"[Frontend-INFO] Loaded Camp+ speaker embedding model: {campplus_model}")

        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
                                                                                "CPUExecutionProvider"])

        logging.info(f"[Frontend-INFO] Loaded Speech tokenizer model: {speech_tokenizer_model}")

        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
            logging.info(f"[Frontend-INFO] Loaded speaker info from {spk2info}")
        else:
            self.spk2info = {}
            logging.info(f"[Frontend-INFO] No speaker info file found, using empty dict")

        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd: # 텍스트 정규화 엔진 초기화
            self.frd = ttsfrd.TtsFrontendEngine() # 설치 환경에 ttsfrd 있으면, 없으면 fallback 으로 wetext 사용
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
            logging.info(f"[Frontend-INFO] Using TTSFRD text normalizer (pinyinvg mode)")

        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine() # 영어 숫자 → 철자 변환 등 ("123" → "one hundred twenty three").
            logging.info(f"[Frontend-INFO] Using WeText normalizer (ZhNormalizer + EnNormalizer)")

    def _extract_text_token(self, text): # 텍스트를 토크나이즈해서 모델 입력용 텐서로 만드는 함수
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will return _extract_text_token_generator!')
            # NOTE add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32).to(self.device)
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            logging.info(f"[BPE-DEBUG | _extract_text_token] input='{text}'")
            logging.info(f"[BPE-DEBUG | _extract_text_token] ids={text_token}")

            try:
                # HuggingFace 기반 토크나이저면 이렇게 원래 subword 단위로 변환 시도
                tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(text_token)
                logging.info(f"[BPE-DEBUG | _extract_text_token] BPE tokens={tokens}")
            except Exception as e:
                logging.info(f"[BPE-DEBUG | _extract_text_token] (convert_ids_to_tokens not available: {e})")

            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
            return text_token, text_token_len

    # _extract_text_token 상단 위 메서드에 사용됨.
    def _extract_text_token_generator(self, text_generator): # 스트리밍(Generator) 입력 모드에서 텍스트를 토큰 단위로 잘라 모델에 흘려보내는 역할
        for text in text_generator: # 1. 외부에서 들어오는 텍스트 조각(Generator)을 하나씩 받음
            text_token, _ = self._extract_text_token(text) # 2. 각 텍스트 조각을 BPE 토큰화 → 텐서로 변환
            for i in range(text_token.shape[1]): # 3. 문장을 토큰 단위(열 단위)로 슬라이스
                yield text_token[:, i: i + 1] # 4. 한 번에 하나의 토큰만 내보냄 (Streaming inference)

    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s' #16kHz 음성 파형 (길이 ≤ 30초)
        # 이게 제로샷 프롬프트 음성을 추출하는 부분, 30초 넘어가는거 넣어 봤는데 해당 오류 나옴
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        
        logging.info(f"[BPE-DEBUG | _extract_speech_token] (speech_token->{speech_token[...,:10]} speech_token_len:->{speech_token_len})")

        return speech_token, speech_token_len
    
    # 음색, 톤, 억양 같은 화자 고유한 목소리 특성을 압축 표현
    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        logging.info(f"[SPK-EMBED-DEBUG] speech shape={speech.shape}")
        logging.info(f"[SPK-EMBED-DEBUG] fbank feat shape={feat.shape}")
        logging.info(f"[SPK-EMBED-DEBUG] embedding dim={embedding.shape} | first 5 vals={embedding[0, :5].tolist()}")

        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will skip text_normalize!')
            return [text]
        if text_frontend is False or text == '':
            return [text] if split is True else text
        text = text.strip()
        if self.use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text) # 합성할 텍스트 토큰화 및 그 길이 
        if zero_shot_spk_id == '': # 새로운 화자를 추출할 경우 아래 과정을 통해 정보를 추출
            prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
            prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k) # 프롬프트 음성을 target 샘플링레이트(resample_rate)로 변환
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
            speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)  # 프롬프트 음성을 discrete speech tokens으로 변환
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                 # cosyvoice2에서는 feature 길이와 speech token 길이 비율을 맞춰야 함 (2:1)
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
                speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
            embedding = self._extract_spk_embedding(prompt_speech_16k)

            # 모델 입력 dict 구성
            model_input = {
                'prompt_text': prompt_text_token,                 # 레퍼런스 텍스트 토큰
                'prompt_text_len': prompt_text_token_len,         # 그 길이
                'llm_prompt_speech_token': speech_token,          # LLM conditioning용 discrete speech tokens
                'llm_prompt_speech_token_len': speech_token_len,  # 그 길이
                'flow_prompt_speech_token': speech_token,         # Flow conditioning에도 같은 토큰 사용
                'flow_prompt_speech_token_len': speech_token_len,
                'prompt_speech_feat': speech_feat,                # frame-level 연속 음향 특징
                'prompt_speech_feat_len': speech_feat_len,        # 그 길이
                'llm_embedding': embedding,                       # LLM용 speaker embedding
                'flow_embedding': embedding                       # Flow용 speaker embedding
            }
        else:
            model_input = self.spk2info[zero_shot_spk_id] # 기존에 저장된 화자 정보(spk2info) 사용 (zero-shot이 아니라 이미 등록된 화자)
        
        # 합성 대상 텍스트 정보 추가
        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate, zero_shot_spk_id)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        model_input = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k, resample_rate, zero_shot_spk_id)
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample) # prompt_speech_resample 다 동일한 프롬프트 음성을 받음
        embedding = self._extract_spk_embedding(prompt_speech_16k) # prompt_speech_resample 다 동일한 프롬프트 음성을 받음
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k) # prompt_speech_resample 다 동일한 프롬프트 음성을 받음
        model_input = {'source_speech_token': source_speech_token, 'source_speech_token_len': source_speech_token_len,
                       'flow_prompt_speech_token': prompt_speech_token, 'flow_prompt_speech_token_len': prompt_speech_token_len,
                       'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
                       'flow_embedding': embedding}
        return model_input