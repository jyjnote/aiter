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
import os
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_speech_16k, self.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot_typing(
            self,
            text_stream,                  # Generator[str | Tuple["RESET", str] | List[int]]
            prompt_text: str,
            prompt_speech_16k,
            zero_shot_spk_id: str = "",
            stream: bool = True,          # <- 전달은 받지만 내부에선 즉시합성 위해 비스트리밍으로 실행
            speed: float = 1.0,
            text_frontend: bool = True,
            chunk_tokens: int | None = None,          # 더 이상 사용하지 않음(호환만 유지)
            interleave_prompt_in_llm: bool = False,   # LLM에 프롬프트 음성토큰 투입 여부
        ):
        """
        [개편] 입력을 받아 문장 단위로 '즉시 합성' 하는 파이프라인.
        - 구두점(.,!?…。！？)이 들어오면 그 시점까지의 문장을 즉시 비스트리밍 합성(stream=False)
        - ("RESET", text) 신호가 오면 해당 text 전체를 즉시 합성 큐에 투입 (구두점 없으면 문단 단위로도 합성)
        - 합성은 백그라운드 스레드에서 진행되어, 합성 중에도 입력 수집/문장 파싱은 계속 진행됨
        - 보이스 클로닝(Zero-shot) 경로를 사용하며, 프롬프트 기반 임베딩/토큰은 1회만 준비 후 재사용
        """
        import re, queue, threading, time, copy, math, torch
        from cosyvoice.utils.file_utils import logging

        # ---------- 0) 보이스 클로닝용 프롬프트 패키지 1회 준비 ----------
        # prompt_text는 zero-shot에선 굳이 쓸 필요 없으므로 '<|endofprompt|>' 로 마감 토큰만 부여
        prompt_pack = self.frontend.frontend_zero_shot(
            "", "<|endofprompt|>", prompt_speech_16k, self.sample_rate, zero_shot_spk_id
        )
        device = self.model.device

        # LLM에 프롬프트 음성토큰을 섞지 않도록 선택한 경우, 해당 필드를 0길이 텐서로 마스킹
        if not interleave_prompt_in_llm:
            prompt_pack["llm_prompt_speech_token"] = torch.zeros(1, 0, dtype=torch.int32, device=device)

        # 이후 문장마다 바뀌는 것은 'text'와 'text_len' 뿐이므로, 나머지 프롬프트 관련 텐서는 재사용
        def build_model_input_for_text(sentence: str) -> dict:
            # 텍스트 정규화(언어별 TN + 문장 길이 제어)
            normed_list = self.frontend.text_normalize(sentence, split=True, text_frontend=text_frontend)
            out = []
            for nc in normed_list:
                t_tok, t_len = self.frontend._extract_text_token(nc)
                pack = copy.copy(prompt_pack)
                pack["text"] = t_tok
                pack["text_len"] = t_len
                out.append((nc, pack))
            return out  # [(정규화문장, model_input_pack), ...]

        # ---------- 1) 파이프라인용 큐/스레드 준비 ----------
        in_q     = queue.Queue()  # feeder → parser 로 들어오는 원문 입력(문자/RESET 등)
        synth_q  = queue.Queue()  # parser → synthesizer 로 넘어가는 "완성 문장"
        out_q    = queue.Queue()  # synthesizer → 최종 제너레이터 로 넘어가는 오디오

        # 1-1) 입력 feeder: 외부 text_stream을 받아 in_q에 그대로 투입
        def feeder():
            for x in text_stream:
                in_q.put(x)
            in_q.put(None)  # 종료 신호
        threading.Thread(target=feeder, daemon=True).start()

        # 1-2) 문장 파서: 구두점 기준으로 문장을 완성시키면 synth_q에 즉시 투입
        #      ("RESET", text) 가 오면 기존 버퍼를 폐기하고, 전달받은 text 전체를 문장 분할 후 투입
        SENT_END = re.compile(r"[.!?…。！？]+")
        buffer_text = []

        def flush_sentence(s: str):
            s = s.strip()
            if not s:
                return
            synth_q.put(s)
            logging.debug("[타이핑] 문장 flush → '%s'", s[:80] + ("..." if len(s) > 80 else ""))

        def split_by_sentence(text: str):
            # 구두점 포함 분할: "안녕. 반가워요!" → ["안녕.", " 반가워요!"]
            parts = re.split(r'(?<=[.!?…。！？])', text)
            # 마지막 조각이 구두점 없이 끝날 수 있어 빈문자 제거
            merged = []
            acc = ""
            for p in parts:
                acc += p
                if SENT_END.search(p):
                    merged.append(acc)
                    acc = ""
            if acc:
                merged.append(acc)
            return [m for m in merged if m.strip()]

        def parser():
            nonlocal buffer_text
            while True:
                it = in_q.get()
                if it is None:
                    # 종료 시, 남은 버퍼를 한 번에 넘김
                    rest = "".join(buffer_text).strip()
                    if rest:
                        for sent in split_by_sentence(rest):
                            flush_sentence(sent)
                    synth_q.put(None)
                    return

                # RESET 신호: 기존 버퍼 폐기 후, 전달 텍스트 전체를 즉시 투입
                if isinstance(it, tuple) and len(it) == 2 and it[0] == "RESET":
                    logging.info("[타이핑] RESET 수신 → 기존버퍼 폐기 & 강제 Flush")
                    buffer_text = []
                    full_text = str(it[1])
                    # 문장 단위로 쪼개서 투입 (구두점 없으면 통문장으로 투입)
                    splitted = split_by_sentence(full_text) or [full_text]
                    for s in splitted:
                        flush_sentence(s)
                    continue

                # 사전 토큰화 입력은 이 경로에선 지원하지 않음(문장 경계 판단이 어렵기 때문)
                if isinstance(it, (list, tuple)) and all(isinstance(x, int) for x in it):
                    logging.warning("[타이핑] 사전토큰화 입력은 즉시합성 경로에서는 미지원 → 무시")
                    continue

                # 일반 문자 입력
                ch = str(it)
                buffer_text.append(ch)
                if SENT_END.search(ch):
                    # 구두점 등장 → 현재까지 버퍼를 문장 단위로 분리하여 모두 투입
                    text = "".join(buffer_text)
                    buffer_text = []
                    for s in split_by_sentence(text):
                        flush_sentence(s)

        threading.Thread(target=parser, daemon=True).start()

        # 1-3) 합성기: synth_q에서 문장을 꺼내 즉시 '비스트리밍' 합성으로 오디오를 생성 후 out_q에 넣음
        def synthesizer():
            while True:
                s = synth_q.get()
                if s is None:
                    out_q.put(None)
                    return

                # 정규화된 문장들에 대해 각각 합성(길면 내부에서 여러개로 쪼개진 리스트가 돌아옴)
                try:
                    for norm_str, model_input in build_model_input_for_text(s):
                        # 즉시 합성: stream=False → 토큰 누적 대기 없음
                        t0 = time.time()
                        for out in self.model.tts(**model_input, stream=False, speed=speed):
                            # out 은 {"tts_speech": Tensor[1, T]} 한 번만 옴
                            wav = out["tts_speech"]
                            secs = float(wav.shape[1]) / float(self.sample_rate)
                            logging.info("[즉시합성] '%s' → %.2fs", norm_str[:40] + ("..." if len(norm_str) > 40 else ""), secs)
                            out_q.put(out)
                        logging.debug("[즉시합성] done | time=%.3fs", time.time() - t0)
                except Exception as e:
                    logging.error("[즉시합성] 실패: %s", e, exc_info=True)

        threading.Thread(target=synthesizer, daemon=True).start()

        # ---------- 2) 최종 제너레이터: out_q 에 쌓이는 오디오를 즉시 Yield ----------
        # 합성 중에도 parser는 계속 입력을 받아 다음 문장을 큐에 넣는다.
        while True:
            o = out_q.get()
            if o is None:
                break
            yield o

