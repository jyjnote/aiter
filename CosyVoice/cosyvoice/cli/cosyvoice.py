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

# 파일: /mnt/raid0/jjy/CosyVoice/cosyvoice/cli/cosyvoice.py
# 클래스: CosyVoice2  (기존 클래스 내부 맨 아래쪽에 추가)
    def inference_zero_shot_typing(
            self,
            text_stream,                  # Generator[str | Tuple[str, str]]
            prompt_text: str,
            prompt_speech_16k,
            zero_shot_spk_id: str = "",
            stream: bool = True,
            speed: float = 1.0,
            text_frontend: bool = True,
            chunk_tokens: int | None = None,
            interleave_prompt_in_llm: bool = False,
        ):
            """
            사용자 타이핑을 바로 토큰으로 흘려보내 bi-stream 합성.
            수정: UI로부터 ("RESET", "full_text") 형태의 신호를 받아 내부 버퍼를 초기화하는 기능 추가.
            """
            import os, queue, threading, time
            import torch
            from cosyvoice.utils.file_utils import logging

            # (0) 프롬프트 음성/텍스트 준비
            norm_prompt_text = self.frontend.text_normalize(
                prompt_text, split=False, text_frontend=text_frontend
            )
            model_input = self.frontend.frontend_zero_shot(
                "", "<|endofprompt|>", prompt_speech_16k, self.sample_rate, zero_shot_spk_id
            )

            # (1) LLM 모델에 프롬프트 음성 토큰을 사용하지 않도록 설정
            device = self.model.device
            if not interleave_prompt_in_llm:
                model_input["llm_prompt_speech_token"] = torch.zeros(1, 0, dtype=torch.int32, device=device)

            # (2) 토크나이저 준비 및 예외 처리 래퍼 함수
            tokenizer = getattr(self.frontend, "tokenizer", None)
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.model_dir, "CosyVoice-BlankEN"))

            def safe_encode(txt: str):
                try:
                    return tokenizer.encode(txt, add_special_tokens=False)
                except TypeError:
                    return tokenizer.encode(txt)
                except Exception as e:
                    logging.error(f"[타이핑] 텍스트 인코딩 실패: {e}")
                    return []

            def safe_decode(ids):
                try:
                    return tokenizer.decode(ids)
                except Exception:
                    return None

            # (3) 텍스트 청크 크기(n) 설정
            try:
                default_chunk = getattr(self.model.llm, "mix_ratio", [5,15])[0]
            except Exception:
                default_chunk = 10
            n_text = chunk_tokens or default_chunk

            # 프롬프트 정보 요약 로그
            prompt_char_len = len(norm_prompt_text)
            prompt_tok_len = len(safe_encode(norm_prompt_text)) if prompt_char_len > 0 else 0
            prompt_secs = float(prompt_speech_16k.shape[1]) / 16000.0 if prompt_speech_16k is not None else 0.0
            ref_llm_tok_len = int(model_input.get("llm_prompt_speech_token", torch.zeros(1,0)).shape[1])
            ref_flow_tok_len = int(model_input.get("flow_prompt_speech_token", torch.zeros(1,0)).shape[1]) if "flow_prompt_speech_token" in model_input else 0
            ref_feat_T = int(model_input.get("prompt_speech_feat", torch.zeros(1,0,80)).shape[1]) if "prompt_speech_feat" in model_input else 0
            logging.debug(
                "[타이핑] 스트리밍 시작 | 스트림=%s 속도=%.3f 인터리브=%s 청크토큰수=%d | "
                "프롬프트_텍스트 글자수=%d 토큰수=%d | 프롬프트_오디오=%.2fs | 참조토큰(llm=%d, flow=%d) 참조특성=%d",
                stream, speed, interleave_prompt_in_llm, n_text,
                prompt_char_len, prompt_tok_len, prompt_secs,
                ref_llm_tok_len, ref_flow_tok_len, ref_feat_T
            )

            # ---------- 4) 입력 스트림을 증분 토큰 제너레이터로 변환 ----------
            q = queue.Queue()

            def feeder():
                for piece in text_stream:
                    q.put(piece)
                q.put(None)  # 종료 신호

            threading.Thread(target=feeder, daemon=True).start()

            buffer_text = ""
            prev_len = 0
            emitted_chunks = 0
            emitted_tokens_total = 0

            def to_token_chunks():
                nonlocal buffer_text, prev_len, emitted_chunks, emitted_tokens_total
                chunk_idx = 0
                while True:
                    item = q.get()
                    if item is None:
                        # 스트림 종료 시 남은 버퍼 모두 처리 (Flush)
                        curr = buffer_text.strip()
                        if curr:
                            ids = safe_encode(buffer_text)
                            delta = ids[prev_len:]
                            delta_len = len(delta)
                            if delta_len > 0:
                                logging.debug(
                                    "[토큰생성기] 스트림 종료, 남은 토큰 처리 | 남은토큰수=%d (이전길이=%d -> 총길이=%d)",
                                    delta_len, prev_len, prev_len + delta_len
                                )
                                for i in range(0, delta_len, n_text):
                                    chunk = delta[i:i+n_text]
                                    if not chunk: continue
                                    chunk_idx += 1
                                    emitted_chunks += 1
                                    emitted_tokens_total += len(chunk)
                                    preview = (safe_decode(chunk) or "").replace("\n", " ")
                                    if len(preview) > 60: preview = preview[:60] + "..."
                                    logging.debug(
                                        "[토큰생성기] >> 출력(남은분량) 청크#%d | 길이=%d | 총출력토큰=%d | 내용='%s'",
                                        chunk_idx, len(chunk), emitted_tokens_total, preview
                                    )
                                    yield torch.tensor([chunk], dtype=torch.int64, device=device)
                                prev_len += delta_len
                        logging.debug("[토큰생성기] 모든 버퍼 처리 완료. 종료합니다.")
                        return

                    # ================== 핵심 수정 부분 시작 ==================
                    text_to_process = ""
                    
                    # 리셋 신호 처리
                    if isinstance(item, tuple) and len(item) == 2 and item[0] == "RESET":
                        logging.warning("[토큰생성기] RESET 신호 수신. 버퍼와 상태를 초기화합니다.")
                        buffer_text = ""
                        prev_len = 0
                        text_to_process = item[1]
                    # 이미 토큰화된 입력 처리 (기존 로직)
                    elif isinstance(item, (list, tuple)) and all(isinstance(x, int) for x in item):
                        logging.debug("[토큰생성기] << 입력 (사전토큰화) | 토큰수=%d", len(item))
                        for i in range(0, len(item), n_text):
                            chunk = item[i:i+n_text]
                            if not chunk: continue
                            chunk_idx += 1
                            emitted_chunks += 1
                            emitted_tokens_total += len(chunk)
                            preview = (safe_decode(chunk) or "").replace("\n", " ")
                            if len(preview) > 60: preview = preview[:60] + "..."
                            logging.debug(
                                "[토큰생성기] >> 출력(사전토큰화) 청크#%d | 길이=%d | 총출력토큰=%d | 내용='%s'",
                                chunk_idx, len(chunk), emitted_tokens_total, preview
                            )
                            yield torch.tensor([chunk], dtype=torch.int64, device=device)
                        continue
                    # 일반 텍스트(델타) 입력
                    else:
                        text_to_process = str(item)
                    # ================== 핵심 수정 부분 종료 ==================

                    # 문자열이면 버퍼에 누적 후, 증가분만 n개 단위로 배출
                    buffer_text += text_to_process
                    #logging.debug("[토큰생성기] << 입력 (텍스트) | 추가글자수=%d | 버퍼총글자수=%d", len(text_to_process), len(buffer_text))

                    ids = safe_encode(buffer_text)
                    delta = ids[prev_len:]
                    delta_len = len(delta)

                    if delta_len >= n_text:
                        emit_len = (delta_len // n_text) * n_text
                        to_emit = delta[:emit_len]
                        logging.debug(
                            "[토큰생성기] 출력 준비 | 신규토큰=%d -> 출력할토큰=%d (단위=%d) | 이전길이=%d 총길이=%d",
                            delta_len, emit_len, n_text, prev_len, len(ids)
                        )
                        for i in range(0, emit_len, n_text):
                            chunk = to_emit[i:i+n_text]
                            if not chunk: continue
                            chunk_idx += 1
                            emitted_chunks += 1
                            emitted_tokens_total += len(chunk)
                            preview = (safe_decode(chunk) or "").replace("\n", " ")
                            if len(preview) > 60: preview = preview[:60] + "..."
                            logging.debug(
                                "[토큰생성기] >> 출력 청크#%d | 길이=%d | 총출력토큰=%d | 내용='%s'",
                                chunk_idx, len(chunk), emitted_tokens_total, preview
                            )
                            yield torch.tensor([chunk], dtype=torch.int64, device=device)
                        prev_len += emit_len
                    else:
                        logging.debug(
                            "[토큰생성기] 대기 (토큰 부족) | 신규토큰=%d < 단위=%d (이전길이=%d 총길이=%d)",
                            delta_len, n_text, prev_len, len(ids)
                        )
            
            # (5) 모델 입력의 'text' 부분에 위에서 만든 제너레이터 연결
            model_input["text"] = to_token_chunks()
            logging.debug("[타이핑] 텍스트 제너레이터 연결 완료 | 청크토큰수=%d", n_text)

            # (6) 스트리밍 합성 시작
            start_time = time.time()
            out_chunks = 0
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                wav = model_output["tts_speech"]
                out_chunks += 1
                samples = int(wav.shape[1])
                secs = samples / float(self.sample_rate)
                rtf = (time.time() - start_time) / max(secs, 1e-3)
                logging.info(
                    "[타이핑-TTS] 오디오 출력 청크#%d | 샘플수=%d 길이=%.2f초 RTF=%.3f | 입력된_텍스트청크수=%d 총토큰수=%d",
                    out_chunks, samples, secs, rtf, emitted_chunks, emitted_tokens_total
                )
                yield model_output
                start_time = time.time()