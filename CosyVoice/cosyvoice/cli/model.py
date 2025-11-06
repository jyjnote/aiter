# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
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
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
import logging

class CosyVoiceModel:
    # CosyVoice 모델 클래스
    # LLM, Flow, HiFT 모듈을 포함하고, TTS 기능을 제공
    # 스트리밍 및 비스트리밍 모드 지원
    def __init__(self,
                 llm: torch.nn.Module,# LLM 모듈
                 flow: torch.nn.Module,# Flow 모듈
                 hift: torch.nn.Module,# HiFT 모듈
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 사용 가능한 장치 설정
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True: # fp16 모드 설정 시, 모델을 half precision으로 변환, 메모리 절약 및 속도 향상, 단, 정확도 저하 가능성 있음, half precision은 16비트 부동소수점 형식
            self.llm.half() # LLM 모델을 half precision으로 변환
            self.flow.half()# Flow 모델을 half precision으로 변환
        self.token_min_hop_len = 2 * self.flow.input_frame_rate # 토큰 최소 홉 길이 설정
        self.token_max_hop_len = 4 * self.flow.input_frame_rate # 토큰 최대 홉 길이 설정 ,홉이란 신호 처리에서 한 프레임에서 다음 프레임으로 이동하는 간격을 의미
        self.token_overlap_len = 20 # 토큰 오버랩 길이 설정
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20 # mel 캐시 길이 설정
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {} # 세션별 생성된 음성 토큰 저장 딕셔너리
        self.llm_end_dict = {} # 세션별 LLM 종료 상태 저장 딕셔너리
        self.mel_overlap_dict = {} # 세션별 mel 오버랩 저장 딕셔너리
        self.flow_cache_dict = {}# 세션별 Flow 캐시 저장 딕셔너리
        self.hift_cache_dict = {}# 세션별 HiFT 캐시 저장 딕셔너리

    def load(self, llm_model, flow_model, hift_model): # 모델 가중치 로드 함수
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True) # LLM 모델 가중치 로드
        self.llm.to(self.device).eval()# LLM 모델을 장치로 이동하고 평가 모드로 설정
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()} 
        # HiFT 모델 가중치 로드, HiFiGAN 형식일 경우 키 이름 변경
        # 'generator.' 접두사를 제거하여 키 이름 변경
        # 예: 'generator.conv1.weight' -> 'conv1.weight'
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        # JIT 모델 로드 함수
        # JIT(Just-In-Time) 컴파일된 모델을 로드하여 추론 속도 향상
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        # TensorRT 모델 로드 함수
        # TensorRT는 NVIDIA의 딥러닝 추론 최적화 라이브러리
        # TensorRT로 변환된 모델을 로드하여 추론 속도 향상
        # TensorRT는 FP16 및 INT8과 같은 저정밀도 연산을 지원하여 성능을 극대화
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        # TensorRT 변환을 위한 입력 크기 및 이름 설정 함수
        # 최소, 최적, 최대 입력 크기 및 입력 이름을 딕셔너리로 반환
        # TensorRT 변환 시 모델의 입력 크기를 미리 정의해야 함
        # 다양한 길이의 입력을 처리할 수 있도록 최소, 최적, 최대 크기를 설정
        # 입력 이름은 모델의 입력 텐서 이름과 일치해야 함
        # 예: ONNX 모델에서 입력 텐서 이름이 'input'인 경우, 입력 이름도 'input'으로 설정
        # 여기서는 Flow 모델의 입력 크기 및 이름을 설정
        # Flow 모델의 입력 크기는 (배치 크기, 채널 수, 시퀀스 길이) 형식
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
            """
            LLM 단계에서 입력 텍스트(Generator or Tensor)를 받아 speech tokens을 생성하고,
            세션(uuid) 별 딕셔너리에 누적 저장하면서 디버그 로그를 출력한다.
            Args:
                text (Generator or Tensor): 입력 텍스트, 제너레이터 또는 텐서 형식
                prompt_text (Tensor): 프롬프트 텍스트 텐서
                llm_prompt_speech_token (Tensor): 프롬프트 음성 토큰 텐서
                llm_embedding (Tensor): 임베딩 텐서
                uuid (str): 세션 식별자
            Returns:
                None

            Note:
                - 스트리밍 입력 모드(Generator)와 일반 입력 모드(Tensor)를 구분하여 처리한다.
                - 스트리밍 입력은 CosyVoice2에서만 지원하며, vllm과는 호환되지 않는다.
                - 비스트리밍 모드에서는 입력 텍스트 정보를 미리 로깅한다.
                - 최종적으로 생성된 모든 음성 토큰을 로그에 출력한다.
                - LLM 작업이 완료되면 llm_end_dict[uuid]를 True로 설정한다.
            Logging:
                - 입력 모드, 원본 텍스트, 토큰 길이, 생성된 토큰 길이 및 비율을 로그에 기록한다.
                - 최종적으로 생성된 모든 음성 토큰을 로그에 기록한다.
                - LLM 작업 완료 시 로그에 기록한다.
            
            """
            # 스트리밍 모드 여부를 판단하고, 로깅에 사용할 변수 초기화
            is_streaming_input = isinstance(text, Generator) # 입력이 제너레이터면 스트리밍 모드
            text_ids, raw_text = None, None # 비스트리밍 모드에서 디코딩된 원본 텍스트 저장용 변수

            with self.llm_context, torch.cuda.amp.autocast( # 자동 혼합 정밀도 컨텍스트 매니저
                # fp16 모드이면서 vllm이 아닌 경우에만 활성화, vllm은 별도의 fp16 처리가 필요할 수 있음, vllm은 매우 큰 언어 모델을 효율적으로 실행하기 위한 라이브러리
                # fp16 모드에서 메모리 사용량을 줄이고 성능을 향상시킬 수 있음
                self.fp16 is True and hasattr(self.llm, 'vllm') is False
            ):
                # ========================
                # 1) 스트리밍 입력 모드
                # ========================
                if is_streaming_input:
                    assert isinstance(self, CosyVoice2Model) and not hasattr(self.llm, 'vllm'), \
                        'streaming input text is only implemented for CosyVoice2 and do not support vllm!'
                        # 스트리밍 입력은 CosyVoice2에서만 지원, vllm과는 호환되지 않음

                    logging.info(f"[INPUT] uuid={uuid} | Streaming (Generator) input detected.")
                    
                    for i in self.llm.inference_bistream( # 스트리밍 입력 처리
                        text=text, # 제너레이터에서 텍스트 조각을 하나씩 가져옴
                        prompt_text=prompt_text.to(self.device), # 프롬프트 텍스트를 장치로 이동
                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),# 프롬프트 텍스트 길이를 텐서로 변환하여 장치로 이동
                        prompt_speech_token=llm_prompt_speech_token.to(self.device),# 프롬프트 음성 토큰을 장치로 이동
                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),# 프롬프트 음성 토큰 길이를 텐서로 변환하여 장치로 이동
                        embedding=llm_embedding.to(self.device)# 임베딩을 장치로 이동
                    ):
                        self.tts_speech_token_dict[uuid].append(i) # 생성된 음성 토큰을 세션별 딕셔너리에 누적 저장

                # ========================
                # 2) 일반 입력 모드
                # ========================
                else:
                    # 비스트리밍 모드에서는 입력 텍스트 정보를 미리 로깅
                    try:
                        text_ids = text.tolist()
                    except Exception:
                        text_ids = str(text)

                    try:
                        tok = get_qwen_tokenizer(
                            token_path="pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN",
                            skip_special_tokens=True
                        )
                        if isinstance(text_ids, list) and len(text_ids) > 0 and isinstance(text_ids[0], list):
                            raw_text = tok.decode(text_ids[0])
                        else:
                            raw_text = tok.decode(text_ids)
                    except Exception as e:
                        raw_text = f"<decode error: {e}>"

                    logging.info(f"[INPUT] uuid={uuid} | Non-streaming (Tensor) input.")
                    logging.info(f"   raw text   : {raw_text}")
                    logging.info(f"   text_ids   : {text_ids}")

                    for i in self.llm.inference(
                        text=text.to(self.device),
                        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_text=prompt_text.to(self.device),
                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                        embedding=llm_embedding.to(self.device),
                        uuid=uuid
                    ):
                        self.tts_speech_token_dict[uuid].append(i) # 생성된 음성 토큰을 세션별 딕셔너리에 누적 저장

            # --- 최종 로그 출력 (스트리밍/비스트리밍 공통) ---
            final_tokens = []
            for tok in self.tts_speech_token_dict.get(uuid, []):
                if isinstance(tok, torch.Tensor):
                    final_tokens.extend(tok.cpu().tolist())
                elif isinstance(tok, (list, tuple)):
                    final_tokens.extend(tok)
                else:
                    final_tokens.append(int(tok))
            
            final_speech_token_len = len(final_tokens)

            if not is_streaming_input:
                input_text_token_len = text.shape[1]
                ratio = final_speech_token_len / input_text_token_len if input_text_token_len > 0 else 0
                
                logging.info(f"[FINAL-MAP | Non-streaming] uuid={uuid}")
                #logging.info(f"   raw text          : {raw_text}")
                logging.info(f"   input_token_len   : {input_text_token_len}")
                logging.info(f"   speech_token_len  : {final_speech_token_len}")
                logging.info(f"   Ratio (speech/input): {ratio:.2f}")
                # 전체 토큰 로그가 너무 길면 터미널이 느려질 수 있으므로, 필요 시 주석 처리
                logging.info(f"   -> all speech tokens which are generated by the model: {final_tokens}")
            else:
                # 스트리밍 모드에서는 입력 길이를 알 수 없으므로, 생성된 토큰 길이만 출력
                logging.info(f"[FINAL-MAP | Streaming] uuid={uuid}")
                logging.info(f"   speech_token_len  : {final_speech_token_len}")
                logging.info(f"   -> all speech tokens which are generated by the model: {final_tokens}")

            # ========================
            # 3) LLM 종료 플래그
            # ========================
            self.llm_end_dict[uuid] = True # LLM 작업이 완료되었음을 표시
            logging.info(
                f"[LLM-END] uuid={uuid} | LLM finished "
                f"| Total Speech Tokens={final_speech_token_len}"
            )


    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist() # 레퍼런스 음성 추출 
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device),
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_token=prompt_token.to(self.device),
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_feat=prompt_feat.to(self.device),
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                      embedding=embedding.to(self.device),
                                                                      flow_cache=self.flow_cache_dict[uuid])

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode' 
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)

            logging.info(f"[HiFiGAN-FINAL] | token2wav uuid={uuid} | mel_len={tts_mel.shape[2]} | wav_len={tts_speech.shape[1]} | speed={speed}")

            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    # def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
    #         prompt_text=torch.zeros(1, 0, dtype=torch.int32),
    #         llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
    #         flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
    #         prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
    #     # this_uuid is used to track variables related to this inference thread
    #     this_uuid = str(uuid.uuid1())
    #     with self.lock:
    #         self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
    #         self.hift_cache_dict[this_uuid] = None
    #         self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
    #         self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
    #     if source_speech_token.shape[1] == 0:
    #         p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
    #     else:
    #         p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
    #     p.start()
    #     if stream is True:
    #         token_hop_len = self.token_min_hop_len
    #         while True:
    #             time.sleep(0.1)
    #             if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
    #                 this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
    #                     .unsqueeze(dim=0)
    #                 this_tts_speech = self.token2wav(token=this_tts_speech_token,
    #                                                  prompt_token=flow_prompt_speech_token,
    #                                                  prompt_feat=prompt_speech_feat,
    #                                                  embedding=flow_embedding,
    #                                                  uuid=this_uuid,
    #                                                  finalize=False)
    #                 yield {'tts_speech': this_tts_speech.cpu()}
    #                 with self.lock:
    #                     self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
    #                 # increase token_hop_len for better speech quality
    #                 token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
    #             if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
    #                 break
    #         p.join()
    #         # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
    #         this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
    #         this_tts_speech = self.token2wav(token=this_tts_speech_token,
    #                                          prompt_token=flow_prompt_speech_token,
    #                                          prompt_feat=prompt_speech_feat,
    #                                          embedding=flow_embedding,
    #                                          uuid=this_uuid,
    #                                          finalize=True)
    #         yield {'tts_speech': this_tts_speech.cpu()}
    #     else:
    #         # deal with all tokens
    #         p.join()
    #         this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
    #         this_tts_speech = self.token2wav(token=this_tts_speech_token,
    #                                          prompt_token=flow_prompt_speech_token,
    #                                          prompt_feat=prompt_speech_feat,
    #                                          embedding=flow_embedding,
    #                                          uuid=this_uuid,
    #                                          finalize=True,
    #                                          speed=speed)
    #         yield {'tts_speech': this_tts_speech.cpu()}
    #     with self.lock:
    #         self.tts_speech_token_dict.pop(this_uuid)
    #         self.llm_end_dict.pop(this_uuid)
    #         self.mel_overlap_dict.pop(this_uuid)
    #         self.hift_cache_dict.pop(this_uuid)
    #         self.flow_cache_dict.pop(this_uuid)
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #         torch.cuda.current_stream().synchronize()
            
    # [V1] 즉시 처리 로직 (이전 코드와 동일)
    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
                prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
            # this_uuid is used to track variables related to this inference thread
            this_uuid = str(uuid.uuid1())
            with self.lock:
                self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
                self.hift_cache_dict[this_uuid] = None
                self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
                self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
            if source_speech_token.shape[1] == 0:
                p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
            else:
                p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
            p.start()
            
            # ================================================================
            # [핵심 수정] 스트리밍 로직 변경
            # ================================================================
            if stream is True:
                # 처리된 토큰의 인덱스를 추적
                processed_tokens_len = 0
                while True:
                    # 0.02초마다 토큰 창고를 확인 (time.sleep(0.1) -> 0.02)
                    time.sleep(0.02)
                    with self.lock:
                        current_tokens_len = len(self.tts_speech_token_dict[this_uuid])
                    
                    # [핵심 수정] 새로 생성된 토큰이 1개라도 있는지 확인
                    if current_tokens_len > processed_tokens_len:
                        # 새로 들어온 모든 토큰을 가져옴
                        with self.lock:
                            new_tokens = self.tts_speech_token_dict[this_uuid][processed_tokens_len:]
                        
                        this_tts_speech_token = torch.tensor(new_tokens).unsqueeze(dim=0)

                        # [주의] finalize=True로 설정하여 매번 독립적인 오디오 조각 생성
                        # 이렇게 하면 오버랩 로직이 비활성화되어 품질이 저하될 수 있음
                        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                        prompt_token=flow_prompt_speech_token,
                                                        prompt_feat=prompt_speech_feat,
                                                        embedding=flow_embedding,
                                                        uuid=this_uuid,
                                                        finalize=True) # finalize=True로 변경
                        
                        if this_tts_speech.numel() > 0:
                            yield {'tts_speech': this_tts_speech.cpu()}
                        
                        # 처리된 토큰 길이를 업데이트
                        processed_tokens_len = current_tokens_len

                    # LLM 스레드가 종료되었고, 모든 토큰을 처리했으면 루프 탈출
                    if self.llm_end_dict[this_uuid] is True and current_tokens_len == processed_tokens_len:
                        break
                p.join()
            # ================================================================
            # 비스트리밍 로직은 기존과 동일
            # ================================================================
            else:
                # deal with all tokens
                p.join()
                this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                prompt_token=flow_prompt_speech_token,
                                                prompt_feat=prompt_speech_feat,
                                                embedding=flow_embedding,
                                                uuid=this_uuid,
                                                finalize=True,
                                                speed=speed)
                yield {'tts_speech': this_tts_speech.cpu()}

            # 세션 정리
            with self.lock:
                self.tts_speech_token_dict.pop(this_uuid)
                self.llm_end_dict.pop(this_uuid)
                self.mel_overlap_dict.pop(this_uuid)
                self.hift_cache_dict.pop(this_uuid)
                self.flow_cache_dict.pop(this_uuid)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.current_stream().synchronize()

class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 15 # 15 for CosyVoice2-0.5B, 10 for CosyVoice2-3B
        
        # --- [수정] Greedy 정책을 위한 최대 홉 개수 설정 ---
        # 한 번에 최대 3개 홉(hop)까지 처리 (더 "시원하게")
        self.token_max_hop_count = 3
        # --- [끝] ---
        
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=model_dir,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.2)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                # [핵심] 여기가 "내부에서 붙이는" 로직입니다.
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                # [핵심] 마지막 조각도 "내부에서 붙입니다."
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech


    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        
        # [수정 1] 세션 ID를 kwargs에서 가져오고, 세션 지속 여부를 확인합니다.
        this_uuid = kwargs.get("session_id", str(uuid.uuid1()))
        is_persistent_session = "session_id" in kwargs

        # [수정 2 - 버그 수정]
        # 캐시 재사용 여부 판단 기준을 tts_speech_token_dict가 아닌 hift_cache_dict로 변경
        with self.lock:
            if not is_persistent_session or this_uuid not in self.hift_cache_dict: # <-- [핵심] hift_cache_dict 기준으로 확인
                logging.info(f"[{this_uuid}] Initializing new cache (HiFiGAN cache not found).")
                self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
                self.hift_cache_dict[this_uuid] = None # <-- HiFiGAN 캐시 초기화
            else:
                logging.info(f"[{this_uuid}] Re-using existing HiFiGAN cache.")
                # LLM 관련 캐시만 초기화 (HiFiGAN 캐시는 유지)
                self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
        
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        
        # ================================================================
        # (이하 "탐욕적(Greedy) 청킹" 정책 코드는 수정 없이 동일합니다)
        # ================================================================
        if stream is True:
            token_offset = 0
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
            
            while True:
                # 0.05초마다 버퍼 확인
                time.sleep(0.05)
                
                with self.lock:
                    current_total_len = len(self.tts_speech_token_dict[this_uuid])
                llm_is_done = self.llm_end_dict[this_uuid]

                # --- 1. 변수 계산 ---
                base_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                available_len = current_total_len - token_offset

                # [필수 조건] 1개 홉 처리에 '최소'로 필요한 토큰 수
                min_required_len = base_hop_len + self.flow.pre_lookahead_len
                
                hops_to_process = 0

                # --- 2. Greedy 정책 적용 ---
                if available_len >= min_required_len:
                    available_for_extra_hops = available_len - min_required_len
                    extra_hops = available_for_extra_hops // self.token_hop_len
                    hops_to_process = 1 + extra_hops
                    hops_to_process = min(hops_to_process, self.token_max_hop_count)
                
                # --- 3. 종료 조건 ---
                if llm_is_done and available_len < min_required_len:
                    break

                # --- 4. 오디오 청크 처리 ---
                if hops_to_process > 0:
                    total_hop_len_to_process = base_hop_len + (hops_to_process - 1) * self.token_hop_len
                    total_tokens_to_pass = token_offset + total_hop_len_to_process + self.flow.pre_lookahead_len

                    with self.lock:
                        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:total_tokens_to_pass]).unsqueeze(dim=0)
                    
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     token_offset=token_offset,
                                                     uuid=this_uuid,
                                                     stream=stream,
                                                     finalize=False) 
                    
                    token_offset += total_hop_len_to_process
                    
                    if this_tts_speech.numel() > 0:
                        yield {'tts_speech': this_tts_speech.cpu()}
            
            # --- 5. 남은 토큰 finalizing ---
            p.join()
            
            with self.lock:
                final_total_len = len(self.tts_speech_token_dict[this_uuid])
            
            if final_total_len > token_offset:
                this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                 prompt_token=flow_prompt_speech_token,
                                                 prompt_feat=prompt_speech_feat,
                                                 embedding=flow_embedding,
                                                 token_offset=token_offset,
                                                 uuid=this_uuid,
                                                 finalize=True) 
                if this_tts_speech.numel() > 0:
                    yield {'tts_speech': this_tts_speech.cpu()}
        
        # ================================================================
        # 비스트리밍 로직 (기존과 동일)
        # ================================================================
        else:
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=0,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        
        # [수정 3] 세션 정리(cleanup) 로직 수정 (안전하게 .pop(key, None) 사용)
        
        if not is_persistent_session: 
            with self.lock:
                logging.info(f"[{this_uuid}] Cleaning up non-persistent session cache.")
                self.tts_speech_token_dict.pop(this_uuid, None)
                self.llm_end_dict.pop(this_uuid, None)
                self.hift_cache_dict.pop(this_uuid, None)
        else:
            logging.info(f"[{this_uuid}] Retaining persistent session cache (HiFiGAN).")
            # LLM 관련 캐시만 정리 (HiFiGAN 캐시는 남김)
            with self.lock:
                self.tts_speech_token_dict.pop(this_uuid, None)
                self.llm_end_dict.pop(this_uuid, None)
                # hift_cache_dict는 남겨둠
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()
