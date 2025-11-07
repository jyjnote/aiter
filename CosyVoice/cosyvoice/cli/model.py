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
        self.token_min_hop_len = 6 * self.flow.input_frame_rate # 토큰 최소 홉 길이 설정 (3x increase for larger chunks)
        self.token_max_hop_len = 12 * self.flow.input_frame_rate # 토큰 최대 홉 길이 설정 (3x increase for larger chunks)
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
        self.space_boundaries_dict = {} # 세션별 공백 위치 (자연스러운 구간 끊김을 위함)

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
            space_token_positions = [] # BPE 220 (공백) 토큰의 위치를 저장

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

                    # BPE 220 (공백) 토큰의 위치 찾기
                    if isinstance(text_ids, list) and len(text_ids) > 0:
                        if isinstance(text_ids[0], list):
                            # 2D 리스트인 경우 (배치)
                            for idx, token_id in enumerate(text_ids[0]):
                                if token_id == 220:  # 공백 토큰
                                    space_token_positions.append(idx)
                        else:
                            # 1D 리스트인 경우
                            for idx, token_id in enumerate(text_ids):
                                if token_id == 220:  # 공백 토큰
                                    space_token_positions.append(idx)

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
                    logging.info(f"   space token positions (BPE 220): {space_token_positions}")

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

                # 공백 토큰 위치를 음성 토큰 인덱스로 매핑
                space_speech_boundaries = []
                if len(space_token_positions) > 0 and ratio > 0:
                    for space_pos in space_token_positions:
                        # 텍스트 토큰 위치를 음성 토큰 위치로 변환
                        speech_boundary = int((space_pos + 1) * ratio)  # +1: 공백 다음 위치
                        if speech_boundary < final_speech_token_len:
                            space_speech_boundaries.append(speech_boundary)

                    # 세션 딕셔너리에 저장
                    with self.lock:
                        self.space_boundaries_dict[uuid] = set(space_speech_boundaries)

                logging.info(f"[FINAL-MAP | Non-streaming] uuid={uuid}")
                #logging.info(f"   raw text          : {raw_text}")
                logging.info(f"   input_token_len   : {input_text_token_len}")
                logging.info(f"   speech_token_len  : {final_speech_token_len}")
                logging.info(f"   Ratio (speech/input): {ratio:.2f}")
                logging.info(f"   space_boundaries (speech token indices): {space_speech_boundaries}")
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


    def find_nearest_space_boundary(self, target_pos, uuid, search_range=50):
        """
        가장 가까운 공백 경계를 찾아 반환합니다.

        Args:
            target_pos: 목표 위치 (음성 토큰 인덱스)
            uuid: 세션 식별자
            search_range: 검색 범위 (앞뒤로 몇 개 토큰까지 검색할지)

        Returns:
            공백 경계 위치, 없으면 target_pos 반환
        """
        if uuid not in self.space_boundaries_dict or not self.space_boundaries_dict[uuid]:
            return target_pos

        boundaries = sorted(self.space_boundaries_dict[uuid])

        # target_pos 이하에서 가장 가까운 경계 찾기 (뒤로 검색)
        valid_boundaries = [b for b in boundaries if target_pos - search_range <= b <= target_pos]

        if valid_boundaries:
            # 가장 가까운 경계 선택
            return max(valid_boundaries)

        # 범위 내에 경계가 없으면 원래 위치 반환
        return target_pos

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
                self.space_boundaries_dict[this_uuid] = set()  # 공백 경계 초기화
            if source_speech_token.shape[1] == 0:
                p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
            else:
                p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
            p.start()
            
            # ================================================================
            # [적응형 청킹] 스트리밍 로직 - 빠른 시작 + 점진적으로 큰 청크
            # ================================================================
            if stream is True:
                # 첫 번째 청크는 작게 시작 (빠른 응답)
                initial_hop_len = 2 * self.flow.input_frame_rate
                token_hop_len = initial_hop_len
                chunk_count = 0

                while True:
                    # 첫 몇 청크는 짧은 간격으로 체크 (빠른 응답)
                    # 이후 청크는 긴 간격으로 체크 (효율성)
                    if chunk_count < 3:
                        time.sleep(0.05)  # 첫 3개 청크: 빠른 체크
                    else:
                        time.sleep(0.15)  # 이후: 더 많은 토큰 축적

                    if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                        # [공백 경계 탐색] 자연스러운 끊김을 위해 공백 위치에서 자르기
                        adjusted_hop_len = self.find_nearest_space_boundary(token_hop_len, this_uuid, search_range=100)

                        # 조정된 위치가 너무 작으면 (최소 절반 이상은 유지) 원래 위치 사용
                        if adjusted_hop_len < token_hop_len * 0.5:
                            adjusted_hop_len = token_hop_len

                        logging.info(f"[CHUNK] uuid={this_uuid} | chunk={chunk_count+1} | original_hop={token_hop_len} | adjusted_hop={adjusted_hop_len}")

                        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:adjusted_hop_len + self.token_overlap_len]) \
                            .unsqueeze(dim=0)
                        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                         prompt_token=flow_prompt_speech_token,
                                                         prompt_feat=prompt_speech_feat,
                                                         embedding=flow_embedding,
                                                         uuid=this_uuid,
                                                         finalize=False)
                        yield {'tts_speech': this_tts_speech.cpu()}
                        with self.lock:
                            # 조정된 길이만큼 제거
                            self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][adjusted_hop_len:]

                            # 공백 경계도 조정 (이미 처리된 토큰 제거)
                            if this_uuid in self.space_boundaries_dict:
                                self.space_boundaries_dict[this_uuid] = {
                                    pos - adjusted_hop_len for pos in self.space_boundaries_dict[this_uuid]
                                    if pos > adjusted_hop_len
                                }

                        chunk_count += 1

                        # 점진적으로 청크 크기 증가: 첫 청크 작게, 이후 점점 크게
                        if chunk_count == 1:
                            token_hop_len = 4 * self.flow.input_frame_rate  # 2번째: 2배
                        elif chunk_count == 2:
                            token_hop_len = 6 * self.flow.input_frame_rate  # 3번째: 3배
                        else:
                            # 4번째부터는 최대 크기로 증가
                            token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))

                    if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                        break
                p.join()
                # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
                this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                 prompt_token=flow_prompt_speech_token,
                                                 prompt_feat=prompt_speech_feat,
                                                 embedding=flow_embedding,
                                                 uuid=this_uuid,
                                                 finalize=True)
                yield {'tts_speech': this_tts_speech.cpu()}
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
                self.space_boundaries_dict.pop(this_uuid, None)  # 공백 경계 정리
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
        self.token_hop_len = 45 # 45 for larger chunks (3x increase from 15)

        # --- [수정] Greedy 정책을 위한 최대 홉 개수 설정 ---
        # 한 번에 최대 9개 홉(hop)까지 처리 (3x increase for 2-3x larger chunks)
        self.token_max_hop_count = 9
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
        self.space_boundaries_dict = {}  # 세션별 공백 위치 (자연스러운 구간 끊김을 위함)

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
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.space_boundaries_dict[this_uuid] = set()  # 공백 경계 초기화
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        
        # ================================================================
        # [적응형 청킹] CosyVoice2Model - 빠른 시작 + 점진적으로 큰 청크
        # ================================================================
        if stream is True:
            token_offset = 0
            chunk_count = 0

            # 동적으로 조정되는 파라미터들
            current_token_hop_len = 15  # 첫 청크는 작게 시작
            current_max_hop_count = 1   # 첫 청크는 1개 홉만 처리

            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / current_token_hop_len) * current_token_hop_len - flow_prompt_speech_token.shape[1])

            while True:
                # 첫 몇 청크는 짧은 간격으로 체크 (빠른 응답)
                if chunk_count < 3:
                    time.sleep(0.05)  # 첫 3개 청크: 빠른 체크
                else:
                    time.sleep(0.15)  # 이후: 더 많은 토큰 축적

                with self.lock:
                    current_total_len = len(self.tts_speech_token_dict[this_uuid])
                llm_is_done = self.llm_end_dict[this_uuid]

                # --- 1. 변수 계산 ---
                base_hop_len = current_token_hop_len + prompt_token_pad if token_offset == 0 else current_token_hop_len
                available_len = current_total_len - token_offset

                # [필수 조건] 1개 홉 처리에 '최소'로 필요한 토큰 수
                min_required_len = base_hop_len + self.flow.pre_lookahead_len

                hops_to_process = 0

                # --- 2. 적응형 Greedy 정책 ---
                if available_len >= min_required_len:
                    # 현재 버퍼에서 처리 가능한 홉 계산
                    available_for_extra_hops = available_len - min_required_len
                    extra_hops = available_for_extra_hops // current_token_hop_len
                    hops_to_process = 1 + extra_hops

                    # 동적으로 조정되는 최대 홉 개수 제한
                    hops_to_process = min(hops_to_process, current_max_hop_count)

                # --- 3. 종료 조건 ---
                if llm_is_done and available_len < min_required_len:
                    break

                # --- 4. 오디오 청크 처리 ---
                if hops_to_process > 0:
                    total_hop_len_to_process = base_hop_len + (hops_to_process - 1) * current_token_hop_len

                    # [공백 경계 탐색] 자연스러운 끊김을 위해 공백 위치에서 자르기
                    target_cut_point = token_offset + total_hop_len_to_process
                    adjusted_cut_point = self.find_nearest_space_boundary(target_cut_point, this_uuid, search_range=150)

                    # 조정된 위치가 너무 작으면 (최소 절반 이상은 유지) 원래 위치 사용
                    if adjusted_cut_point < target_cut_point * 0.5:
                        adjusted_cut_point = target_cut_point

                    adjusted_hop_len = adjusted_cut_point - token_offset

                    logging.info(f"[CHUNK-V2] uuid={this_uuid} | chunk={chunk_count+1} | original_hop={total_hop_len_to_process} | adjusted_hop={adjusted_hop_len}")

                    total_tokens_to_pass = token_offset + adjusted_hop_len + self.flow.pre_lookahead_len

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

                    token_offset += adjusted_hop_len
                    chunk_count += 1

                    # --- 5. 점진적으로 파라미터 증가 ---
                    if chunk_count == 1:
                        current_token_hop_len = 25  # 2번째: 약 1.7배
                        current_max_hop_count = 2   # 최대 2개 홉
                    elif chunk_count == 2:
                        current_token_hop_len = 35  # 3번째: 약 2.3배
                        current_max_hop_count = 3   # 최대 3개 홉
                    elif chunk_count >= 3:
                        current_token_hop_len = 45  # 4번째부터: 최대 크기
                        current_max_hop_count = 9   # 최대 9개 홉

                    if this_tts_speech.numel() > 0:
                        yield {'tts_speech': this_tts_speech.cpu()}
            
            # --- 5. 남은 토큰 finalizing ---
            p.join()
            
            with self.lock:
                final_total_len = len(self.tts_speech_token_dict[this_uuid])
            
            # 아직 처리되지 않은 토큰이 남아있는 경우
            if final_total_len > token_offset:
                this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                 prompt_token=flow_prompt_speech_token,
                                                 prompt_feat=prompt_speech_feat,
                                                 embedding=flow_embedding,
                                                 token_offset=token_offset,
                                                 uuid=this_uuid,
                                                 finalize=True) # [중요] 마지막은 True
                if this_tts_speech.numel() > 0:
                    yield {'tts_speech': this_tts_speech.cpu()}
        
        # ================================================================
        # 비스트리밍 로직 (기존과 동일)
        # ================================================================
        else:
            # deal with all tokens
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
        
        # 세션 정리
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.space_boundaries_dict.pop(this_uuid, None)  # 공백 경계 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()

    # (V2 모델의 주석 처리된 이전 tts 메서드 - 참고용)
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
    #     if source_speech_token.shape[1] == 0:
    #         p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
    #     else:
    #         p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
    #     p.start()
    #     if stream is True:
    #         token_offset = 0
    #         prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
    #         while True:
    #             time.sleep(0.1)
    #             this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
    #             if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.flow.pre_lookahead_len:
    #                 this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
    #                 this_tts_speech = self.token2wav(token=this_tts_speech_token,
    #                                                  prompt_token=flow_prompt_speech_token,
    #                                                  prompt_feat=prompt_speech_feat,
    #                                                  embedding=flow_embedding,
    #                                                  token_offset=token_offset,
    #                                                  uuid=this_uuid,
    #                                                  stream=stream,
    #                                                  finalize=False)
    #                 token_offset += this_token_hop_len
    #                 yield {'tts_speech': this_tts_speech.cpu()}
    #             if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.flow.pre_lookahead_len:
    #                 break
    #         p.join()
    #         # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
    #         this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
    #         this_tts_speech = self.token2wav(token=this_tts_speech_token,
    #                                          prompt_token=flow_prompt_speech_token,
    #                                          prompt_feat=prompt_speech_feat,
    #                                          embedding=flow_embedding,
    #                                          token_offset=token_offset,
    #                                          uuid=this_uuid,
    #                                          finalize=True)
    #         yield {'tts_speech': this_tts_speech.cpu()}
    #     else:
    #         # (비스트리밍 로직)
    #     with self.lock:
    #         # (세션 정리)