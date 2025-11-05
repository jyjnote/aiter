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
        if self.fp16 is True: # fp16 모드 설정 시, 모델을 half precision으로 변환, 메모리 절약 및 속도 향상
            self.llm.half() # LLM 모델을 half precision으로 변환
            self.flow.half()# Flow 모델을 half precision으로 변환
        self.token_min_hop_len = 2 * self.flow.input_frame_rate # 토큰 최소 홉 길이 설정
        self.token_max_hop_len = 4 * self.flow.input_frame_rate # 토큰 최대 홉 길이 설정
        self.token_overlap_len = 20 # 토큰 오버랩 길이 설정
        # mel fade in out
        # mel overlap length 계산
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
        
        # [!!! V3 수정 !!!] 신호 딕셔너리 추가
        self.llm_text_trigger_dict = {}

    def load(self, llm_model, flow_model, hift_model): # 모델 가중치 로드 함수
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True) # LLM 모델 가중치 로드
        self.llm.to(self.device).eval()# LLM 모델을 장치로 이동하고 평가 모드로 설정
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()} 
        # HiFT 모델 가중치 로드, HiFiGAN 형식일 경우 키 이름 변경
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        # JIT 모델 로드 함수
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        # TensorRT 모델 로드 함수
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
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)] # 최소 입력 크기 설정
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)] #  최적 입력 크기 설정
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)] # 최대 입력 크기 설정
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}


    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
            """
            [!!! 수정됨 v19 (Device Fix) !!!]
            - 'bistream'을 사용하지 않는 V18 '직렬' 방식을 유지합니다.
            - [버그 수정] V18에서 'torch.concat'시 발생한 'cpu/cuda' 디바이스
                         불일치 에러를 해결합니다.
            - 'current_prompt_text'와 'current_prompt_speech_token'을
              concat할 때, 양쪽 텐서 모두에 '.to(self.device)'를
              명시적으로 호출하여 디바이스를 통일시킵니다.
            """

            thread_name = threading.current_thread().name
            logging.info(f"[LLM-START-V19] uuid={uuid} | Thread Name: {thread_name} | LLM 스레드 시작 (직렬 방식).")

            is_streaming_input = isinstance(text, Generator) 
            
            # [V19] llm.inference에 전달할 '누적' 변수 초기화
            # (중요) llm_job에 전달된 초기 프롬프트는 수정하지 않도록 복사본 사용
            current_prompt_text = prompt_text.clone()
            current_prompt_speech_token = llm_prompt_speech_token.clone()

            with self.llm_context, torch.cuda.amp.autocast( 
                self.fp16 is True and hasattr(self.llm, 'vllm') is False
            ):
                # ========================
                # 1) 스트리밍 입력 모드 (V19 핵심 수정)
                # ========================
                if is_streaming_input:
                    logging.info(f"[INPUT-V19] uuid={uuid} | Streaming (Generator) input detected.")
                    logging.info(f"[INPUT-V19] 'inference_bistream' 대신 'inference'를 직렬 호출합니다.")

                    # [V19] 'frontend.py'(V18/V19)가 이제 '단어 청크' 텐서를 yield합니다.
                    chunk_count = 0

                    for text_chunk_tensor in text: # e.g., tensor([[57026, 132264, 220]])
                        chunk_count += 1
                        logging.info(f"[LLM-JOB-V19] uuid={uuid} | --- Chunk {chunk_count} 처리 시작 ---")
                        
                        # [V19] 'inference_bistream' 대신 'inference'를 호출합니다.
                        # 'inference'는 Generator이므로, 반환된 토큰을 루프로 받습니다.
                        
                        # [V19] 'inference'가 yield한 '새로운' 토큰을 저장할 임시 리스트
                        newly_generated_tokens = []

                        for i in self.llm.inference(
                            text=text_chunk_tensor.to(self.device), # 현재 청크 (e.g. "문장을 ")
                            text_len=torch.tensor([text_chunk_tensor.shape[1]], dtype=torch.int32).to(self.device),
                            
                            # [V19] 누적된 프롬프트 전달
                            prompt_text=current_prompt_text.to(self.device), # 이전까지의 모든 텍스트 (e.g. "<s>...여기에 ")
                            prompt_text_len=torch.tensor([current_prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                            prompt_speech_token=current_prompt_speech_token.to(self.device), # 이전까지의 모든 음성
                            prompt_speech_token_len=torch.tensor([current_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                            
                            embedding=llm_embedding.to(self.device),
                            uuid=uuid
                        ):
                            # [V19] i는 개별 토큰 ID(int 또는 0-dim tensor)일 수 있음
                            token_item = i.item() if isinstance(i, torch.Tensor) else int(i)
                            self.tts_speech_token_dict[uuid].append(token_item) # [V19] 메인 버퍼에 즉시 추가
                            newly_generated_tokens.append(token_item)          # [V19] 누적용 임시 버퍼에 추가

                        logging.info(f"[LLM-JOB-V19] uuid={uuid} | --- Chunk {chunk_count} 처리 완료 ---")
                        logging.info(f"[LLM-JOB-V19] | 이번 청크에서 {len(newly_generated_tokens)}개 토큰 생성됨.")
                        logging.info(f"[LLM-JOB-V19] | 현재 총 토큰 수: {len(self.tts_speech_token_dict[uuid])}")

                        # [V19] '직렬' 방식이므로, '한 청크' 합성이 끝나면
                        # 'tts' 스레드에게 합성이 끝났다고 '즉시' 알려줘야 합니다.
                        with self.lock:
                            current_speech_len = len(self.tts_speech_token_dict[uuid])
                            self.llm_text_trigger_dict[uuid].append(current_speech_len)
                            logging.info(f"[LLM-SIGNAL-V19] uuid={uuid} | Chunk {chunk_count} 완료. '직렬' 신호 전송. Speech len: {current_speech_len}")

                        # [V19] 다음 'inference' 호출을 위해 프롬프트(상태)를 수동 누적
                        
                        # [!!! V19 버그 수정 !!!]
                        # 양쪽 텐서 모두 .to(self.device)를 명시적으로 호출
                        current_prompt_text = torch.concat([
                            current_prompt_text.to(self.device), 
                            text_chunk_tensor.to(self.device)
                        ], dim=1)
                        
                        if newly_generated_tokens:
                            new_tokens_tensor = torch.tensor([newly_generated_tokens], dtype=torch.int32).to(self.device)
                            
                            # [!!! V19 버그 수정 !!!]
                            # 양쪽 텐서 모두 .to(self.device)를 명시적으로 호출
                            current_prompt_speech_token = torch.concat([
                                current_prompt_speech_token.to(self.device), 
                                new_tokens_tensor.to(self.device)
                            ], dim=1)


                # ========================
                # 2) 일반 입력 모드 (V19: 변경 없음)
                # ========================
                else:
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

                    logging.info(f"[INPUT-V19] uuid={uuid} | Non-streaming (Tensor) input.")
                    logging.info(f"   raw text     : {raw_text}")
                    logging.info(f"   text_ids     : {text_ids}")

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
                        self.tts_speech_token_dict[uuid].append(i)

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
                logging.info(f"   input_token_len   : {input_text_token_len}")
                logging.info(f"   speech_token_len  : {final_speech_token_len}")
                logging.info(f"   Ratio (speech/input): {ratio:.2f}")
            else:
                logging.info(f"[FINAL-MAP | Streaming] uuid={uuid}")
                logging.info(f"   speech_token_len  : {final_speech_token_len}")

            # ========================
            # 3) LLM 종료 플래그
            # ========================
            self.llm_end_dict[uuid] = True 
            logging.info(
                f"[LLM-END-V19] uuid={uuid} | 성공적으로 LLM 스레드의 작업이 완료되었습니다. "
                f"| Total Speech Tokens={final_speech_token_len}"
            )


    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist() # 레퍼런스 음성 추출 
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        '''
            (CosyVoiceModel의 token2wav, V1용)
        '''
        with torch.cuda.amp.autocast(self.fp16): # 자동 혼합 정밀도 컨텍스트 매니저
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
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:] # mel 오버랩 부분 저장
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len] # mel 오버랩 부분 제외

            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)

            if self.hift_cache_dict[uuid] is not None: # 이전에 저장된 HiFT 캐시가 있는 경우
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window) 
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len] # 오디오 신호 오버랩 부분 제외


        else: # finalize가 True인 경우
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode' 
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)

            logging.info(f"[HiFiGAN-FINAL] | token2wav uuid={uuid} | mel_len={tts_mel.shape[2]} | wav_len={tts_speech.shape[1]} | speed={speed}")

            if self.hift_cache_dict[uuid] is not None: # 이전에 저장된 HiFT 캐시가 있는 경우
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window) 
        return tts_speech


    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
                prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
            # (CosyVoiceModel의 tts, V1용)
            
            this_uuid = str(uuid.uuid1())

            logging.info("==========================================================")
            logging.info("========== 🚀 CosyVoiceModel.tts (V1) CALLED 🚀 ==========")
            logging.info("==========================================================")

            with self.lock:
                self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
                self.hift_cache_dict[this_uuid] = None
                self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
                self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
                
                # [!!! V3 수정 !!!] V1 모델은 신호 로직을 사용하지 않으므로,
                # V2에서만 사용하더라도 키 에러를 방지하기 위해 빈 리스트로 초기화합니다.
                # (또는 V1의 llm_job은 신호를 안 보내므로 이 줄은 없어도 무방하나, 안전을 위해 추가)
                self.llm_text_trigger_dict[this_uuid] = [] 
                
            if source_speech_token.shape[1] == 0:
                # [V19] V19 llm_job 호출
                p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
            else:
                p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
            
            logging.info(f"[THREAD-START] uuid={this_uuid} | Starting LLM thread: {p.name}")
            p.start()
            
            if stream is True:
                # (V1의 스트리밍 로직은 수정하지 않음 - 원본 코드)
                processed_tokens_len = 0
                while True:
                    time.sleep(0.02)

                    with self.lock:
                        current_tokens_len = len(self.tts_speech_token_dict[this_uuid])
                    
                    if current_tokens_len > processed_tokens_len:
                        with self.lock:
                            new_tokens = self.tts_speech_token_dict[this_uuid][processed_tokens_len:]
                        
                        this_tts_speech_token = torch.tensor(new_tokens).unsqueeze(dim=0)

                        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                        prompt_token=flow_prompt_speech_token,
                                                        prompt_feat=prompt_speech_feat,
                                                        embedding=flow_embedding,
                                                        uuid=this_uuid,
                                                        finalize=True) # finalize=True
                        
                        if this_tts_speech.numel() > 0:
                            yield {'tts_speech': this_tts_speech.cpu()}
                        
                        processed_tokens_len = current_tokens_len

                    if self.llm_end_dict[this_uuid] is True and current_tokens_len == processed_tokens_len:
                        break
                p.join()
            else:
                # 비스트리밍
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
                self.llm_text_trigger_dict.pop(this_uuid) # [!!! V3] 추가된 딕셔너리 정리
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.current_stream().synchronize()

class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        
        # [!!! 핵심 1: 부모 클래스의 __init__ 호출]
        # 이 줄이 llm_text_trigger_dict를 포함한 
        # CosyVoiceModel의 모든 변수를 초기화합니다.
        super().__init__(llm, flow, hift, fp16)

        # [!!!] 아래는 CosyVoice2Model 고유의 변수들
        # (부모와 중복되는 self.llm, self.fp16 등은 super()가 처리했으므로
        #  사실 제거해도 되지만, 원본 코드 구조 유지를 위해 그냥 둡니다.)
        
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
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        # [!!!] self.llm_context와 self.lock은 super()가 이미 생성했으므로 주석 처리
        # self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        # self.lock = threading.Lock() 
        
        # [!!! 핵심 2: 딕셔너리 중복 정의 제거]
        # 딕셔너리들은 super()가 이미 생성했으므로, 여기서 재정의하면 안 됩니다.
        # self.tts_speech_token_dict = {} # (제거)
        # self.llm_end_dict = {} # (제거)
        # self.hift_cache_dict = {} # (제거)
        
        # [!!! V18: 이 변수들은 더 이상 사용되지 않습니다 !!!]
        # llm_job (V18)이 더 이상 래퍼를 사용하지 않음
        self.space_token_id = 220
        self.space_trigger_count = 3

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
        '''
            (CosyVoice2Model의 token2wav, V2용)
        '''
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device),# 음성 토큰을 mel 스펙트로그램으로 변환
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),# 토큰 길이 텐서를 장치로 이동
                                             prompt_token=prompt_token.to(self.device),# 프롬프트 토큰을 장치로 이동
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             streaming=stream,
                                             finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:] 
        # 스트리밍 모드에서 토큰 오프셋에 해당하는 mel 부분만 선택
        # append hift cache
        if self.hift_cache_dict[uuid] is not None: # 이전에 저장된 HiFT 캐시가 있는 경우
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2) # mel 스펙트로그램에 캐시된 mel을 이어붙임
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False: # 스트리밍 모드에서 마지막 청크가 아닐 때
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None: # 이전에 저장된 HiFT 캐시가 있는 경우
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window) # 오디오 신호 오버랩 부분 페이드 인/아웃 처리
            
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:], # mel 오버랩 부분 저장
                                          'source': tts_source[:, :, -self.source_cache_len:],# source 오버랩 부분 저장
                                          'speech': tts_speech[:, -self.source_cache_len:]}# speech 오버랩 부분 저장
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else: # 스트리밍 모드에서 마지막 청크일 때
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source) # HiFT 모델을 사용하여 mel 스펙트로그램을 실제 오디오 신호로 변환
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
                prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
                
                # [1. Setup]
                this_uuid = str(uuid.uuid1())
                
                logging.info("==========================================================")
                logging.info("======= 🚀 CosyVoice2Model.tts (V19-Serial-Infer) CALLED 🚀 =======") # 버전 v19
                logging.info("==========================================================")
                
                with self.lock:
                    self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
                    self.hift_cache_dict[this_uuid] = None
                    self.llm_text_trigger_dict[this_uuid] = [] 
                
                if source_speech_token.shape[1] == 0:
                    # [V19] V19 llm_job 호출
                    p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
                else:
                    p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
                
                logging.info(f"[THREAD-START] uuid={this_uuid} | Starting LLM/VC thread: {p.name}")
                p.start()

                # ================================================================
                # [핵심 수정] V19: '직렬' 신호 처리 (V18/V12와 동일)
                # - llm_job(V19)가 '직렬'로 청크를 처리하고 '정확한' 신호를 보냅니다.
                # - V15의 'time.sleep(0.2)' 대기 로직을 '제거'합니다.
                # - V12/V7 (증분 방식) 로직으로 복귀합니다.
                # ================================================================
                if stream is True: 
                    token_offset = 0   # 오디오로 변환 완료된 *음성 토큰*의 위치
                    trigger_offset = 0 # 처리 완료된 *신호(Trigger)*의 인덱스
                    
                    while True:
                        time.sleep(0.02) # 0.02초마다 신호 확인

                        with self.lock:
                            trigger_list = self.llm_text_trigger_dict[this_uuid]
                            llm_is_done = self.llm_end_dict[this_uuid]
                            
                            process_now = False
                            process_until_len = 0 

                        if len(trigger_list) > trigger_offset:
                            # 신호가 수신됨. (V19이므로 이 길이는 100% 정확함)
                            process_until_len = trigger_list[trigger_offset]
                            
                            this_chunk_hop_len = process_until_len - token_offset
                            
                            if this_chunk_hop_len > 0:
                                process_now = True
                            else:
                                logging.warning(f"[STREAM-PROCESS-V19] uuid={this_uuid} | Signal {trigger_offset} received, but hop_len is 0. Skipping.")
                                trigger_offset += 1 
                        
                        if llm_is_done and (not process_now or trigger_offset >= len(trigger_list)):
                            break 

                        if process_now:
                            with self.lock:
                                all_speech_tokens = self.tts_speech_token_dict[this_uuid]
                                # V19: process_until_len은 정확하므로 그대로 사용
                                this_tts_speech_token = torch.tensor(all_speech_tokens[:process_until_len]).unsqueeze(dim=0)
                            
                            logging.info(f"[STREAM-PROCESS-V19] uuid={this_uuid} | Processing Trigger {trigger_offset}. Tokens from {token_offset} to {process_until_len} (hop_len={this_chunk_hop_len})...")
                            
                            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                            prompt_token=flow_prompt_speech_token,
                                                            prompt_feat=prompt_speech_feat,
                                                            embedding=flow_embedding,
                                                            token_offset=token_offset, # 증분 시작 위치
                                                            uuid=this_uuid,
                                                            stream=stream, 
                                                            # [!!! V19 (문제 2 해결)]
                                                            # 오디오 끊김(click)을 막기 위해 finalize=False
                                                            finalize=False) 
                            
                            token_offset = process_until_len # 오프셋을 처리한 최종 길이로 업데이트
                            trigger_offset += 1             # 신호 1개 소진
                            
                            if this_tts_speech.numel() > 0:
                                logging.info(f"[STREAM-YIELD-V19] uuid={this_uuid} | Yielding audio chunk. New speech_token_offset={token_offset}")
                                yield {'tts_speech': this_tts_speech.cpu()}
                    
                    # --- 4. 최종 청크 처리 (루프 종료 후) ---
                    logging.info(f"[STREAM-END-V19] uuid={this_uuid} | Loop broken. Joining thread...")
                    p.join()
                    
                    final_available_tokens = 0
                    with self.lock:
                        final_available_tokens = len(self.tts_speech_token_dict[this_uuid])
                    
                    if final_available_tokens > token_offset:
                        logging.info(f"[STREAM-FINAL-V19] uuid={this_uuid} | Processing final remaining tokens. (Offset: {token_offset} -> Total: {final_available_tokens})")
                        
                        with self.lock:
                            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
                        
                        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                        prompt_token=flow_prompt_speech_token,
                                                        prompt_feat=prompt_speech_feat,
                                                        embedding=flow_embedding,
                                                        token_offset=token_offset,
                                                        uuid=this_uuid,
                                                        finalize=True) # ★마지막★이므로 True
                        
                        if this_tts_speech.numel() > 0:
                            yield {'tts_speech': this_tts_speech.cpu()}
                    else:
                        logging.info(f"[STREAM-FINAL-V19] uuid={this_uuid} | No remaining tokens to process.")

                
                # ================================================================
                # [3. Non-Streaming Logic: 원본과 동일]
                # ================================================================
                else:
                    logging.info(f"[NON-STREAM] uuid={this_uuid} | Running in non-streaming mode. Waiting for LLM thread to join...")
                    # deal with all tokens
                    p.join()
                    logging.info(f"[NON-STREAM] uuid={this_uuid} | LLM thread joined. Processing all tokens at once.")
                    
                    with self.lock:
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
                
                # [4. Cleanup]
                logging.info(f"[CLEANUP] uuid={this_uuid} | Cleaning up session dictionaries.")
                with self.lock:
                    self.tts_speech_token_dict.pop(this_uuid)
                    self.llm_end_dict.pop(this_uuid)
                    self.hift_cache_dict.pop(this_uuid)
                    self.llm_text_trigger_dict.pop(this_uuid) 
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.current_stream().synchronize()
                logging.info(f"[TTS-END] uuid={this_uuid} | TTS job finished.")