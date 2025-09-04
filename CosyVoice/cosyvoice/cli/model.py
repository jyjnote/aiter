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
import time
from typing import Generator
import torch
import numpy as np
import threading
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
import scipy.ndimage
from collections import Counter
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper


class CosyVoiceModel:

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
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        self.speech_window = np.hamming(2 * self.source_cache_len)
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
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
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

# cosyvoice/cli/model.py 파일의 CosyVoiceModel 클래스 내부

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        from collections import Counter

        # 입력 텍스트 정보 로그 (수정 없음)
        text_ids, raw_text = None, None
        if not isinstance(text, Generator):
            try:
                text_ids = text.tolist()
            except Exception:
                text_ids = str(text)
            try:
                from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
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

        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            
            # --- 추가: BGM 유발 의심 토큰 제어 로직 ---
            # 분석을 통해 확인된 BGM 유발 가능성이 높은 토큰 ID 목록
            # 1 안녕 6162, 3927, 4754, 3514

            SUSPECT_BGM_TOKENS = {4174, 4146, 4227, 4218, 6405, 4173, 3931, 4147, 4009, 4255,6162, 3927, 4754, 3514,2058,4245,3921/
                                  4137, 2139, 4000, 4299, 1815, 3921, 4254, 3921, 6439, 2121, 870,4003}
            # 의심 토큰을 대체할, 상대적으로 안전한 토큰 ID
            SAFE_TOKEN_ID = 3921 
            # --- 추가 끝 ---

            if isinstance(text, Generator):
                # (스트리밍 로직은 복잡하므로 여기서는 비스트리밍에만 우선 적용합니다)
                for i in self.llm.inference_bistream(
                        text=text,
                        prompt_text=prompt_text.to(self.device),
                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                        embedding=llm_embedding.to(self.device)
                    ):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
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
                    
                    original_token_id = i
                    
                    # --- 추가: 의심 토큰 검사 및 강제 교체 ---
                    if original_token_id in SUSPECT_BGM_TOKENS:
                        i = SAFE_TOKEN_ID  # 문제가 되는 토큰을 안전한 토큰으로 교체
                        print(f"!!! BGM Token CLAMPED: Original={original_token_id} -> Replaced={i} !!!")
                    # --- 추가 끝 ---
                    
                    self.tts_speech_token_dict[uuid].append(i)

        self.llm_end_dict[uuid] = True
        
        # (토큰 빈도 분석 로직은 디버깅을 위해 그대로 둡니다)
        try:
            final_tokens = []
            for tok_item in self.tts_speech_token_dict.get(uuid, []):
                if isinstance(tok_item, torch.Tensor):
                    final_tokens.extend(tok_item.cpu().flatten().tolist())
                elif isinstance(tok_item, (list, tuple)):
                    final_tokens.extend(tok_item)
                else:
                    final_tokens.append(int(tok_item))

            if final_tokens:
                token_counts = Counter(final_tokens)
                most_common_tokens = token_counts.most_common(30)
                
                print("\n" + "="*20 + " TOKEN FREQUENCY ANALYSIS " + "="*20)
                if raw_text:
                    print(f"[ANALYSIS] Input Text: '{raw_text}'")
                print(f"[ANALYSIS] Total generated tokens: {len(final_tokens)}")
                print("[ANALYSIS] Top 30 most common tokens (Token ID: Count):")
                
                columns = 3
                col_width = 20
                formatted_lines = []
                for i in range(0, len(most_common_tokens), columns):
                    line_parts = []
                    for j in range(columns):
                        if i + j < len(most_common_tokens):
                            token_id, count = most_common_tokens[i+j]
                            line_parts.append(f"Token {token_id:<5}: {count:<4} times")
                    formatted_lines.append(" | ".join(part.ljust(col_width) for part in line_parts))
                
                for line in formatted_lines:
                    print(f"    {line}")

                print("="*64 + "\n")
        except Exception as e:
            print(f"[ANALYSIS-ERROR] Failed to analyze tokens for UUID {uuid}: {e}")

    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, sample_rate, finalize=False, speed=1.0, bgm_removal: bool = False, **kwargs):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device),
                                                                       token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                       prompt_token=prompt_token.to(self.device),
                                                                       prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                       prompt_feat=prompt_feat.to(self.device),
                                                                       prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                       embedding=embedding.to(self.device),
                                                                       flow_cache=self.flow_cache_dict[uuid])
        
        if finalize:
            try:
                full_token_sequence = token.cpu().flatten().tolist()
                hop_length = self.hift.hop_length if hasattr(self.hift, 'hop_length') else 256
                token_mel_ratio = self.flow.token_mel_ratio if hasattr(self.flow, 'token_mel_ratio') else 4
                time_per_mel_frame = hop_length / sample_rate

                timestamp_file = time.strftime('%Y%m%d_%H%M%S')
                analysis_filename = f"debug_token_timeline_{timestamp_file}.txt"
                
                with open(analysis_filename, 'w', encoding='utf-8') as f:
                    f.write(f"UUID: {uuid}\n")
                    f.write(f"Total Tokens: {len(full_token_sequence)}\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{'Index':<10s} | {'Token ID':<10s} | {'Start (s)':<18s} | {'End (s)':<18s}\n")
                    f.write("-" * 60 + "\n")

                    for i, token_id in enumerate(full_token_sequence):
                        start_mel_frame = i * token_mel_ratio
                        end_mel_frame = (i + 1) * token_mel_ratio
                        start_time_s = start_mel_frame * time_per_mel_frame
                        end_time_s = end_mel_frame * time_per_mel_frame
                        f.write(f"{i:<10d} | {token_id:<10d} | {start_time_s:<18.4f} | {end_time_s:<18.4f}\n")
                
                print(f"[DEBUG] Token timeline saved: {os.path.abspath(analysis_filename)}")

            except Exception as e:
                print(f"[ANALYSIS-ERROR] Failed to generate token timeline file: {e}")

        if bgm_removal:
            tts_mel_numpy = tts_mel.squeeze().cpu().numpy()
            kernel_size = (1, 3) 
            filtered_mel_numpy = scipy.ndimage.median_filter(tts_mel_numpy, size=kernel_size)
            tts_mel = torch.from_numpy(filtered_mel_numpy).unsqueeze(0).to(self.device)

        if self.mel_overlap_dict[uuid].shape[2] != 0:
             tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
            
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
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
                
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), 
            stream=False, speed=1.0, bgm_removal: bool = False, **kwargs):
        
        sample_rate = kwargs.get('sample_rate')
        if sample_rate is None:
            raise ValueError("tts method must be called with a 'sample_rate' keyword argument.")

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

        kwargs_for_token2wav = kwargs.copy()
        kwargs_for_token2wav['sample_rate'] = sample_rate

        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False,
                                                     bgm_removal=bgm_removal,
                                                     **kwargs_for_token2wav)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             bgm_removal=bgm_removal,
                                             **kwargs_for_token2wav)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed,
                                             bgm_removal=bgm_removal,
                                             **kwargs_for_token2wav)
            yield {'tts_speech': this_tts_speech.cpu()}
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
        # CosyVoiceModel의 __init__을 명시적으로 호출하여 부모 클래스의 속성을 초기화합니다.
        super().__init__(llm, flow, hift, fp16)
        
        self.token_hop_len = 25
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        self.speech_window = np.hamming(2 * self.source_cache_len)

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

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, sample_rate, stream=False, finalize=False, speed=1.0, bgm_removal: bool = False, **kwargs):
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
        
        if finalize:
            try:
                full_token_sequence = token.cpu().flatten().tolist()
                hop_length = self.hift.hop_length if hasattr(self.hift, 'hop_length') else 480 
                token_mel_ratio = self.flow.token_mel_ratio if hasattr(self.flow, 'token_mel_ratio') else 4
                time_per_mel_frame = hop_length / sample_rate

                timestamp_file = time.strftime('%Y%m%d_%H%M%S')
                analysis_filename = f"debug_token_timeline_{timestamp_file}.txt"
                
                with open(analysis_filename, 'w', encoding='utf-8') as f:
                    f.write(f"UUID: {uuid}\n")
                    f.write(f"Total Tokens: {len(full_token_sequence)}\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{'Index':<10s} | {'Token ID':<10s} | {'Start (s)':<18s} | {'End (s)':<18s}\n")
                    f.write("-" * 60 + "\n")

                    for i, token_id in enumerate(full_token_sequence):
                        start_mel_frame = i * token_mel_ratio
                        end_mel_frame = (i + 1) * token_mel_ratio
                        start_time_s = start_mel_frame * time_per_mel_frame
                        end_time_s = end_mel_frame * time_per_mel_frame
                        f.write(f"{i:<10d} | {token_id:<10d} | {start_time_s:<18.4f} | {end_time_s:<18.4f}\n")
                
                print(f"[DEBUG] Token timeline saved: {os.path.abspath(analysis_filename)}")

            except Exception as e:
                print(f"[ANALYSIS-ERROR] Failed to generate token timeline file: {e}")
        
        if bgm_removal:
            tts_mel_numpy = tts_mel.squeeze().cpu().numpy()
            kernel_size = (1, 3) 
            filtered_mel_numpy = scipy.ndimage.median_filter(tts_mel_numpy, size=kernel_size)
            tts_mel = torch.from_numpy(filtered_mel_numpy).unsqueeze(0).to(self.device)

        tts_mel = tts_mel[:, :, token_offset * (self.flow.token_mel_ratio if hasattr(self.flow, 'token_mel_ratio') else 4):]
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
            
        if finalize is False:
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
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
                
        return tts_speech
    
    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), 
            stream=False, speed=1.0, bgm_removal: bool = False, **kwargs):

        sample_rate = kwargs.get('sample_rate')
        if sample_rate is None:
            raise ValueError("tts method must be called with a 'sample_rate' keyword argument.")

        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()

        kwargs_for_token2wav = kwargs.copy()
        kwargs_for_token2wav['sample_rate'] = sample_rate

        if stream is True:
            token_offset = 0
            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
            while True:
                time.sleep(0.1)
                this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     token_offset=token_offset,
                                                     uuid=this_uuid,
                                                     stream=stream,
                                                     finalize=False,
                                                     bgm_removal=bgm_removal,
                                                     **kwargs_for_token2wav)
                    token_offset += this_token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=token_offset,
                                             uuid=this_uuid,
                                             finalize=True,
                                             bgm_removal=bgm_removal,
                                             **kwargs_for_token2wav)
            yield {'tts_speech': this_tts_speech.cpu()}
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
                                             speed=speed,
                                             bgm_removal=bgm_removal,
                                             **kwargs_for_token2wav)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()