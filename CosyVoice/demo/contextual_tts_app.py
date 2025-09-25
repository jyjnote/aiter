#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import torchaudio
from typing import Generator, List

# 프로젝트 경로 설정
# (이전과 동일)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PROJ_DIR)

from cosyvoice.cli.cosyvoice import CosyVoice2
import sys # <--- sys 모듈 import 추가
from cosyvoice.utils.file_utils import load_wav, logging

# --- 강력한 로깅 설정으로 교체 ---
# 다른 라이브러리가 설정한 기존 로거를 모두 제거
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 우리의 설정을 다시 적용 (INFO 레벨, 표준 출력으로 강제)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

class ContextualTTSStreamer:
    """
    지정된 개수의 이전 어절을 '음향 문맥(Acoustic Context)'으로 참고하여
    TTS 스트리밍을 관리하는 클래스
    """
    def __init__(self, model: CosyVoice2, prompt_wav_path: str):
        self.model = model
        self.prompt_speech_16k = load_wav(prompt_wav_path, 16000)
        self.sample_rate = self.model.sample_rate
        # 각 어절별 토큰 리스트를 저장할 히스토리
        self.chunk_token_history: List[List[int]] = []
        logging.info("ContextualTTSStreamer 초기화 완료. (N개 어절 참고 모드)")

    def reset(self):
        """문맥(speech tokens) 히스토리를 초기화합니다."""
        self.chunk_token_history = []
        logging.info("음향 문맥 히스토리가 초기화되었습니다.")

    def stream_chunk(self, text_chunk: str, context_chunk_size: int = 1) -> Generator[torch.Tensor, None, None]:
        """
        하나의 텍스트 어절(chunk)을 생성합니다.

        Args:
            text_chunk (str): 음성으로 변환할 새로운 텍스트 어절.
            context_chunk_size (int): 문맥으로 참고할 이전 어절의 개수 (기본값: 1).
        """
        if not text_chunk.strip():
            return

        # --- 📝 핵심 로직: 참고할 이전 토큰들을 준비 ---
        # 1. 히스토리에서 마지막 N개의 어절 토큰 리스트를 가져옵니다.
        num_history_chunks = len(self.chunk_token_history)
        chunks_to_use = self.chunk_token_history[max(0, num_history_chunks - context_chunk_size):]
        
        # 2. 여러 어절의 토큰 리스트를 하나의 리스트로 합칩니다.
        previous_speech_tokens = [token for chunk_tokens in chunks_to_use for token in chunk_tokens]
        # ---------------------------------------------

        tagged_chunk = f"<|ko|>{text_chunk}"
        
        logging.info(f"음성 생성 요청 (참고 어절 개수: {min(num_history_chunks, context_chunk_size)})...")
        logging.info(f"  - 이전 Speech Tokens (문맥): {len(previous_speech_tokens)}개")
        logging.info(f"  - 대상 텍스트 (tts_text) : '{tagged_chunk}'")

        try:
            for out in self.model.inference_with_acoustic_prompt(
                tts_text=tagged_chunk,
                prompt_speech_16k=self.prompt_speech_16k,
                previous_speech_tokens=previous_speech_tokens,
                stream=False,
                speed=1.0
            ):
                audio_tensor = out["tts_speech"].cpu()
                final_tokens_for_this_chunk = out["speech_tokens"]
                
                logging.info(f"'{text_chunk}' 어절 생성 완료. 생성된 Speech Tokens: {len(final_tokens_for_this_chunk)}개")
                
                # 다음 어절을 위해, 현재 생성된 토큰 리스트를 히스토리에 추가합니다.
                self.chunk_token_history.append(final_tokens_for_this_chunk)
                
                yield audio_tensor
        
        except Exception as e:
            logging.error(f"TTS 생성 중 오류 발생: {e}", exc_info=True)


def main():
    # --- 1. 모델 및 프롬프트 초기화 ---
    model_dir = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice-KSS-Finetuned")
    prompt_wav = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")

    print("서버 시작: CosyVoice2 로드 중...")
    cosyvoice = CosyVoice2(model_dir=model_dir, fp16=False)
    print("모델 로드 완료.")

    # --- 2. 스트리머 객체 생성 ---
    streamer = ContextualTTSStreamer(model=cosyvoice, prompt_wav_path=prompt_wav)

    # --- 3. TTS 스트리밍 시뮬레이션 ---
    full_sentence = "바이스트림으로 한번 테스트 해볼까나? 아 제발 좀 작동좀 해라."
    chunks = full_sentence.split(' ')

    print(f"\n입력 문장: '{full_sentence}'")
    print(f"어절 단위로 음성 생성을 시작합니다: {chunks}")

    output_dir = "tts_output_chunks_acoustic"
    os.makedirs(output_dir, exist_ok=True)
    
    all_audio_chunks = []

    # ⭐ 이전 어절을 몇 개 참고할지 여기서 조절할 수 있습니다.
    CONTEXT_CHUNKS = 1 

    start_time = time.time()
    streamer.reset() # 시작 전 초기화
    for i, chunk_text in enumerate(chunks):
        print(f"\n--- {i+1}번째 어절 처리 중: '{chunk_text}' ---")
        
        for full_audio_for_chunk in streamer.stream_chunk(chunk_text, context_chunk_size=CONTEXT_CHUNKS):
            all_audio_chunks.append(full_audio_for_chunk)

            output_path = os.path.join(output_dir, f"chunk_{i}_{chunk_text}.wav")
            torchaudio.save(output_path, full_audio_for_chunk, streamer.sample_rate)
            print(f"-> 오디오 파일 저장 완료: {output_path}")

    # --- 4. 전체 문장 오디오 생성 및 저장 ---
    if all_audio_chunks:
        final_full_audio = torch.cat(all_audio_chunks, dim=1)
        final_output_path = os.path.join(output_dir, "final_full_sentence.wav")
        torchaudio.save(final_output_path, final_full_audio, streamer.sample_rate)
        print(f"\n--- 최종 결과 ---")
        print(f"모든 어절을 합친 전체 문장 오디오 저장 완료: {final_output_path}")

    end_time = time.time()
    print(f"총 소요 시간: {end_time - start_time:.2f}초")


if __name__ == "__main__":
    main()