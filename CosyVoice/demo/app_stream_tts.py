#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import io
import time
import torch
import torchaudio
from typing import Generator

# 프로젝트 경로 설정
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PROJ_DIR)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContextualTTSStreamer:
    """
    어절 단위로 문맥을 유지하며 TTS 스트리밍을 관리하는 클래스
    """
    def __init__(self, model: CosyVoice2, prompt_wav_path: str):
        self.model = model
        self.prompt_speech_16k = load_wav(prompt_wav_path, 16000)
        self.sample_rate = self.model.sample_rate
        self.full_text_context = ""
        logging.info("ContextualTTSStreamer가 초기화되었습니다.")

    def reset(self):
        """문맥을 초기화합니다."""
        self.full_text_context = ""
        logging.info("문맥이 초기화되었습니다.")

    def stream_chunk(self, text_chunk: str) -> Generator[torch.Tensor, None, None]:
        """
        하나의 텍스트 어절(chunk)을 받아 오디오 텐서(Tensor)를 스트리밍으로 반환합니다.

        Args:
            text_chunk (str): 음성으로 변환할 새로운 텍스트 어절 (예: "뭐해")

        Yields:
            torch.Tensor: 생성된 오디오 청크
        """
        if not text_chunk.strip():
            return

        # 언어 태그 추가 (예시: 한국어)
        # 실제 애플리케이션에서는 언어 감지 로직이 필요할 수 있습니다.
        tagged_chunk = f"<|ko|>{text_chunk}"
        
        logging.info(f"음성 생성 요청...")
        logging.info(f"  - 문맥 (instruct_text): '{self.full_text_context}'")
        logging.info(f"  - 대상 (tts_text)    : '{tagged_chunk}'")

        try:
            # inference_instruct2를 사용하여 스트리밍 생성
            # instruct_text에는 이전까지의 문맥을, tts_text에는 현재 어절을 전달
            for out in self.model.inference_instruct2(
                tts_text=tagged_chunk,
                instruct_text=self.full_text_context,
                prompt_speech_16k=self.prompt_speech_16k,
                stream=True,
                speed=1.0
            ):
                audio_tensor = out["tts_speech"].cpu()
                yield audio_tensor
            
            logging.info(f"'{text_chunk}' 어절 생성 완료.")

            # 현재 어절을 다음을 위한 문맥에 추가
            if self.full_text_context:
                self.full_text_context += f" {text_chunk}"
            else:
                self.full_text_context = text_chunk
        
        except Exception as e:
            logging.error(f"TTS 생성 중 오류 발생: {e}", exc_info=True)


def main():
    # --- 1. 모델 및 프롬프트 초기화 ---
    model_dir = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice-KSS-Finetuned")
    prompt_wav = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"모델 디렉터리를 찾을 수 없습니다: {model_dir}")
    if not os.path.isfile(prompt_wav):
        raise FileNotFoundError(f"프롬프트 WAV 파일을 찾을 수 없습니다: {prompt_wav}")

    print("서버 시작: CosyVoice2 로드 중...")
    cosyvoice = CosyVoice2(model_dir=model_dir, fp16=False)
    print("모델 로드 완료.")

    # --- 2. 스트리머 객체 생성 ---
    streamer = ContextualTTSStreamer(model=cosyvoice, prompt_wav_path=prompt_wav)

    # --- 3. TTS 스트리밍 시뮬레이션 ---
    full_sentence = "오늘은 뭐해 집에서 쉴래?"
    chunks = full_sentence.split(' ')

    print(f"\n입력 문장: '{full_sentence}'")
    print(f"어절 단위로 음성 생성을 시작합니다: {chunks}")

    output_dir = "tts_output_chunks"
    os.makedirs(output_dir, exist_ok=True)
    
    # 생성된 모든 오디오 청크를 모으기 위한 리스트
    all_audio_chunks = []

    start_time = time.time()
    for i, chunk_text in enumerate(chunks):
        print(f"\n--- {i+1}번째 어절 처리 중: '{chunk_text}' ---")
        
        # 현재 어절에 대한 오디오 스트림을 받아 처리
        audio_stream_for_chunk = []
        for audio_chunk_tensor in streamer.stream_chunk(chunk_text):
            audio_stream_for_chunk.append(audio_chunk_tensor)
            # (실제 스트리밍 앱에서는 여기서 바로 클라이언트로 전송)

        if audio_stream_for_chunk:
            # 어절 단위로 생성된 오디오를 하나의 텐서로 합침
            full_audio_for_chunk = torch.cat(audio_stream_for_chunk, dim=1)
            all_audio_chunks.append(full_audio_for_chunk)

            # 디버깅을 위해 각 어절별 wav 파일 저장
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