import os
import sys
import torch
import torchaudio

# --- [설정] ---
# app_stream_tts.py와 동일한 경로 설정을 사용합니다.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(THIS_DIR)

sys.path.append(PROJ_DIR)
sys.path.append(os.path.join(PROJ_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

# --- [합성할 텍스트] ---
# 로그에 나온 텍스트 예시입니다. 원하는 문장으로 변경하세요.
TEXT_TO_SYNTHESIZE = "한번 계속 입력을 하여 끊기는 느낌이 있는지 확인해 봅시다."

# --- [모델 경로] ---
MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice-KSS-Finetuned")
PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")
OUTPUT_WAV = os.path.join(THIS_DIR, "full_sentence_test.wav") # 저장될 파일 이름

# --- [1. 모델 로드] ---
print("모델 로드 중...")
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"{MODEL_DIR} does not exist!")
if not os.path.isfile(PROMPT_WAV):
    raise FileNotFoundError(f"{PROMPT_WAV} not found!")

cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=False)
prompt_speech_16k = load_wav(PROMPT_WAV, 16000)
SAMPLE_RATE = cosyvoice.sample_rate
print("모델 로드 완료.")

# --- [2. 전체 문장 합성 (stream=False)] ---
print(f"합성 시작: '{TEXT_TO_SYNTHESIZE}'")

try:
    # stream=False로 설정하면, for 루프는 단 한 번만 실행됩니다.
    for out in cosyvoice.inference_instruct2(
            tts_text=TEXT_TO_SYNTHESIZE,
            instruct_text="",
            prompt_speech_16k=prompt_speech_16k,
            zero_shot_spk_id="",
            stream=False, # <--- [핵심] 전체 문장 동시 합성
            speed=1.0,
            text_frontend=True # <--- cosyvoice가 알아서 문장을 정규화하도록 함
    ):
        final_audio_tensor = out["tts_speech"].cpu()
        
        # --- [3. 파일 저장] ---
        torchaudio.save(OUTPUT_WAV, final_audio_tensor, SAMPLE_RATE, format="wav")
        print(f"\n✅ 합성 완료! 파일이 '{OUTPUT_WAV}'에 저장되었습니다.")
        print(f"오디오 길이: {final_audio_tensor.shape[1] / SAMPLE_RATE:.2f} 초")
        break # stream=False이므로 루프는 한 번만 돌지만, 명시적으로 탈출

except Exception as e:
    print(f"\n❌ 합성 중 오류 발생: {e}")