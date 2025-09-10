#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import requests
import json
import uuid
import threading
import base64
import wave
import random
from tqdm import tqdm
from sseclient import SSEClient

# ==============================
# 설정 (Configuration)
# ==============================
SERVER_URL = "http://127.0.0.1:8000"
INPUT_FILE = "/app/demo/sentences.txt"
OUTPUT_DIR = "test_outputs"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
LOG_FILE = os.path.join(OUTPUT_DIR, "test_summary.csv")

# 타이핑 속도 시뮬레이션 (초 단위)
MIN_TYPING_DELAY = 0.05
MAX_TYPING_DELAY = 0.2

# ==============================
# SSE 오디오 수신 스레드
# ==============================
class AudioReceiverThread(threading.Thread):
    def __init__(self, session_id, output_path):
        super().__init__()
        self.session_id = session_id
        self.output_path = output_path
        self.audio_frames = []
        self.finished = threading.Event()
        # CosyVoice는 16000Hz, 16-bit Mono로 오디오를 생성합니다.
        self.CHANNELS = 1
        self.SAMPWIDTH = 2
        self.FRAMERATE = 16000

    def run(self):
        try:
            url = f"{SERVER_URL}/sse_audio?sid={self.session_id}"
            response = requests.get(url, stream=True)
            client = SSEClient(response)

            for event in client.events():
                if event.event == 'message':
                    try:
                        data = json.loads(event.data)
                        if data.get("type") == "audio" and data.get("b64wav"):
                            wav_bytes = base64.b64decode(data["b64wav"])
                            # WAV 헤더(44바이트)를 제외하고 순수 오디오 데이터만 저장
                            self.audio_frames.append(wav_bytes[44:])
                        elif data.get("type") == "end":
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue
        finally:
            self.save_wav()
            self.finished.set()

    def save_wav(self):
        if not self.audio_frames:
            print(f"[{self.session_id}] 수신된 오디오 프레임이 없어 파일을 저장하지 않습니다.")
            return
            
        with wave.open(self.output_path, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPWIDTH)
            wf.setframerate(self.FRAMERATE)
            wf.writeframes(b''.join(self.audio_frames))
        print(f"[{self.session_id}] 오디오 파일 저장 완료: {self.output_path}")

# ==============================
# 메인 테스트 로직
# ==============================
def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("timestamp,input_sentence,output_audio_path,synthesis_duration_sec,status,error_message\n")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"총 {len(sentences)}개 문장으로 테스트를 시작합니다.")
    print(f"서버 URL: {SERVER_URL}")

    for sentence in tqdm(sentences, desc="테스트 진행률"):
        session_id = str(uuid.uuid4())
        output_path = os.path.join(AUDIO_DIR, f"{session_id}.wav")
        start_time = time.time()
        status = "SUCCESS"
        error_msg = ""
        
        try:
            receiver = AudioReceiverThread(session_id, output_path)
            receiver.start()

            current_text = ""
            for char in sentence:
                current_text += char
                requests.post(f"{SERVER_URL}/type", json={"sid": session_id, "text": current_text}, timeout=5)
                time.sleep(random.uniform(MIN_TYPING_DELAY, MAX_TYPING_DELAY))
            
            receiver.finished.wait(timeout=60)
            if not receiver.finished.is_set():
                status = "FAIL"
                error_msg = "Timeout: 60초 내에 오디오 스트림이 종료되지 않았습니다."

        except Exception as e:
            status = "FAIL"
            error_msg = str(e).replace('"', "'")
            if 'receiver' in locals() and receiver.is_alive():
                receiver.finished.set()
        
        duration = time.time() - start_time
        
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f'"{time.strftime("%Y-%m-%d %H:%M:%S")}","{sentence}","{output_path}",{duration:.2f},{status},"{error_msg}"\n')

if __name__ == "__main__":
    main()