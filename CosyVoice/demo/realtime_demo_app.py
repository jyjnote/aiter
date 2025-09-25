#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import io
import json
import base64
import queue
import threading
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Generator

import torch
import torchaudio
from flask import Flask, request, Response, render_template, jsonify

# --- 프로젝트 경로 설정 ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PROJ_DIR)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

# --- 강력한 로깅 설정 ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- 1. 모델 로드 ---
MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice-KSS-Finetuned")
PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")

logging.info("서버 시작: CosyVoice2 로드 중...")
cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=False)
logging.info("모델 로드 완료.")

app = Flask(__name__)

# --- ContextualTTSStreamer 클래스 (이전 코드와 동일) ---
# (이 파일 안에 클래스를 직접 포함시켜 단일 파일로 실행 가능하게 만듭니다)
class ContextualTTSStreamer:
    def __init__(self, model: CosyVoice2, prompt_wav_path: str):
        self.model = model
        self.prompt_speech_16k = load_wav(prompt_wav_path, 16000)
        self.sample_rate = self.model.sample_rate
        self.last_speech_tokens: List[int] = []
        logging.info("ContextualTTSStreamer 초기화 완료. (직전 어절 참고 모드)")

    def reset(self):
        self.last_speech_tokens = []
        logging.info("음향 문맥(speech tokens)이 초기화되었습니다.")

    def stream_chunk(self, text_chunk: str) -> Generator[torch.Tensor, None, None]:
        if not text_chunk.strip():
            return
        tagged_chunk = f"<|ko|>{text_chunk}"
        logging.info(f"음성 생성 요청: '{tagged_chunk}', 이전 Tokens: {len(self.last_speech_tokens)}개")
        try:
            for out in self.model.inference_with_acoustic_prompt(
                tts_text=tagged_chunk,
                prompt_speech_16k=self.prompt_speech_16k,
                previous_speech_tokens=self.last_speech_tokens,
                stream=False,
                speed=1.0
            ):
                audio_tensor = out["tts_speech"].cpu()
                final_tokens = out["speech_tokens"]
                logging.info(f"'{text_chunk}' 생성 완료. Tokens: {len(final_tokens)}개")
                self.last_speech_tokens = final_tokens
                yield audio_tensor
        except Exception as e:
            logging.error(f"TTS 생성 중 오류: {e}", exc_info=True)

# --- 2. 세션 관리 ---
@dataclass
class TTSSession:
    sid: str
    streamer: ContextualTTSStreamer
    text_q: queue.Queue = field(default_factory=queue.Queue)
    sse_q: queue.Queue = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker: Optional[threading.Thread] = None
    last_processed_text: str = ""

SESSIONS: Dict[str, TTSSession] = {}
SESS_LOCK = threading.Lock()

# 기존의 tts_worker 함수를 아래 내용으로 전체 교체

def tts_worker(sess: TTSSession):
    """백그라운드에서 TTS 작업을 처리하는 스레드"""
    logging.info(f"[{sess.sid}] TTS Worker 시작.")
    while not sess.stop_event.is_set():
        try:
            text_chunk = sess.text_q.get(timeout=1)

            # --- 핵심 로직 3: 리셋 작업 처리 ---
            if text_chunk == "RESET_CONTEXT":
                logging.info(f"[{sess.sid}] Worker가 문맥을 리셋합니다.")
                sess.streamer.reset()
                continue
            
            for audio_tensor in sess.streamer.stream_chunk(text_chunk):
                buf = io.BytesIO()
                torchaudio.save(buf, audio_tensor, sess.streamer.sample_rate, format="wav")
                buf.seek(0)
                wav_bytes = buf.read()
                b64_wav = base64.b64encode(wav_bytes).decode('utf-8')
                sse_data = json.dumps({"type": "audio", "b64wav": b64_wav})
                sess.sse_q.put(sse_data)
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"[{sess.sid}] Worker Error: {e}", exc_info=True)
    logging.info(f"[{sess.sid}] TTS Worker 중지.")

def get_or_create_session(sid: str) -> TTSSession:
    with SESS_LOCK:
        if sid not in SESSIONS:
            streamer = ContextualTTSStreamer(model=cosyvoice, prompt_wav_path=PROMPT_WAV)
            sess = TTSSession(sid=sid, streamer=streamer)
            sess.worker = threading.Thread(target=tts_worker, args=(sess,), daemon=True)
            sess.worker.start()
            SESSIONS[sid] = sess
            logging.info(f"[{sid}] 새 세션 생성.")
        return SESSIONS[sid]

# --- 3. Flask 라우트(Endpoints) ---
@app.route("/")
def index():
    return render_template("index.html")

# 기존의 @app.route("/process", methods=["POST"]) 함수를 아래 내용으로 전체 교체

@app.route("/process", methods=["POST"])
def process_text():
    data = request.json
    sid = data.get("sid")
    full_text = data.get("full_text", "")
    
    sess = get_or_create_session(sid)
    
    # 문장 종결 구두점 정의
    sentence_terminators = ".!?"
    
    new_text = ""
    # 이전에 처리한 텍스트와 현재 전체 텍스트를 비교
    if full_text.startswith(sess.last_processed_text):
        new_text = full_text[len(sess.last_processed_text):]
    else: # 사용자가 텍스트를 지우거나 중간을 수정한 경우, 전체 리셋
        logging.info(f"[{sid}] 텍스트가 변경되어 문맥 리셋.")
        sess.streamer.reset()
        new_text = full_text

    if new_text.strip():
        chunks = new_text.split(' ')
        for chunk in chunks:
            clean_chunk = chunk.strip()
            if not clean_chunk:
                continue

            # --- 핵심 로직 1: 구두점만 있는 어절은 건너뛰기 ---
            # 모든 문자가 구두점인지 확인 (정규식이나 더 복잡한 로직도 가능)
            is_only_punctuation = all(char in '.,!?' for char in clean_chunk)
            if is_only_punctuation:
                logging.info(f"[{sid}] 구두점 어절 건너뛰기: '{clean_chunk}'")
                continue

            logging.info(f"[{sid}] 새 어절 큐잉: '{clean_chunk}'")
            sess.text_q.put(clean_chunk)

            # --- 핵심 로직 2: 문장 종결 시 문맥 리셋 ---
            # 현재 어절에 종결 구두점이 포함되어 있는지 확인
            if any(term in clean_chunk for term in sentence_terminators):
                # 리셋 작업을 큐에 넣어 TTS 작업 순서와 동기화
                sess.text_q.put("RESET_CONTEXT")
                logging.info(f"[{sid}] 문장 종결 감지. 다음 어절부터 문맥 초기화 예정.")

    sess.last_processed_text = full_text
    return jsonify({"ok": True})

@app.route("/sse_audio")
def sse_audio():
    sid = request.args.get("sid")
    if not sid:
        return Response("Session ID가 필요합니다.", status=400)
    
    sess = get_or_create_session(sid)

    def event_stream():
        while not sess.stop_event.is_set():
            try:
                data = sess.sse_q.get(timeout=0.5)
                yield f"data: {data}\n\n"
            except queue.Empty:
                # 핑 메시지를 보내 연결 유지
                yield 'data: {"type":"ping"}\n\n'

    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=8000, debug=False, threaded=True) # debug=True일 경우 werkzeug이 이중 로딩되어 모델을 두번 로드할수 있으므로 주의
    app.run(host="0.0.0.0", port=8000, debug=False)