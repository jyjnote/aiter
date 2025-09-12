#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  CUDA_VISIBLE_DEVICES=4 python -u demo/app_stream_tts.py
import os
import sys
import io
import re
import time
import json
import base64
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Generator

import torch
import torchaudio
from flask import Flask, request, Response, render_template, jsonify

import random
import numpy as np

# ==============================
# 0) App & Model Init
# ==============================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(THIS_DIR)

sys.path.append(PROJ_DIR)
sys.path.append(os.path.join(PROJ_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

# --- 강력한 로깅 설정 ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
# --- 설정 끝 ---

app = Flask(__name__)

MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice2-0.5B")
PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"{MODEL_DIR} does not exist! (expected CosyVoice2-0.5B)")
if not os.path.isfile(PROMPT_WAV):
    raise FileNotFoundError(f"{PROMPT_WAV} not found!")

print("서버 시작: CosyVoice2 로드…")
cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=False)
prompt_speech_16k = load_wav(PROMPT_WAV, 16000)
print("모델 로드 완료.")

SAMPLE_RATE = cosyvoice.sample_rate

# ==============================
# Helper Functions
# ==============================
def is_korean(char: str) -> bool:
    if not char: return False
    return '\uac00' <= char <= '\ud7a3'

def denoise_audio_chunk(wav_bytes: bytes, sample_rate: int) -> bytes:
    if not wav_bytes: return wav_bytes
    try:
        buf = io.BytesIO(wav_bytes)
        waveform, sr = torchaudio.load(buf)
        enhanced_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, 80)
        out_buf = io.BytesIO()
        torchaudio.save(out_buf, enhanced_waveform, sample_rate, format="wav")
        out_buf.seek(0)
        return out_buf.read()
    except Exception as e:
        logging.error(f"Denoising error: {e}")
        return wav_bytes

# ==============================
# 1) Session Management
# ==============================
@dataclass
class Session:
    sid: str
    text: str = ""
    last_flush_idx: int = 0
    last_input_ts: float = field(default_factory=time.time)
    tts_q: queue.Queue = field(default_factory=queue.Queue)
    sse_q: queue.Queue = field(default_factory=queue.Queue)

SESSIONS: Dict[str, Session] = {}
SESS_LOCK = threading.Lock()

def get_or_create_session(sid: str) -> Session:
    with SESS_LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            sess = Session(sid=sid)
            SESSIONS[sid] = sess
            # ✨ tts_worker 스레드를 여기서 바로 시작하지 않고, 필요할 때 시작하도록 변경
    return sess

def start_tts_worker_if_needed(sess: Session):
    with SESS_LOCK:
        # 이미 워커가 실행 중인지 확인하는 로직은 단순화를 위해 생략
        # 실험 스크립트는 세션마다 한 번만 호출하므로 문제 없음
        t = threading.Thread(target=tts_worker, args=(sess,), daemon=True)
        t.start()

# ==============================
# 2) Sentence Extraction & Queuing
# ==============================
def enqueue_flushable_sentences(sess: Session):
    sentence = sess.text.strip()
    if sentence:
        logging.debug(f"[{sess.sid}] enqueue sentence: {sentence[:80]}")
        sess.tts_q.put(sentence)
        start_tts_worker_if_needed(sess) # ✨ 문장이 들어오면 워커 시작

# ==============================
# 3) TTS Worker & Synthesizer
# ==============================
def stream_sentence_to_wav_chunks(sentence: str) -> Generator[bytes, None, None]:
    SEED = 1986
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logging.info(f"Random seeds fixed to {SEED}")

    final_sentence = sentence.strip()
    is_ko = is_korean(final_sentence.lstrip()[:1])

    if is_ko:
        tagged_sentence = f"<|ko|>{final_sentence}"
        instruction = "Please pronounce it clearly and articulately in Korean."
    else:
        tagged_sentence = f"<|en|>{final_sentence}"
        instruction = "Please pronounce it clearly and articulately in English."

    try:
        for out in cosyvoice.inference_instruct2(
                tts_text=tagged_sentence,
                instruct_text=instruction,
                prompt_speech_16k=prompt_speech_16k,
                zero_shot_spk_id="",
                stream=False, # 실험용으로 False 설정
                speed=1.0,
                text_frontend=True
            ):
            audio_chunk = out["tts_speech"].cpu()
            buf = io.BytesIO()
            torchaudio.save(buf, audio_chunk, SAMPLE_RATE, format="wav")
            buf.seek(0)
            yield buf.read()
    except Exception as e:
        logging.error(f"TTS (instruct2) Error: {e}", exc_info=True)

# ✨✨✨ 자동 실험에 맞게 수정된 tts_worker 함수 ✨✨✨
def tts_worker(sess: Session):
    """TTS 작업을 처리하는 백그라운드 스레드 (자동 실험용 버전)"""
    print(f"[{sess.sid}] TTS worker started.")
    
    try:
        sentence = sess.tts_q.get(timeout=10.0)
    except queue.Empty:
        sentence = None

    if sentence:
        try:
            for wav_chunk_bytes in stream_sentence_to_wav_chunks(sentence):
                if wav_chunk_bytes:
                    enhanced_chunk_bytes = denoise_audio_chunk(wav_chunk_bytes, SAMPLE_RATE)
                    b64 = base64.b64encode(enhanced_chunk_bytes).decode("utf-8")
                    sess.sse_q.put(json.dumps({"type": "audio", "b64wav": b64}))
        except Exception as e:
            logging.error(f"TTS Error: {e}", exc_info=True)
            sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))

    sess.sse_q.put(json.dumps({"type": "end"}))
    print(f"[{sess.sid}] TTS worker finished and sent 'end' signal.")
    # 세션 정리 (선택 사항이지만 메모리 관리에 좋음)
    with SESS_LOCK:
        if sess.sid in SESSIONS:
            del SESSIONS[sess.sid]

# ==============================
# 4) HTTP Routes
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/type", methods=["POST"])
def type_event():
    data = request.get_json(force=True)
    sid = data.get("sid") or str(uuid.uuid4())
    text = data.get("text", "")

    sess = get_or_create_session(sid)
    sess.text = text
    
    # 실험 스크립트는 문장 전체를 한 번에 보내므로, 바로 큐에 넣고 워커 시작
    enqueue_flushable_sentences(sess)
    
    return jsonify({"ok": True})

@app.route("/sse_audio")
def sse_audio():
    sid = request.args.get("sid") or str(uuid.uuid4())
    sess = get_or_create_session(sid)

    def event_stream():
        while True:
            try:
                msg = sess.sse_q.get(timeout=20.0) # 타임아웃을 늘려 서버가 응답할 시간을 줌
                yield f"data: {msg}\n\n"
                # 'end' 메시지를 받으면 스트림 종료
                if '"type": "end"' in msg:
                    break
            except queue.Empty:
                # 타임아웃 발생 시 스트림 종료
                break
    
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
    }
    return Response(event_stream(), headers=headers)

# ==============================
# 5) Shutdown Route (For Automation)
# ==============================
@app.route('/shutdown')
def shutdown():
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_func()
    return 'Server shutting down...'

# ==============================
# 6) Run
# ==============================
if __name__ == "__main__":
    # 자동 실험을 위해 debug=False로 설정
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)