#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  CUDA_VISIBLE_DEVICES=4 python -u demo/tester_bistream.py
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

# MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice2-0.5B")
# PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")


MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice-KSS-Finetuned")
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
    return sess

def start_tts_worker_if_needed(sess: Session):
    with SESS_LOCK:
        t = threading.Thread(target=tts_worker, args=(sess,), daemon=True)
        t.start()

# ==============================
# 2) Sentence Queuing
# ==============================
def enqueue_full_sentence(sess: Session):
    sentence = sess.text.strip()
    if sentence:
        logging.debug(f"[{sess.sid}] enqueue full sentence for bistream: {sentence[:80]}")
        sess.tts_q.put(sentence)
        start_tts_worker_if_needed(sess)

# ==============================
# 3) TTS Worker & Synthesizer (BISTREAM VERSION)
# ==============================
def _create_text_generator(text: str, chunk_size: int = 2) -> Generator[str, None, None]:
    """텍스트를 작은 조각으로 나누어 yield하는 제너레이터 (bistream용)"""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]) + " "
        logging.info(f"Yielding text for bistream: '{chunk.strip()}'")
        yield chunk
        time.sleep(0.05) # 실제 스트리밍 환경처럼 약간의 딜레이

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
        instruction = ""
    else:
        tagged_sentence = f"<|en|>{final_sentence}"
        instruction = ""

    try:
        # ✨✨✨ 핵심 변경: 텍스트 제너레이터를 tts_text로 전달하고, stream=True로 설정 ✨✨✨
        tts_generator = _create_text_generator(tagged_sentence)
        for out in cosyvoice.inference_instruct2(
                tts_text=tts_generator,
                instruct_text=instruction,
                prompt_speech_16k=prompt_speech_16k,
                zero_shot_spk_id="",
                stream=True, # bistream 모드에서는 오디오 청크를 스트리밍으로 받아야 함
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

def tts_worker(sess: Session):
    """TTS 작업을 처리하는 백그라운드 스레드 (bistream 실험용 버전)"""
    print(f"[{sess.sid}] TTS worker started (bistream mode).")
    
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
    
    enqueue_full_sentence(sess)
    
    return jsonify({"ok": True})

@app.route("/sse_audio")
def sse_audio():
    sid = request.args.get("sid") or str(uuid.uuid4())
    sess = get_or_create_session(sid)

    def event_stream():
        while True:
            try:
                msg = sess.sse_q.get(timeout=20.0)
                yield f"data: {msg}\n\n"
                if '"type": "end"' in msg:
                    break
            except queue.Empty:
                break
    
    headers = { "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Content-Type": "text/event-stream", "Connection": "keep-alive" }
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
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)