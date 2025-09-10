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

# --- 강력한 로깅 설정 (기존 basicConfig 대체) ---
# 모든 기존 로거의 핸들러를 제거 (Flask/werkzeug의 기본 핸들러 포함)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 여기에 우리의 설정을 다시 적용 (stream=sys.stdout 추가)
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
    """주어진 문자가 한글 범위(가-힣)에 있는지 확인합니다."""
    if not char:
        return False
    return '\uac00' <= char <= '\ud7a3'

def denoise_audio_chunk(wav_bytes: bytes, sample_rate: int) -> bytes:
    """
    오디오 청크(bytes)에서 저주파 노이즈(럼블)를 제거합니다.
    """
    if not wav_bytes:
        return wav_bytes

    try:
        # 1. Bytes -> Tensor 변환
        buf = io.BytesIO(wav_bytes)
        waveform, sr = torchaudio.load(buf)

        # 2. 저주파 노이즈 제거를 위한 High-pass 필터 적용
        # 80Hz 이하의 소리(웅웅거리는 배경음, 마이크 럼블 등)를 줄입니다.
        cutoff_freq = 80
        enhanced_waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq)

        # 3. Tensor -> Bytes 변환
        out_buf = io.BytesIO()
        torchaudio.save(out_buf, enhanced_waveform, sample_rate, format="wav")
        out_buf.seek(0)
        return out_buf.read()
    except Exception as e:
        # 오디오 처리 중 오류 발생 시 원본을 그대로 반환
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
    last_flush_ts: float = field(default_factory=time.time)
    tts_q: queue.Queue = field(default_factory=queue.Queue)
    sse_q: queue.Queue = field(default_factory=queue.Queue)
    worker_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

SESSIONS: Dict[str, Session] = {}
SESS_LOCK = threading.Lock()
KEEPALIVE_SEC = 15.0

def get_or_create_session(sid: str) -> Session:
    with SESS_LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            sess = Session(sid=sid)
            SESSIONS[sid] = sess
            t = threading.Thread(target=tts_worker, args=(sess,), daemon=True)
            t.start()
            sess.worker_thread = t
    return sess

# ==============================
# 2) Sentence Extraction & Queuing
# ==============================
def enqueue_flushable_sentences(sess: Session, force: bool = False):
    new_segment = sess.text[sess.last_flush_idx:]
    if not new_segment:
        return

    consumed = 0
    sentences: List[str] = []
    
    sentence_re = re.compile(r"[^\.!\?…。\！？,;:]*[\.!\?…。\！？,;:]")

    for m in sentence_re.finditer(new_segment):
        end = m.end()
        chunk = new_segment[:end].strip()
        if chunk:
            sentences.append(chunk)
        new_segment = new_segment[end:]
        consumed += end

    sess.last_flush_idx += consumed

    if force:
        rest = new_segment.strip()
        if rest:
            punctuation = ".!?…。！？,;:"
            if not rest.endswith(tuple(punctuation)):
                rest += "."
            sentences.append(rest)
        sess.last_flush_idx = len(sess.text)

    if sentences:
        sess.last_flush_ts = time.time()
        for s in sentences:
            logging.debug(f"[{sess.sid}] enqueue sentence: {s[:80]}{'...' if len(s)>80 else ''}")
            sess.tts_q.put(s)

# ==============================
# 3) TTS Worker & Synthesizer
# ==============================
def stream_sentence_to_wav_chunks(sentence: str) -> Generator[bytes, None, None]:
    """한 문장을 받아 오디오 청크(chunk)들을 스트리밍으로 반환하는 제너레이터"""

    SEED = 1986
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logging.info(f"Random seeds fixed to {SEED}")


    final_sentence = sentence.strip()

    first_char = final_sentence.lstrip()[:1]
    is_ko = is_korean(first_char)

    instruction = ""
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
                stream=True,
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
    """TTS 작업을 처리하는 백그라운드 스레드"""
    last_keepalive = time.time()
    
    print(f"[{sess.sid}] TTS worker started.")
    
    IDLE_FLUSH_TIMEOUT_SEC = 1.0 

    while not sess.stop_event.is_set():
        now = time.time()
        has_new_text = sess.last_flush_idx < len(sess.text)
        is_idle_timeout = (now - sess.last_input_ts) >= IDLE_FLUSH_TIMEOUT_SEC
        
        if has_new_text and is_idle_timeout:
            logging.debug(f"[{sess.sid}] Flushing due to IDLE timeout ({IDLE_FLUSH_TIMEOUT_SEC}s).")
            enqueue_flushable_sentences(sess, force=True)

        try:
            sentence = sess.tts_q.get(timeout=0.1)
        except queue.Empty:
            sentence = None

        if sentence:
            try:
                chunk_count = 0
                for wav_chunk_bytes in stream_sentence_to_wav_chunks(sentence):
                    if wav_chunk_bytes:
                        # <<<--- 여기서 노이즈 제거 함수 호출! ---
                        enhanced_chunk_bytes = denoise_audio_chunk(wav_chunk_bytes, SAMPLE_RATE)
                        # ----------------------------------------
                        
                        b64 = base64.b64encode(enhanced_chunk_bytes).decode("utf-8")
                        sess.sse_q.put(json.dumps({"type": "audio", "b64wav": b64}))
                        chunk_count += 1
                
                if chunk_count == 0:
                    sess.sse_q.put(json.dumps({"type": "log", "msg": "empty audio stream"}))

            except Exception as e:
                logging.error(f"TTS Error: {e}", exc_info=True)
                sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))

        if (time.time() - last_keepalive) >= KEEPALIVE_SEC:
            sess.sse_q.put(json.dumps({"type": "ping"}))
            last_keepalive = time.time()

    sess.sse_q.put(json.dumps({"type": "end"}))
    print(f"[{sess.sid}] TTS worker stopped.")

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
    sess.last_input_ts = time.time()
    
    enqueue_flushable_sentences(sess, force=False)
    
    return jsonify({"ok": True})

@app.route("/sse_audio")
def sse_audio():
    sid = request.args.get("sid") or str(uuid.uuid4())
    sess = get_or_create_session(sid)

    def event_stream():
        last_ping = time.time()
        while not sess.stop_event.is_set():
            try:
                msg = sess.sse_q.get(timeout=0.5)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                if (time.time() - last_ping) >= KEEPALIVE_SEC:
                    yield 'data: {"type":"ping"}\n\n'
                    last_ping = time.time()

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
    }
    return Response(event_stream(), headers=headers)

# ==============================
# 5) Run
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)