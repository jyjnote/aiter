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
from typing import Dict, Optional, List

import torch
import torchaudio
from flask import Flask, request, Response, render_template, jsonify

# ==============================
# 0) App & Model Init
# ==============================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(THIS_DIR)

sys.path.append(PROJ_DIR)
sys.path.append(os.path.join(PROJ_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

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

FLUSH_INTERVAL_SEC = 0.5
KEEPALIVE_SEC      = 15.0

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
            sentences.append(rest)
        sess.last_flush_idx = len(sess.text)

    for s in sentences:
        logging.debug(f"[{sess.sid}] enqueue sentence: {s[:80]}{'...' if len(s)>80 else ''}")
        sess.tts_q.put(s)
        sess.last_flush_ts = time.time()

# ==============================
# 3) TTS Worker & Synthesizer
# ==============================
# <--- 수정: 언어 태그를 사용하는 최종 버전 ---
def synth_sentence_to_wav_bytes(sentence: str) -> bytes:
    # 1. 언어 감지 및 태그 부착
    first_char = sentence.lstrip()[:1]
    is_ko = is_korean(first_char)

    if is_ko:
        tagged_sentence = f"<|ko|>{sentence}"
        print(f"[TTS Worker] DEBUG: Korean detected. Applying <|ko|> tag.")
    else:
        tagged_sentence = f"<|en|>{sentence}"
        print(f"[TTS Worker] DEBUG: English/Other detected. Applying <|en|> tag.")

    wav_parts = []

    # 2. 제너레이터를 통해 문장 전달
    def text_generator():
        yield ("RESET", tagged_sentence)

    # 3. GitHub 이슈에서 권장하는 방식으로 TTS 호출
    for out in cosyvoice.inference_zero_shot_typing(
            text_stream=text_generator(),
            prompt_text="", 
            prompt_speech_16k=prompt_speech_16k,
            zero_shot_spk_id="",
            stream=False,
            speed=1.0,
            text_frontend=True,
            interleave_prompt_in_llm=False
        ):
        wav_parts.append(out["tts_speech"].cpu())

    if not wav_parts:
        return b""

    wav_cat = torch.cat(wav_parts, dim=1)
    buf = io.BytesIO()
    torchaudio.save(buf, wav_cat, SAMPLE_RATE, format="wav")
    buf.seek(0)
    return buf.read()


# <--- 수정: 지능형 타임아웃 로직이 적용된 최종 버전 ---
def tts_worker(sess: Session):
    last_keepalive = time.time()
    
    print(f"[{sess.sid}] TTS worker started.")

    ENGLISH_FAST_TIMEOUT_SEC = 0.8
    ENGLISH_SLOW_TIMEOUT_SEC = 2.0
    KOREAN_TIMEOUT_SEC = 2.0
    
    FLUSH_PUNCT = re.compile(r"[\.!\?…。\！？,;:]")

    while not sess.stop_event.is_set():
        now = time.time()

        if (now - sess.last_flush_ts) >= FLUSH_INTERVAL_SEC and sess.last_flush_idx < len(sess.text):
            
            last_real_char = sess.text.rstrip()[-1:] if sess.text.rstrip() else ""
            is_ko_mode = is_korean(last_real_char)

            should_flush = False
            flush_reason = ""
            
            last_char = sess.text[-1:] if sess.text else ""

            if is_ko_mode:
                if FLUSH_PUNCT.match(last_char):
                    should_flush = True
                    flush_reason = f"Korean Punctuation ('{last_char}')"
                elif last_char.isspace():
                    should_flush = True
                    flush_reason = "Korean Space"
                elif (now - sess.last_input_ts) >= KOREAN_TIMEOUT_SEC:
                    should_flush = True
                    flush_reason = f"Korean Timeout ({KOREAN_TIMEOUT_SEC}s)"
            else:
                flush_on_punct = FLUSH_PUNCT.match(last_char)
                flush_on_fast_timeout = last_char.isspace() and (now - sess.last_input_ts) >= ENGLISH_FAST_TIMEOUT_SEC
                flush_on_slow_timeout = (now - sess.last_input_ts) >= ENGLISH_SLOW_TIMEOUT_SEC and sess.text[sess.last_flush_idx:].strip()
                
                if flush_on_punct:
                    should_flush = True
                    flush_reason = f"English Punctuation ('{last_char}')"
                elif flush_on_fast_timeout:
                    should_flush = True
                    flush_reason = f"English Fast Timeout after space ({ENGLISH_FAST_TIMEOUT_SEC}s)"
                elif flush_on_slow_timeout:
                    should_flush = True
                    flush_reason = f"English Slow Fallback Timeout ({ENGLISH_SLOW_TIMEOUT_SEC}s)"

            if should_flush:
                print(f"[{sess.sid}] DEBUG: Flushing triggered by: {flush_reason}.")
                enqueue_flushable_sentences(sess, force=True)

        try:
            sentence = sess.tts_q.get(timeout=0.1)
        except queue.Empty:
            sentence = None

        if sentence:
            try:
                wav_bytes = synth_sentence_to_wav_bytes(sentence)
                if wav_bytes:
                    b64 = base64.b64encode(wav_bytes).decode("utf-8")
                    sess.sse_q.put(json.dumps({"type": "audio", "b64wav": b64}))
                else:
                    sess.sse_q.put(json.dumps({"type": "log", "msg": "empty audio"}))
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