#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from flask import Flask, request, Response, render_template_string, jsonify

# ==============================
# 0) App & Model Init
# ==============================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../CosyVoice/demo
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
PUNCT = re.compile(r"[\.!\?…。\！？]")

# ==============================
# 1) 세션 관리
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

FLUSH_INTERVAL_SEC = 1.0
WORD_TIMEOUT_SEC   = 2.0
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
# 2) 문장 추출 & 큐잉
# ==============================
def enqueue_flushable_sentences(sess: Session, force: bool = False):
    new_segment = sess.text[sess.last_flush_idx:]
    if not new_segment:
        return

    consumed = 0
    sentences: List[str] = []

    for m in re.finditer(r"[^\.!\?…。\！？]*[\.!\?…。\！？]", new_segment):
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
# 3) TTS 워커
# ==============================
def synth_sentence_to_wav_bytes(sentence: str) -> bytes:
    wav_parts = []

    # CosyVoice 내부 inference 호출
    for out in cosyvoice.inference_zero_shot_typing(
            text_stream=[sentence],
            prompt_text="<|endofprompt|>",
            prompt_speech_16k=prompt_speech_16k,
            zero_shot_spk_id="",
            stream=False,   # 즉시합성
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


def tts_worker(sess: Session):
    last_keepalive = time.time()

    while not sess.stop_event.is_set():
        now = time.time()

        # --- 혼합 전략 ---
        if (now - sess.last_flush_ts) >= FLUSH_INTERVAL_SEC and sess.last_flush_idx < len(sess.text):
            last_char = sess.text[-1] if sess.text else ""
            if last_char.isspace() or PUNCT.match(last_char):
                enqueue_flushable_sentences(sess, force=True)
            elif (now - sess.last_input_ts) >= WORD_TIMEOUT_SEC:
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
                sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))

        if (time.time() - last_keepalive) >= KEEPALIVE_SEC:
            sess.sse_q.put(json.dumps({"type": "ping"}))
            last_keepalive = time.time()

    sess.sse_q.put(json.dumps({"type": "end"}))

# ==============================
# 4) HTTP Routes
# ==============================
@app.route("/")
def index():
    html = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Streaming Text → TTS</title>
  <style>
    body {
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Noto Sans KR', sans-serif;
      line-height: 1.4;
      padding: 24px;
    }
    textarea {
      width: 100%;
      height: 400px;        /* ✅ 크기 크게 */
      font-size: 18px;      /* ✅ 글자 크게 */
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #ccc;
      resize: vertical;     /* ✅ 세로 크기 조절 가능 */
    }
  </style>
</head>
<body>
  <h1>Streaming Text → TTS (혼합 전략)</h1>
  <textarea id="ta" placeholder="여기에 계속 타이핑 해보세요."></textarea>
  <audio id="player" controls></audio>
  <div id="log" style="white-space: pre-wrap;"></div>
<script>
(function() {
  const sid = crypto.randomUUID();
  const ta = document.getElementById('ta');
  const log = document.getElementById('log');
  const player = document.getElementById('player');

  const q = [];
  let playing = false;
  function enqueueAndPlay(b64) {
    const byteChars = atob(b64);
    const byteNums = new Array(byteChars.length);
    for (let i=0; i<byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
    const blob = new Blob([new Uint8Array(byteNums)], {type: 'audio/wav'});
    const url = URL.createObjectURL(blob);
    q.push(url);
    if (!playing) playNext();
  }
  function playNext() {
    if (q.length === 0) { playing = false; return; }
    playing = true;
    const url = q.shift();
    player.src = url;
    player.play().catch(err => {
      log.textContent += "\\n[AUDIO] play error: " + err;
      playing = false;
    });
  }
  player.onended = () => playNext();

  const es = new EventSource("/sse_audio?sid=" + encodeURIComponent(sid));
  es.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'audio' && msg.b64wav) {
        enqueueAndPlay(msg.b64wav);
      }
    } catch (err) {
      log.textContent += "\\n[SSE] parse error: " + err;
    }
  };

  function sendText(force=false) {
    const text = ta.value;
    fetch('/type', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ sid, text, force })
    }).catch(()=>{});
  }

  ta.addEventListener('input', () => sendText(false));
})();
</script>
</body>
</html>
    """
    return render_template_string(html)

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
