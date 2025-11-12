#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import io
import time
import json
import base64
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Generator, Any

import torch
import torchaudio
from flask import Flask, request, Response, render_template, jsonify, send_from_directory
from flask_cors import CORS 

# ==============================
# 0) App & Model Init
# ==============================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(THIS_DIR)

sys.path.append(PROJ_DIR)
sys.path.append(os.path.join(PROJ_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

app = Flask(__name__)
CORS(app) 

# [수정] 오디오 파일 저장을 비활성화합니다.
SAVE_ACCUMULATED_AUDIO = False

OUTPUT_DIR = os.path.join(PROJ_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Audio output directory set to: {OUTPUT_DIR}")

MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice-KSS-Finetuned")
PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"{MODEL_DIR} does not exist!")
if not os.path.isfile(PROMPT_WAV):
    raise FileNotFoundError(f"{PROMPT_WAV} not found!")

print("서버 시작: CosyVoice2 로드…")
cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=False)
prompt_speech_16k = load_wav(PROMPT_WAV, 16000)
print("모델 로드 완료.")

SAMPLE_RATE = cosyvoice.sample_rate

# ==============================
# 1) Session Management
# ==============================
@dataclass
class Session:
    sid: str
    last_sent_len: int = 0
    sse_q: queue.Queue = field(default_factory=queue.Queue)
    text_stream_q: queue.Queue = field(default_factory=queue.Queue)
    worker_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    accumulated_audio: List[torch.Tensor] = field(default_factory=list)
    single_space_count: int = 0
    keep_context: bool = True

SESSIONS: Dict[str, Session] = {}
SESS_LOCK = threading.Lock()

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
# 3) TTS Worker & Synthesizer [수정된 버전]
# ==============================
def tts_worker(sess: Session):
    print(f"[{sess.sid}] TTS worker started. Waiting for first text chunk...")

    while not sess.stop_event.is_set():
        
        try:
            first_chunk_or_signal: Any = sess.text_stream_q.get(timeout=1.0) 
        except queue.Empty:
            continue 

        if isinstance(first_chunk_or_signal, tuple):
            marker, keep_context = first_chunk_or_signal
            sess.keep_context = keep_context
            
            if marker is None:
                # [수정] 'end' 메시지는 구두점(keep_context=False)일 때만 전송
                if not keep_context:
                    logging.info(f"[{sess.sid}] Got EOS Signal (Keep Context: False). Sending 'end'.")
                    sess.sse_q.put(json.dumps({"type": "end"}))
                else:
                    logging.info(f"[{sess.sid}] Got EOS Signal (Keep Context: True). Not sending 'end' message.")
                continue
            else:
                logging.warning(f"[{sess.sid}] Received unexpected tuple: {first_chunk_or_signal}")
                continue
        
        first_chunk = first_chunk_or_signal
        logging.info(f"[{sess.sid}] First chunk received. Starting inference run.")
        text_for_this_run = first_chunk 

        sess.accumulated_audio.clear()

        def text_generator() -> Generator[str, None, None]:
            nonlocal text_for_this_run
            
            logging.info(f"[{sess.sid}] Yielding first chunk: '{first_chunk.strip()}'")
            yield first_chunk
            
            buffer = "" 
            while not sess.stop_event.is_set():
                try:
                    chunk_or_signal = sess.text_stream_q.get(timeout=0.1) 
                    
                    if isinstance(chunk_or_signal, tuple):
                        marker, keep_context = chunk_or_signal
                        sess.keep_context = keep_context
                        
                        if marker is None:
                            if buffer:
                                logging.info(f"[{sess.sid}] EOS marker. Flushing buffer: '{buffer}'")
                                text_for_this_run += buffer 
                                yield buffer
                                buffer = ""
                            logging.info(f"[{sess.sid}] EOS marker received (Keep Context: {keep_context}). Terminating text generator.")
                            break 
                    
                    else:
                        chunk = chunk_or_signal
                        buffer += chunk
                        
                        last_space_index = buffer.rfind(' ')
                        if last_space_index != -1:
                            to_yield = buffer[:last_space_index + 1]
                            buffer = buffer[last_space_index + 1:]
                            text_for_this_run += to_yield
                            logging.info(f"[{sess.sid}] Yielding by space: '{to_yield.strip()}' | Remaining in buffer: '{buffer.strip()}'")
                            yield to_yield
                                
                except queue.Empty:
                    continue
            
            if buffer:
                logging.warning(f"[{sess.sid}] Text generator exited loop unexpectedly. Final flush: '{buffer}'")
                text_for_this_run += buffer 
                yield buffer
            
            logging.info(f"[{sess.sid}] Text generator finished.")


        use_frontend = True 
        logging.info(f"[{sess.sid}] Starting new TTS inference loop. Using Text Frontend: {use_frontend}")
        
        all_chunks_for_this_run = []
        try:
            for out in cosyvoice.inference_instruct2(
                    tts_text=text_generator(), 
                    instruct_text="",
                    prompt_speech_16k=prompt_speech_16k,
                    zero_shot_spk_id="",
                    stream=True,
                    speed=1.0,
                    text_frontend=use_frontend,
                    session_id=sess.sid,
                    keep_context=sess.keep_context
            ):
                audio_chunk = out["tts_speech"].cpu()
                if audio_chunk.numel() > 0:
                    all_chunks_for_this_run.append(audio_chunk)

        except Exception as e:
            logging.error(f"[{sess.sid}] TTS (instruct2) Error: {e}", exc_info=True)
            sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))
            continue

        if not all_chunks_for_this_run:
            logging.warning(f"[{sess.sid}] No audio was generated for this run.")
            continue
            
        final_audio = torch.cat(all_chunks_for_this_run, dim=1)
        final_text = text_for_this_run.strip()

        # --- [로직 수정 시작] ---
        
        # [수정] 이제 keep_context 여부와 관계없이 모든 오디오를 Base64로 인코딩하여
        # 'streaming_audio' 또는 'final_audio' 타입으로 보냅니다.
        
        try:
            buf = io.BytesIO()
            torchaudio.save(buf, final_audio, SAMPLE_RATE, format="wav")
            buf.seek(0)
            wav_chunk_bytes = buf.read()
            b64 = base64.b64encode(wav_chunk_bytes).decode("utf-8")
            
            if not sess.keep_context:
                # === 최종 배치 (Final Batch) ===
                # (구두점 입력 시, keep_context=False)
                # 'final_audio' 타입으로 Base64와 텍스트를 함께 보냅니다.
                logging.info(f"[{sess.sid}] This was a FINAL batch (keep_context=False). Sending 'final_audio' with Base64.")
                sess.sse_q.put(json.dumps({
                    "type": "final_audio",
                    "b64wav": b64,
                    "text": final_text
                }))
            
            else:
                # === 중간 배치 (Interim Batch) ===
                # (누적 공백 2회 시, keep_context=True)
                # 'streaming_audio' 타입으로 Base64만 보냅니다.
                logging.info(f"[{sess.sid}] This was an INTERIM batch (keep_context=True). Sending 'streaming_audio'.")
                sess.sse_q.put(json.dumps({"type": "streaming_audio", "b64wav": b64}))

        except Exception as e:
             logging.error(f"[{sess.sid}] ❌ Failed to encode/send audio as Base64: {e}")

        # --- [로직 수정 끝] ---
        
        logging.info(f"[{sess.sid}] Inference run finished. Waiting for next text chunk...")
    
    print(f"[{sess.sid}] TTS worker fully stopped.")


# ==============================
# 4) HTTP Routes
# ==============================

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    # 이 라우트는 이제 사용되지 않지만, 호환성을 위해 남겨둡니다.
    logging.info(f"Serving file: {filename} from {OUTPUT_DIR}")
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/type", methods=["POST"])
def type_event():
    data = request.get_json(force=True)
    sid = data.get("sid") or str(uuid.uuid4())
    text = data.get("text", "")
    force_from_client = data.get("force", False)

    sess = get_or_create_session(sid)
    
    new_text_chunk = None
    force_from_server = False
    
    if len(text) > sess.last_sent_len:
        new_text_chunk = text[sess.last_sent_len:]
        
        if new_text_chunk.endswith(' ') and not new_text_chunk.endswith('  '):
            sess.single_space_count += 1
            logging.info(f"[{sid}] Single space detected. Count: {sess.single_space_count}")
        elif not new_text_chunk.strip() == "":
            logging.info(f"[{sid}] Non-space text detected. Resetting space count.")
            sess.single_space_count = 0

        sess.text_stream_q.put(new_text_chunk)
        logging.info(f"[{sid}] Queued new text chunk: '{new_text_chunk.strip()}'")
        sess.last_sent_len = len(text)
    
    if sess.single_space_count >= 2:
        logging.info(f"[{sid}] TRIGGER HIT (2 cumulative spaces). Setting server_force=true.")
        force_from_server = True
        sess.single_space_count = 0

    if force_from_client:
        logging.info(f"[{sid}] Force flush (Client). Queueing EOS (Keep Context: False).")
        sess.text_stream_q.put((None, False))
    elif force_from_server:
        logging.info(f"[{sid}] Force flush (Server). Queueing EOS (Keep Context: True).")
        sess.text_stream_q.put((None, True))

    return jsonify({"ok": True})

@app.route("/sse_audio")
def sse_audio():
    sid = request.args.get("sid") or str(uuid.uuid4())
    sess = get_or_create_session(sid)

    def event_stream():
        while not sess.stop_event.is_set():
            try:
                msg = sess.sse_q.get(timeout=0.5)
                yield f"data: {msg}\n\n"
                if '"type":"end"' in msg:
                    logging.info(f"[{sid}] SSE 'end' marker sent to client.")
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

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
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)