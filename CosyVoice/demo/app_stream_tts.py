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
import re  # <--- [수정] 정규표현식 모듈 추가
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
# 3) TTS Worker & Synthesizer [필터링 로직 대폭 수정]
# ==============================

# [수정] 한글, 영어, 숫자 중 하나라도 포함되어 있는지 확인하는 정규식
HAS_CONTENT_REGEX = re.compile(r'[a-zA-Z0-9가-힣]')

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

        # --- [필터 1] ---
        # 큐에서 처음 꺼낸 텍스트(first_chunk)가 유효한 내용인지 검사
        if not HAS_CONTENT_REGEX.search(first_chunk):
            logging.info(f"[{sess.sid}] [FILTER] First chunk '{first_chunk.strip()}' has no content. Skipping this run.")
            # 유효한 내용이 아니면(예: "!!! " 또는 "   ") 이번 추론 실행을 건너뛰고
            # 다음 텍스트 조각을 기다립니다.
            continue
        
        logging.info(f"[{sess.sid}] First chunk received: '{first_chunk.strip()}'. Starting inference run.")
        text_for_this_run = first_chunk 
        sess.accumulated_audio.clear()

        def text_generator() -> Generator[str, None, None]:
            nonlocal text_for_this_run
            
            # (위에서 이미 검증했으므로) 첫 번째 조각은 바로 yield
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
                            # --- [필터 2] ---
                            # (EOS 신호 수신 시) 남아있는 버퍼가 유효한지 검사
                            if buffer and HAS_CONTENT_REGEX.search(buffer):
                                logging.info(f"[{sess.sid}] [FILTER] EOS marker. Flushing valid buffer: '{buffer.strip()}'")
                                text_for_this_run += buffer 
                                yield buffer
                            elif buffer:
                                logging.info(f"[{sess.sid}] [FILTER] EOS marker. Discarding invalid buffer: '{buffer.strip()}'")
                            
                            buffer = "" # 버퍼 비우기
                            logging.info(f"[{sess.sid}] EOS marker received (Keep Context: {keep_context}). Terminating text generator.")
                            break # text_generator 루프 종료
                    
                    else:
                        chunk = chunk_or_signal
                        buffer += chunk
                        
                        last_space_index = buffer.rfind(' ')
                        if last_space_index != -1:
                            to_yield = buffer[:last_space_index + 1]
                            buffer = buffer[last_space_index + 1:]
                            
                            # --- [필터 3] ---
                            # (공백 기준 분리 시) yield할 조각(to_yield)이 유효한지 검사
                            if HAS_CONTENT_REGEX.search(to_yield):
                                text_for_this_run += to_yield
                                logging.info(f"[{sess.sid}] [FILTER] Yielding valid chunk: '{to_yield.strip()}' | Remaining in buffer: '{buffer.strip()}'")
                                yield to_yield
                            else:
                                # (예: "Hello !!! ") 에서 "!!! " 부분이 여기에 해당
                                logging.info(f"[{sess.sid}] [FILTER] Discarding invalid chunk: '{to_yield.strip()}' | Remaining in buffer: '{buffer.strip()}'")
                                
                except queue.Empty:
                    continue
            
            # (루프가 비정상 종료 시) 마지막 버퍼 검사 (필터 2와 유사)
            if buffer and HAS_CONTENT_REGEX.search(buffer):
                logging.warning(f"[{sess.sid}] [FILTER] Text generator exited loop. Final valid flush: '{buffer.strip()}'")
                text_for_this_run += buffer 
                yield buffer
            elif buffer:
                logging.warning(f"[{sess.sid}] [FILTER] Text generator exited loop. Discarding invalid buffer: '{buffer.strip()}'")
            
            logging.info(f"[{sess.sid}] Text generator finished.")

        # --- [이하 로직은 동일] ---

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
        
        try:
            buf = io.BytesIO()
            torchaudio.save(buf, final_audio, SAMPLE_RATE, format="wav")
            buf.seek(0)
            wav_chunk_bytes = buf.read()
            b64 = base64.b64encode(wav_chunk_bytes).decode("utf-8")
            
            if not sess.keep_context:
                logging.info(f"[{sess.sid}] This was a FINAL batch (keep_context=False). Sending 'final_audio' with Base64.")
                sess.sse_q.put(json.dumps({
                    "type": "final_audio",
                    "b64wav": b64,
                    "text": final_text
                }))
            
            else:
                logging.info(f"[{sess.sid}] This was an INTERIM batch (keep_context=True). Sending 'streaming_audio'.")
                sess.sse_q.put(json.dumps({"type": "streaming_audio", "b64wav": b64}))

        except Exception as e:
             logging.error(f"[{sess.sid}] ❌ Failed to encode/send audio as Base64: {e}")
        
        logging.info(f"[{sess.sid}] Inference run finished. Waiting for next text chunk...")
    
    print(f"[{sess.sid}] TTS worker fully stopped.")


# ==============================
# 4) HTTP Routes
# ==============================
# (이하 /type, /sse_audio, app.run() 등 나머지 코드는 수정할 필요 없이 동일합니다.)
# (app_stream_tts.py의 나머지 부분을 그대로 두시면 됩니다.)

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    logging.info(f"Serving file: {filename} from {OUTPUT_DİR}")
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

        # [참고] 여기서 1차 필터링을 할 수도 있지만, 
        # tts_worker에서 처리하는 것이 더 견고합니다. (예: "Hello"와 "!!!"가 따로 들어오는 경우)
        # 따라서 여기서는 필터링 없이 큐에 넣습니다.
        sess.text_stream_q.put(new_text_chunk)
        logging.info(f"[{sid}] Queued new text chunk: '{new_text_chunk.strip()}'")
        sess.last_sent_len = len(text)
    
    if sess.single_space_count >= 4:
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