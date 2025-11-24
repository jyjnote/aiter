#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import io
import time  # time.perf_counter()를 위해 사용
import json
import base64
import queue
import threading
import uuid
import re 
import csv  # <--- CSV 모듈

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

# --- CSV 파일 설정 ---
MEASUREMENT_FILE = os.path.join(THIS_DIR, 'delay_measurements.csv')
MEASUREMENT_LOCK = threading.Lock()

# CSV 파일 헤더 초기화
try:
    with MEASUREMENT_LOCK:
        with open(MEASUREMENT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # [수정] '생성 시간(초)' 및 'RTF' 컬럼 추가
            writer.writerow(['측정 시간', '배치 순서', '재생 딜레이(초)', '오디오 길이(초)', '배치 간 시간(초)', '생성 시간(초)', 'RTF'])
    logging.info(f"측정 로그 파일이 {MEASUREMENT_FILE} (으)로 초기화되었습니다.")
except IOError as e:
    logging.error(f"CSV 파일 쓰기 실패: {e}")

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
# 3) TTS Worker & Synthesizer [RTF 측정 추가]
# ==============================

HAS_CONTENT_REGEX = re.compile(r'[a-zA-Z0-9가-힣]')

def tts_worker(sess: Session):
    print(f"[{sess.sid}] TTS worker started. Waiting for first text chunk...")

    last_ready_time = None       
    last_audio_duration = 0.0    
    total_gap_time = 0.0         
    total_gap_events = 0         

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
                    sess.sse_q.put(json.dumps({"type": "end"}))
                continue
        
        first_chunk = first_chunk_or_signal

        if not HAS_CONTENT_REGEX.search(first_chunk):
            continue
        
        logging.info(f"[{sess.sid}] Starting inference run for: '{first_chunk.strip()}'")
        text_for_this_run = first_chunk 
        sess.accumulated_audio.clear()

        def text_generator() -> Generator[str, None, None]:
            nonlocal text_for_this_run
            yield first_chunk
            buffer = "" 
            while not sess.stop_event.is_set():
                try:
                    chunk_or_signal = sess.text_stream_q.get(timeout=0.1) 
                    if isinstance(chunk_or_signal, tuple):
                        marker, keep_context = chunk_or_signal
                        sess.keep_context = keep_context
                        if marker is None:
                            if buffer and HAS_CONTENT_REGEX.search(buffer):
                                text_for_this_run += buffer 
                                yield buffer
                            break 
                    else:
                        chunk = chunk_or_signal
                        buffer += chunk
                        last_space_index = buffer.rfind(' ')
                        if last_space_index != -1:
                            to_yield = buffer[:last_space_index + 1]
                            buffer = buffer[last_space_index + 1:]
                            if HAS_CONTENT_REGEX.search(to_yield):
                                text_for_this_run += to_yield
                                yield to_yield
                except queue.Empty:
                    continue
            if buffer and HAS_CONTENT_REGEX.search(buffer):
                text_for_this_run += buffer 
                yield buffer

        use_frontend = True 
        all_chunks_for_this_run = []
        
        # --- [RTF 측정 시작] ---
        inference_start_time = time.perf_counter()
        
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
            logging.error(f"[{sess.sid}] TTS error: {e}")
            continue

        # --- [RTF 측정 종료] ---
        inference_end_time = time.perf_counter()
        inference_duration = inference_end_time - inference_start_time

        if not all_chunks_for_this_run:
            continue
            
        final_audio = torch.cat(all_chunks_for_this_run, dim=1)
        final_text = text_for_this_run.strip()

        # --- [측정 로직 (RTF 포함)] ---
        current_ready_time = time.perf_counter()
        current_audio_duration = final_audio.shape[1] / SAMPLE_RATE
        
        # RTF 계산: (생성에 걸린 시간) / (생성된 오디오 길이)
        # 낮을수록 빠름 (0.5 means generating 10s audio took 5s)
        current_rtf = inference_duration / current_audio_duration if current_audio_duration > 0 else 0.0

        playback_gap = 0.0
        time_between_batches = 0.0

        if last_ready_time is not None:
            time_between_batches = current_ready_time - last_ready_time
            playback_gap = time_between_batches - last_audio_duration
            
            total_gap_time += playback_gap
            total_gap_events += 1
            
            logging.info(f"[{sess.sid}] [MEASURE] Gap: {playback_gap:.4f}s | RTF: {current_rtf:.4f}")

            try:
                with MEASUREMENT_LOCK:
                    with open(MEASUREMENT_FILE, 'a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.strftime('%Y-%m-%d %H:%M:%S'), 
                            total_gap_events, 
                            f"{playback_gap:.4f}", 
                            f"{current_audio_duration:.4f}",
                            f"{time_between_batches:.4f}",
                            f"{inference_duration:.4f}", # 생성 시간
                            f"{current_rtf:.4f}"         # RTF
                        ])
            except IOError as e:
                logging.error(f"CSV 쓰기 오류: {e}")

        last_ready_time = current_ready_time
        last_audio_duration = current_audio_duration
        # --- [측정 로직 끝] ---

        try:
            buf = io.BytesIO()
            torchaudio.save(buf, final_audio, SAMPLE_RATE, format="wav")
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            
            msg_type = "streaming_audio" if sess.keep_context else "final_audio"
            msg = {"type": msg_type, "b64wav": b64}
            if not sess.keep_context:
                msg["text"] = final_text
            
            sess.sse_q.put(json.dumps(msg))

        except Exception as e:
             logging.error(f"Encoding error: {e}")
        
    print(f"[{sess.sid}] TTS worker fully stopped.")


# ==============================
# 4) HTTP Routes
# ==============================
@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
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
    
    if len(text) > sess.last_sent_len:
        new_text_chunk = text[sess.last_sent_len:]
        if new_text_chunk.endswith(' ') and not new_text_chunk.endswith('  '):
            sess.single_space_count += 1
        elif not new_text_chunk.strip() == "":
            sess.single_space_count = 0

        sess.text_stream_q.put(new_text_chunk)
        sess.last_sent_len = len(text)
    
    force_from_server = False
    if sess.single_space_count >= 2:
        force_from_server = True
        sess.single_space_count = 0

    if force_from_client:
        sess.text_stream_q.put((None, False))
    elif force_from_server:
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
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(event_stream(), headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)