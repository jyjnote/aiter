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
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Generator, Any

import torch
import torchaudio
from flask import Flask, request, Response, render_template, jsonify, send_file
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

app = Flask(__name__, template_folder='templates')
CORS(app) 

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
    sse_q: queue.Queue = field(default_factory=queue.Queue)
    text_q_A: queue.Queue = field(default_factory=queue.Queue)
    text_q_B: queue.Queue = field(default_factory=queue.Queue)
    worker_A: Optional[threading.Thread] = None
    worker_B: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    accumulated_audio_chunks: List[torch.Tensor] = field(default_factory=list)
    audio_lock: threading.Lock = field(default_factory=threading.Lock)

SESSIONS: Dict[str, Session] = {}
SESS_LOCK = threading.Lock()

def get_or_create_session(sid: str) -> Session:
    with SESS_LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            sess = Session(sid=sid)
            SESSIONS[sid] = sess
            
            # [수정] tts_worker의 큐 로직이 변경됨
            t_A = threading.Thread(target=tts_worker, args=(sess, sess.text_q_A, "A"), daemon=True)
            t_B = threading.Thread(target=tts_worker, args=(sess, sess.text_q_B, "B"), daemon=True)
            
            t_A.start()
            t_B.start()
            
            sess.worker_A = t_A
            sess.worker_B = t_B
    return sess

# ==============================
# 3) TTS Worker (Batch ID 적용)
# ==============================

HAS_CONTENT_REGEX = re.compile(r'[a-zA-Z0-9가-힣]')

def tts_worker(sess: Session, text_queue: queue.Queue, worker_id: str):
    print(f"[{sess.sid}] TTS 워커 {worker_id} 시작.")

    while not sess.stop_event.is_set():
        try:
            # [수정] 큐에서 (텍스트, keep_context, batch_id) 작업을 한 번에 가져옴
            job_data: Any = text_queue.get(timeout=1.0) 
            text_chunk, keep_context, batch_id = job_data
            
        except queue.Empty:
            continue 
        except Exception as e:
            logging.warning(f"[{sess.sid}][Worker {worker_id}] 작업 큐 에러: {e}")
            continue

        if not HAS_CONTENT_REGEX.search(text_chunk):
            logging.info(f"[{sess.sid}][Worker {worker_id}] [FILTER] '{text_chunk.strip()}' 건너뜁니다.")
            continue
        
        logging.info(f"[{sess.sid}][Worker {worker_id}] 작업 시작 (Batch ID: {batch_id}). 텍스트: '{text_chunk.strip()}'")
        
        all_chunks_for_this_run = []
        try:
            worker_session_id = f"{sess.sid}_{worker_id}"
            
            for out in cosyvoice.inference_instruct2(
                    tts_text=text_chunk, 
                    instruct_text="",
                    prompt_speech_16k=prompt_speech_16k,
                    zero_shot_spk_id="",
                    stream=False,
                    speed=1.0,
                    text_frontend=True,
                    session_id=worker_session_id, 
                    keep_context=keep_context
            ):
                audio_chunk = out["tts_speech"].cpu()
                if audio_chunk.numel() > 0:
                    all_chunks_for_this_run.append(audio_chunk)

        except Exception as e:
            logging.error(f"[{sess.sid}][Worker {worker_id}] TTS (instruct2) Error (Batch ID: {batch_id}): {e}", exc_info=True)
            sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))
            continue

        if not all_chunks_for_this_run:
            logging.warning(f"[{sess.sid}][Worker {worker_id}] 오디오가 생성되지 않았습니다 (Batch ID: {batch_id}).")
            continue
            
        final_audio = torch.cat(all_chunks_for_this_run, dim=1)
        
        # 다운로드를 위해 오디오 조각 저장
        try:
            with sess.audio_lock:
                # [수정] 순서 보장을 위해 (batch_id, audio) 쌍으로 저장
                sess.accumulated_audio_chunks.append((batch_id, final_audio))
            logging.info(f"[{sess.sid}][Worker {worker_id}] 오디오 조각 저장 완료 (Batch ID: {batch_id})")
        except Exception as e:
            logging.error(f"[{sess.sid}][Worker {worker_id}] 오디오 조각 저장 실패: {e}")

        try:
            buf = io.BytesIO()
            torchaudio.save(buf, final_audio, SAMPLE_RATE, format="wav")
            buf.seek(0)
            wav_chunk_bytes = buf.read()
            b64 = base64.b64encode(wav_chunk_bytes).decode("utf-8")
            
            # [수정] SSE 메시지에 "batchId" 추가
            sess.sse_q.put(json.dumps({
                "type": "audio_chunk",
                "b64wav": b64,
                "text": text_chunk.strip(),
                "batchId": batch_id # [핵심] 순번 태그
            }))
            logging.info(f"[{sess.sid}][Worker {worker_id}] 오디오 청크를 SSE 큐로 전송 완료 (Batch ID: {batch_id}).")

            # [수정] 'end' 메시지에도 "batchId" 추가
            if not keep_context:
                sess.sse_q.put(json.dumps({"type": "end", "batchId": batch_id}))

        except Exception as e:
             logging.error(f"[{sess.sid}][Worker {worker_id}] ❌ Failed to encode/send audio as Base64: {e}")
        
        logging.info(f"[{sess.sid}][Worker {worker_id}] 작업 완료 (Batch ID: {batch_id}). 다음 작업 대기 중...")
    
    print(f"[{sess.sid}] TTS 워커 {worker_id}가 중지되었습니다.")


# ==============================
# 4) HTTP Routes (Batch ID 적용)
# ==============================

@app.route("/")
def index():
    return render_template("index_pingpong.html") 

@app.route("/synthesize_chunk", methods=["POST"])
def synthesize_chunk():
    data = request.get_json(force=True)
    sid = data.get("sid") or str(uuid.uuid4())
    text_chunk = data.get("text_chunk", "")
    worker_id = data.get("worker_id", "A") 
    is_final = data.get("is_final", False)
    batch_id = data.get("batch_id", 0) # [수정] 클라이언트로부터 batch_id 수신

    sess = get_or_create_session(sid)
    
    q_to_use = sess.text_q_A if worker_id == "A" else sess.text_q_B
    
    if text_chunk:
        keep_context = not is_final
        # [수정] (텍스트, keep_context, batch_id) 튜플을 큐에 한 번만 넣음
        q_to_use.put((text_chunk, keep_context, batch_id))
        logging.info(f"[{sid}] 텍스트 조각을 [Worker {worker_id}] 큐에 추가 (Batch ID: {batch_id}, Final: {is_final})")
    else:
        logging.warning(f"[{sid}] 빈 텍스트 조각 수신 (Batch ID: {batch_id}). 무시.")

    return jsonify({"ok": True})

@app.route("/download_audio")
def download_audio():
    sid = request.args.get("sid")
    if not sid:
        return "Session ID(sid)가 필요합니다.", 400
        
    sess = SESSIONS.get(sid)
    if not sess:
        return "세션을 찾을 수 없습니다.", 404

    try:
        with sess.audio_lock:
            if not sess.accumulated_audio_chunks:
                return "생성된 오디오가 없습니다.", 404
            
            logging.info(f"[{sid}] 오디오 다운로드 요청. {len(sess.accumulated_audio_chunks)}개 조각 병합 중...")
            
            # [수정] batch_id 순서대로 정렬
            sorted_chunks = sorted(sess.accumulated_audio_chunks, key=lambda x: x[0])
            
            # 정렬된 오디오 텐서만 추출
            final_audio_tensors = [audio for batch_id, audio in sorted_chunks]
            
            full_audio = torch.cat(final_audio_tensors, dim=1)
        
        buf = io.BytesIO()
        torchaudio.save(buf, full_audio, SAMPLE_RATE, format="wav")
        buf.seek(0)
        
        logging.info(f"[{sid}] WAV 파일 생성 완료. 다운로드 전송 시작...")
        
        return send_file(
            buf,
            as_attachment=True,
            download_name=f"synthesis_{sid}.wav",
            mimetype="audio/wav"
        )
        
    except Exception as e:
        logging.error(f"[{sid}] 오디오 다운로드 파일 생성 중 오류: {e}")
        return "파일 생성 중 오류 발생", 500


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