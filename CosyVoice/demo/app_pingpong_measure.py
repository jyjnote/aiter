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
import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any

import torch
import torchaudio
from flask import Flask, request, Response, render_template, jsonify, send_file
from flask_cors import CORS 

# [이전 설정 코드들 생략... 위와 동일]
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(THIS_DIR)
sys.path.append(PROJ_DIR)
sys.path.append(os.path.join(PROJ_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)

app = Flask(__name__)
CORS(app) 

# 모델 로드
MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice-KSS-Finetuned")
PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")
print("서버 시작: CosyVoice2 로드…")
cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=False)
prompt_speech_16k = load_wav(PROMPT_WAV, 16000)
print("모델 로드 완료.")
SAMPLE_RATE = cosyvoice.sample_rate

# --- CSV 파일 설정 ---
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
MEASUREMENT_FILE = os.path.join(THIS_DIR, f'delay_measurements_{current_time_str}.csv')
MEASUREMENT_LOCK = threading.Lock()

# [수정] CSV 헤더 변경: 순수 타임스탬프 위주로 기록
with MEASUREMENT_LOCK:
    with open(MEASUREMENT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'BatchID', 
            'Worker', 
            'Request_Start_Time(Epoch)',  # 요청 시작 시각 (절대시간)
            'Gen_End_Time(Epoch)',        # 생성 완료 시각 (절대시간)
            'Audio_Duration(Sec)',        # 오디오 길이
            'Inference_Time(Sec)',        # 생성 소요 시간
            'RTF'
        ])

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
            t_A = threading.Thread(target=tts_worker, args=(sess, sess.text_q_A, "A"), daemon=True)
            t_B = threading.Thread(target=tts_worker, args=(sess, sess.text_q_B, "B"), daemon=True)
            t_A.start(); t_B.start()
            sess.worker_A = t_A; sess.worker_B = t_B
    return sess

HAS_CONTENT_REGEX = re.compile(r'[a-zA-Z0-9가-힣]')

def tts_worker(sess: Session, text_queue: queue.Queue, worker_id: str):
    print(f"[{sess.sid}] TTS 워커 {worker_id} 대기 중...")
    
    while not sess.stop_event.is_set():
        try:
            job_data = text_queue.get(timeout=1.0)
            text_chunk, keep_context, batch_id = job_data
        except queue.Empty:
            continue
        
        if not HAS_CONTENT_REGEX.search(text_chunk):
            continue

        logging.info(f"[Batch {batch_id}] Worker {worker_id} 시작: '{text_chunk.strip()}'")

        # [측정 1] 시작 시간 기록 (절대 시간)
        req_start_time = time.time()       # 타임스탬프 기록용
        perf_start = time.perf_counter()   # RTF 계산용 정밀 시간

        all_chunks = []
        try:
            worker_session_id = f"{sess.sid}_{worker_id}"
            # 핑퐁이므로 stream=False로 한 번에 생성 (권장)
            for out in cosyvoice.inference_instruct2(
                tts_text=text_chunk, instruct_text="", prompt_speech_16k=prompt_speech_16k,
                zero_shot_spk_id="", stream=False, speed=1.0, text_frontend=True,
                session_id=worker_session_id, keep_context=keep_context
            ):
                if out["tts_speech"].numel() > 0:
                    all_chunks.append(out["tts_speech"].cpu())
        except Exception as e:
            logging.error(f"Error Batch {batch_id}: {e}")
            continue

        if not all_chunks: continue

        # [측정 2] 종료 시간 기록
        perf_end = time.perf_counter()
        gen_end_time = time.time()  # 완료 절대 시간

        inference_sec = perf_end - perf_start
        final_audio = torch.cat(all_chunks, dim=1)
        audio_sec = final_audio.shape[1] / SAMPLE_RATE
        rtf = inference_sec / audio_sec if audio_sec > 0 else 0.0

        # [기록] CSV 저장 (Gap 계산은 여기서 안 함!)
        try:
            with MEASUREMENT_LOCK:
                with open(MEASUREMENT_FILE, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        batch_id, 
                        worker_id, 
                        f"{req_start_time:.4f}", # 언제 시작했는지
                        f"{gen_end_time:.4f}",   # 언제 끝났는지 (준비 완료)
                        f"{audio_sec:.4f}", 
                        f"{inference_sec:.4f}", 
                        f"{rtf:.4f}"
                    ])
        except Exception as e:
            logging.error(f"CSV Error: {e}")

        # 전송
        try:
            buf = io.BytesIO()
            torchaudio.save(buf, final_audio, SAMPLE_RATE, format="wav")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            sess.sse_q.put(json.dumps({
                "type": "audio_chunk", "b64wav": b64, "batchId": batch_id
            }))
            if not keep_context:
                sess.sse_q.put(json.dumps({"type": "end", "batchId": batch_id}))
        except: pass
        
        logging.info(f"[Batch {batch_id}] 완료 (RTF: {rtf:.2f})")

# --- Routes (기존과 동일) ---
@app.route("/synthesize_chunk", methods=["POST"])
def synthesize_chunk():
    data = request.get_json(force=True)
    sid = data.get("sid") or str(uuid.uuid4())
    text = data.get("text_chunk", "")
    # 클라이언트가 지정한 worker_id를 무시하고 서버가 직접 배분하고 싶으면 여기서 수정
    # 지금은 클라이언트 제어 방식을 유지합니다.
    worker_id = data.get("worker_id", "A") 
    batch_id = data.get("batch_id", 0)
    
    sess = get_or_create_session(sid)
    q = sess.text_q_A if worker_id == "A" else sess.text_q_B
    if text: q.put((text, True, batch_id)) # keep_context=True로 고정 (핑퐁에선 보통 문맥 끊음)
    
    return jsonify({"ok": True})

@app.route("/sse_audio")
def sse_audio():
    # (기존과 동일)
    sid = request.args.get("sid")
    sess = get_or_create_session(sid if sid else str(uuid.uuid4()))
    def stream():
        while True:
            try: yield f"data: {sess.sse_q.get(timeout=1)}\n\n"
            except: yield 'data: {"type":"ping"}\n\n'
    return Response(stream(), headers={"Content-Type": "text/event-stream"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)