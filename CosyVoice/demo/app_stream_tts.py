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
from typing import Dict, Optional, List, Generator
from collections import defaultdict

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

SAVE_ACCUMULATED_AUDIO = True

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
cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=True) 
prompt_speech_16k = load_wav(PROMPT_WAV, 16000)
print("모델 로드 완료.")

SAMPLE_RATE = cosyvoice.sample_rate

# [핵심] 병렬 작업자 수 설정
NUM_LLM_WORKERS = 2 
NUM_CONVERTER_WORKERS = 2 

# ==============================
# 1) Session Management (Task-based Rewrite)
# ==============================
@dataclass
class Session:
    sid: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    
    # --- Input Buffering ---
    text_buffer: str = "" # 'force=False' 텍스트를 누적하는 버퍼
    task_q: queue.Queue = field(default_factory=queue.Queue) # (task_id, text_chunk) 덩어리를 워커에게 전달하는 큐
    next_task_id: int = 0 # 다음 작업에 할당할 ID

    # --- Output Buffering (sse_audio가 사용) ---
    sse_q: queue.Queue = field(default_factory=queue.Queue) # (task_id, msg_json) - final_audio, end, log 메시지용
    
    # { task_id -> { seq_id -> msg_data } }
    # 병렬 워커들이 생성한 오디오 청크가 순서 없이 저장되는 캐시
    buffer_cache: Dict[int, Dict[int, str]] = field(default_factory=lambda: defaultdict(dict))

    # --- Worker Management ---
    llm_worker_threads: Dict[str, threading.Thread] = field(default_factory=dict)
    converter_threads: Dict[str, threading.Thread] = field(default_factory=dict)
    
    # --- Dynamic Prompt ---
    dynamic_prompt_16k: Optional[torch.Tensor] = None
    prompt_lock: threading.Lock = field(default_factory=threading.Lock) # 프롬프트 업데이트 시 충돌 방지
    
    # --- Converter Queue ---
    # LLM 워커가 (task_id, seq_id, audio_chunk)를 넣는 큐
    conversion_job_q: queue.Queue = field(default_factory=queue.Queue)


SESSIONS: Dict[str, Session] = {}
SESS_LOCK = threading.Lock()


# [수정] Converter (Task ID 인지)
def audio_converter_worker(sess: Session, worker_id: int):
    """
    conversion_job_q에서 작업을 받아 Base64 인코딩 후,
    (task_id, seq_id) 태그와 함께 buffer_cache에 저장합니다.
    """
    logging.info(f"[{sess.sid} | Converter-{worker_id}] Worker START.")
    
    while not sess.stop_event.is_set():
        try:
            job = sess.conversion_job_q.get(timeout=0.1) 
        except queue.Empty:
            continue

        if job is None: 
            logging.info(f"[{sess.sid} | Converter-{worker_id}] Shutting down.")
            break
        
        audio_chunk = job['audio_chunk']
        task_id = job['task_id']
        seq_id = job['seq_id']
        
        logging.info(f"[{sess.sid} | Converter-{worker_id}] Processing Task={task_id}, SEQ_ID={seq_id} START.")

        if audio_chunk.numel() > 0:
            try:
                buf = io.BytesIO()
                torchaudio.save(buf, audio_chunk, SAMPLE_RATE, format="wav")
                buf.seek(0)
                wav_chunk_bytes = buf.read()
                b64 = base64.b64encode(wav_chunk_bytes).decode("utf-8")
                
                msg_data = json.dumps({
                    "type": "streaming_audio", "b64wav": b64
                })
                
                # [핵심] task_id와 seq_id를 키로 사용하여 캐시에 저장
                with SESS_LOCK:
                    sess.buffer_cache[task_id][seq_id] = msg_data
                
            except Exception as e:
                logging.error(f"[{sess.sid} | Converter-{worker_id}] Conversion Error: {e}")

        sess.conversion_job_q.task_done()
        logging.info(f"[{sess.sid} | Converter-{worker_id}] Processing Task={task_id}, SEQ_ID={seq_id} END.")

    logging.info(f"[{sess.sid} | Converter-{worker_id}] Worker fully stopped.")


# [수정] LLM Worker (Task-based)
def llm_worker(sess: Session, worker_id: int):
    """
    task_q에서 "작업 덩어리"를 가져와 전체 TTS 파이프라인을 실행합니다.
    """
    logging.info(f"[{sess.sid} | LLM-Worker-{worker_id}] Worker START. Waiting for task...")

    while not sess.stop_event.is_set():
        
        # 1. Get a *full task* (e.g., "여기에 문장을 입력")
        try:
            task_data = sess.task_q.get(timeout=1.0) 
        except queue.Empty:
            continue 

        if task_data is None:
            continue
        
        task_id, text_for_this_run = task_data
        
        logging.info(f"[{sess.sid} | LLM-Worker-{worker_id}] **INFERENCE RUN START (TaskID={task_id}).** Text: '{text_for_this_run}'")
        
        # 이 작업(Task) 전용 로컬 변수
        accumulated_audio_chunks = []
        next_seq_id_for_this_task = 0

        # 2. 이 Task의 텍스트를 스트리밍 LLM에 공급할 text_generator 정의
        def text_generator() -> Generator[str, None, None]:
            # 전체 텍스트 덩어리를 공백 기준(혹은 다른 기준)으로 쪼개서 yield
            # (bistream LLM이 청크 단위 입력을 가정하므로)
            words = text_for_this_run.strip().split(' ')
            for i, word in enumerate(words):
                if not word: # 빈 문자열 스킵
                    continue
                
                chunk = word + ' ' # 공백을 다시 붙여서 전달 (마지막 제외)
                if i == len(words) - 1:
                    chunk = word # 마지막 단어
                
                logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] Yielding chunk: '{chunk}'")
                yield chunk
            
            logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] Text generator finished.")
            # 제너레이터가 종료되면, cosyvoice.inference_instruct2가 알아서 EOS 처리

        use_frontend = True 
        logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] Starting TTS inference loop. Using Text Frontend: {use_frontend}")
        
        # 3. 프롬프트 결정 (락 사용)
        with sess.prompt_lock:
            if sess.dynamic_prompt_16k is not None:
                current_prompt_16k = sess.dynamic_prompt_16k
                logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] Using DYNAMIC prompt (len: {current_prompt_16k.shape[1]/16000:.2f}s)")
            else:
                current_prompt_16k = prompt_speech_16k
                logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] Using STATIC prompt (default).")

        # 4. TTS 추론 실행
        try:
            logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] **Pipeline Execution START.**")
            
            # [!!! 핵심 수정 !!!]
            # `session_id=sess.sid`를 전달하여 model.py가 오디오 캐시를 유지하도록 함
            for out in cosyvoice.inference_instruct2(
                    tts_text=text_generator(), 
                    instruct_text="",
                    prompt_speech_16k=current_prompt_16k,
                    zero_shot_spk_id="",
                    stream=True,
                    speed=1.0,
                    text_frontend=use_frontend,
                    **{"session_id": sess.sid} # [핵심] 이 줄 추가
            ):
                if sess.stop_event.is_set(): break
                
                audio_chunk = out["tts_speech"].cpu()
                accumulated_audio_chunks.append(audio_chunk) 

                if audio_chunk.numel() > 0:
                    # [핵심] 이 Task 내부에서의 순서 ID
                    seq_id = next_seq_id_for_this_task
                    next_seq_id_for_this_task += 1 

                    logging.info(f"[{sess.sid} | Task={task_id}] => Handing SEQ_ID={seq_id} to Converter Queue.")
                    # [핵심] Task ID와 함께 컨버터 큐에 전송
                    sess.conversion_job_q.put({
                        'task_id': task_id,
                        'audio_chunk': audio_chunk,
                        'seq_id': seq_id,
                    })

        except Exception as e:
            logging.error(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] TTS (instruct2) Error: {e}", exc_info=True)
            # [핵심] 에러 로그도 Task ID와 함께 sse_q에 전송
            sess.sse_q.put((task_id, json.dumps({"type": "log", "msg": f"TTS error (Task {task_id}): {e}"})))
        
        logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] **Pipeline Execution END.**")

        # 5. 최종 오디오 처리 및 프롬프트 업데이트
        if accumulated_audio_chunks and text_for_this_run.strip(): 
            try:
                logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] **FINAL AUDIO PROCESSING START.**")
                
                final_audio = torch.cat(accumulated_audio_chunks, dim=1)
                final_text = text_for_this_run.strip()
                
                save_filename = f"session_audio_{sess.sid}_task{task_id}_{int(time.time())}.wav" 
                save_path = os.path.join(OUTPUT_DIR, save_filename)
                
                torchaudio.save(save_path, final_audio, SAMPLE_RATE, format="wav")
                logging.info(f"[{sess.sid} | Task={task_id}] ✅ Accumulated audio saved successfully to: {save_path}")

                file_url = f"/outputs/{save_filename}"
                logging.info(f"[{sess.sid} | Task={task_id}] SENDING 'final_audio' with URL DATA: {file_url}")
                
                # Dynamic prompt update (locked)
                with sess.prompt_lock:
                    max_prompt_frames = 16000 * 30 
                    final_audio_16k = torchaudio.transforms.Resample(
                        orig_freq=SAMPLE_RATE, new_freq=16000
                    )(final_audio.cpu())
                    
                    if final_audio_16k.shape[1] > max_prompt_frames:
                        logging.info(f"[{sess.sid} | Task={task_id}] Dynamic prompt exceeds 30s. Truncating to last 30s.")
                        final_audio_16k = final_audio_16k[:, -max_prompt_frames:]
                    sess.dynamic_prompt_16k = final_audio_16k
                    logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] ✅ Dynamic prompt (len: {final_audio_16k.shape[1]/16000:.2f}s) saved.")

                # [핵심] final_audio 메시지도 Task ID와 함께 sse_q에 전송
                sess.sse_q.put((task_id, json.dumps({
                    "type": "final_audio",
                    "url": file_url,
                    "text": final_text
                })))
                
                logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] **FINAL AUDIO PROCESSING END.**")

            except Exception as e:
                logging.error(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] ❌ Final Audio/Prompt Error: {e}")
            finally:
                accumulated_audio_chunks.clear()
        else:
            # 오디오가 생성되지 않았더라도, sse_audio가 다음 Task로 넘어갈 수 있도록 'end' 신호를 보냄
            logging.info(f"[{sess.sid} | LLM-Worker-{worker_id} | Task={task_id}] No audio generated. Sending 'end' signal.")
            sess.sse_q.put((task_id, json.dumps({"type": "end"})))
        
        logging.info(f"[{sess.sid} | LLM-Worker-{worker_id}] Inference RUN END (TaskID={task_id}). Waiting for next task...")


# [수정] 세션 생성 시, LLM/Converter 풀 시작
def get_or_create_session(sid: str) -> Session:
    with SESS_LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            sess = Session(sid=sid)
            SESSIONS[sid] = sess
            
            # 1. 병렬 LLM 워커 풀 시작
            for i in range(NUM_LLM_WORKERS):
                t_llm = threading.Thread(target=llm_worker, args=(sess, i + 1), daemon=True)
                t_llm.start()
                sess.llm_worker_threads[f'llm-worker-{i+1}'] = t_llm
            
            # 2. 병렬 오디오 컨버터 워커 풀 시작
            for i in range(NUM_CONVERTER_WORKERS):
                t_converter = threading.Thread(target=audio_converter_worker, args=(sess, i + 1), daemon=True)
                t_converter.start()
                sess.converter_threads[f'converter-{i+1}'] = t_converter
                
    return sess
# ==============================


# ==============================
# 4) HTTP Routes (순차 재생 보장 로직 적용)
# ==============================

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    logging.info(f"Serving file: {filename} from {OUTPUT_DIR}")
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

@app.route("/")
def index():
    return render_template("index.html") 

# [수정] /type (작업 덩어리 생성)
@app.route("/type", methods=["POST"])
def type_event():
    data = request.get_json(force=True)
    sid = data.get("sid") or str(uuid.uuid4())
    text = data.get("text", "")
    force = data.get("force", False) # 'force=True'가 '작업 덩어리'의 끝을 의미

    sess = get_or_create_session(sid)
    
    # [수정] last_sent_len 대신 text_buffer 사용
    current_text_chunk = text[len(sess.text_buffer):]
    
    if force:
        # [핵심] force=True일 때,
        # 1. 버퍼에 쌓인 텍스트 + 현재 텍스트를 합쳐 '하나의 작업 덩어리'를 만듦
        full_text_chunk = sess.text_buffer + current_text_chunk
        full_text_chunk = full_text_chunk.strip()
        
        if not full_text_chunk:
            logging.info(f"[{sid}] Force flush requested but buffer is empty. Ignoring.")
            sess.text_buffer = "" # 버퍼는 비워줌
            return jsonify({"ok": True})

        # 2. 이 덩어리에 Task ID 할당
        with SESS_LOCK: # next_task_id는 공유 자원이므로 락
            task_id = sess.next_task_id
            sess.next_task_id += 1
        
        # 3. (Task ID, 텍스트 덩어리)를 task_q에 넣음
        sess.task_q.put((task_id, full_text_chunk))
        logging.info(f"[{sid}] Queued new TASK (ID={task_id}): '{full_text_chunk}'")
        
        # 4. 다음 덩어리를 위해 텍스트 버퍼 초기화
        sess.text_buffer = ""
    
    else:
        # [핵심] force=False일 때는 큐에 넣지 않고, 텍스트 버퍼에 누적
        sess.text_buffer += current_text_chunk
        logging.info(f"[{sid}] Buffering text: '{sess.text_buffer}'")

    return jsonify({"ok": True})

# [수정] /sse_audio (절대 순서 보장)
@app.route("/sse_audio")
def sse_audio():
    sid = request.args.get("sid") or str(uuid.uuid4())
    sess = get_or_create_session(sid)
    
    # [핵심] SSE 스레드가 기대하는 Task ID와 Seq ID
    expected_task_id = 0
    expected_seq_id = 0

    # Task ID에 해당하는 final/end 메시지를 임시 저장하는 캐시
    final_message_cache: Dict[int, str] = {}

    def event_stream():
        nonlocal expected_task_id, expected_seq_id
        
        while not sess.stop_event.is_set():
            
            # 1. (최우선) 오디오 청크(streaming_audio) 전송
            #    현재 기대하는 Task ID의 캐시가 존재하는지 확인
            msg_data = None
            with SESS_LOCK:
                if expected_task_id in sess.buffer_cache:
                    # 현재 기대하는 Seq ID의 청크가 도착했는지 확인
                    if expected_seq_id in sess.buffer_cache[expected_task_id]:
                        # [순서 보장] 순서에 맞는 청크 발견!
                        msg_data = sess.buffer_cache[expected_task_id].pop(expected_seq_id)
                        expected_seq_id += 1 # 다음 순서 ID 증가
            
            if msg_data:
                # 청크를 발견하면 즉시 전송
                logging.info(f"[{sid}] SSE: Sending Task={expected_task_id}, SEQ_ID={expected_seq_id - 1}.")
                yield f"data: {msg_data}\n\n"
                time.sleep(0.005) # 브라우저 디코딩 시간
                continue # 다음 청크를 즉시 확인

            # 2. (차선) 최종 메시지 (final_audio, end, log) 처리
            #    (오디오 청크를 보낼 것이 없을 때만 실행됨)
            
            # sse_q에 쌓인 메시지들을 로컬 캐시로 이동
            try:
                while True:
                    task_id, msg_json = sess.sse_q.get_nowait()
                    msg_obj = json.loads(msg_json)
                    
                    if msg_obj["type"] == "final_audio" or msg_obj["type"] == "end":
                        # final 또는 end 메시지는 Task ID별로 캐시에 저장
                        logging.info(f"[{sid}] SSE: Caching FINAL message for Task={task_id}.")
                        final_message_cache[task_id] = msg_json
                    else:
                        # log 등 기타 메시지는 즉시 전송
                        logging.info(f"[{sid}] SSE: Sending (non-blocking log): {msg_json}")
                        yield f"data: {msg_json}\n\n"
                        
            except queue.Empty:
                pass # sse_q가 비었으면 다음 로직으로
            
            # 3. (핵심) Task 전환 로직
            #    현재 기대하는 Task ID의 final/end 메시지가 캐시에 있는지 확인
            if expected_task_id in final_message_cache:
                msg_json = final_message_cache.pop(expected_task_id)
                logging.info(f"[{sid}] SSE: Sending Task={expected_task_id} FINAL message.")
                yield f"data: {msg_json}\n\n"
                
                # [순서 보장] Task 전환!
                logging.info(f"[{sid}] SSE: Task {expected_task_id} FINISHED. Moving to Task {expected_task_id + 1}.")
                
                # 이전 Task의 오디오 캐시 정리 (메모리 누수 방지)
                with SESS_LOCK:
                    if expected_task_id in sess.buffer_cache:
                        sess.buffer_cache.pop(expected_task_id)
                        logging.info(f"[{sid}] SSE: Cleared buffer cache for Task={expected_task_id}.")

                expected_task_id += 1
                expected_seq_id = 0 # 다음 Task의 0번 청크부터 다시 시작
                continue # 즉시 다음 Task의 0번 청크 확인

            # 4. 보낼 것이 아무것도 없을 때 (Ping)
            time.sleep(0.05) 
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
