#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  CUDA_VISIBLE_DEVICES=4 python -u demo/app_stream_tts.py
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

import torch
import torchaudio
from flask import Flask, request, Response, render_template, jsonify

# ==============================
# 0) App & Model Init
# ==============================
# 현재 스크립트 파일의 절대 경로를 가져옴
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트의 루트 디렉토리 경로를 설정 (현재 디렉토리의 부모)
PROJ_DIR = os.path.dirname(THIS_DIR)

# 파이썬이 모듈을 찾을 수 있도록 프로젝트 경로를 추가
sys.path.append(PROJ_DIR)
sys.path.append(os.path.join(PROJ_DIR, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

# 기본 로깅 핸들러를 제거하여 Flask의 로거와 충돌 방지
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# 새로운 로깅 설정을 구성
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨을 INFO로 설정하여 너무 상세한 로그는 제외
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

app = Flask(__name__)

# 모델과 제로샷 음성 프롬프트의 경로 설정
MODEL_DIR = os.path.join(PROJ_DIR, "pretrained_models", "CosyVoice-KSS-Finetuned")
PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt1.wav")

# 설정된 경로에 파일이 실제로 존재하는지 확인
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"{MODEL_DIR} does not exist!")
if not os.path.isfile(PROMPT_WAV):
    raise FileNotFoundError(f"{PROMPT_WAV} not found!")

# 서버 시작 시 모델을 메모리에 로드
print("서버 시작: CosyVoice2 로드…")
cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=False)
prompt_speech_16k = load_wav(PROMPT_WAV, 16000)
print("모델 로드 완료.")

# 모델의 샘플 레이트를 변수에 저장
SAMPLE_RATE = cosyvoice.sample_rate

def is_korean(char: str) -> bool:
    """입력된 문자가 한글인지 확인하는 유틸리티 함수"""
    if not char: return False
    return '\uac00' <= char <= '\ud7a3'

# ==============================
# 1) Session Management
# ==============================
@dataclass
class Session:
    """각 사용자의 연결 상태와 데이터를 관리하는 세션 클래스"""
    sid: str  # 고유한 세션 ID
    last_sent_len: int = 0  # 이전에 처리한 텍스트의 길이
    sse_q: queue.Queue = field(default_factory=queue.Queue)  # 생성된 오디오를 클라이언트로 보내는 큐
    text_stream_q: queue.Queue = field(default_factory=queue.Queue)  # 클라이언트로부터 받은 텍스트를 저장하는 큐
    worker_thread: Optional[threading.Thread] = None  # TTS 작업을 처리하는 백그라운드 스레드
    stop_event: threading.Event = field(default_factory=threading.Event)  # 스레드를 안전하게 종료하기 위한 이벤트
    restart_tts: threading.Event = field(default_factory=threading.Event)  # TTS 추론을 재시작하기 위한 이벤트

# 모든 활성 세션을 저장하는 딕셔너리
SESSIONS: Dict[str, Session] = {}
# 여러 스레드가 동시에 SESSIONS 딕셔너리에 접근하는 것을 방지하기 위한 Lock
SESS_LOCK = threading.Lock()

def get_or_create_session(sid: str) -> Session:
    """세션 ID를 기반으로 기존 세션을 가져오거나, 없으면 새로 생성하는 함수"""
    with SESS_LOCK:  # Lock을 사용하여 스레드 안전성 확보
        sess = SESSIONS.get(sid)
        if sess is None:  # 해당 세션 ID가 딕셔너리에 없으면
            sess = Session(sid=sid)  # 새로운 세션 객체 생성
            SESSIONS[sid] = sess  # 딕셔너리에 새 세션 추가
            # 각 세션마다 별도의 TTS 작업자 스레드를 생성하여 독립적으로 운영
            t = threading.Thread(target=tts_worker, args=(sess,), daemon=True)
            t.start()  # 스레드 시작
            sess.worker_thread = t
    return sess

# # ==============================
# # 3) TTS Worker & Synthesizer - [핵심 수정]
# # ==============================
# def tts_worker(sess: Session):
#     """백그라운드에서 TTS 변환 작업을 수행하는 함수"""
#     print(f"[{sess.sid}] TTS worker started.")

#     # stop_event가 설정될 때까지 외부 루프를 계속 실행하여 TTS 재시작을 가능하게 함
#     while not sess.stop_event.is_set():
#         # 루프가 새로 시작될 때마다 재시작 이벤트를 초기화
#         sess.restart_tts.clear()

#         def text_generator() -> Generator[str, None, None]:
#             """세션의 text_stream_q에서 텍스트를 받아 어절 단위로 잘라주는 제너레이터"""
#             buffer = ""  # 클라이언트로부터 들어오는 텍스트 조각을 임시 저장하는 버퍼
#             while not sess.stop_event.is_set():
#                 # 재시작 신호(force=true)가 오면 현재 추론을 중단하고 새 추론을 준비
#                 if sess.restart_tts.is_set():
#                     # 재시작 전, 버퍼에 남아있는 텍스트가 있다면 모두 모델로 보내서 처리
#                     if buffer:
#                         logging.info(f"[{sess.sid}] Restart triggered. Flushing buffer: '{buffer}'")
#                         yield buffer
#                         buffer = ""
#                     logging.info(f"[{sess.sid}] Restart signal received. Terminating text generator.")
#                     break
                
#                 try:
#                     # 텍스트 큐에서 새로운 텍스트 조각을 가져옴 (0.1초 타임아웃)
#                     chunk = sess.text_stream_q.get(timeout=0.1)
#                     buffer += chunk
                    
#                     # 버퍼에서 마지막 띄어쓰기 위치를 찾음
#                     last_space_index = buffer.rfind(' ')
                    
#                     if last_space_index != -1:
#                         # 띄어쓰기를 찾았다면, 그 지점까지의 텍스트(완성된 어절)를 모델로 전달
#                         to_yield = buffer[:last_space_index + 1]
#                         # 전달한 부분은 버퍼에서 제거하고, 나머지는 다음 조각과 합치기 위해 남겨둠
#                         buffer = buffer[last_space_index + 1:]
                        
#                         logging.info(f"[{sess.sid}] Yielding by space: '{to_yield.strip()}' | Remaining in buffer: '{buffer.strip()}'")
#                         yield to_yield

#                 except queue.Empty:
#                     # 큐가 비어있으면 루프를 계속 돌며 새 텍스트를 기다림
#                     continue

#             # 외부 루프가 종료될 때, 버퍼에 남아있는 최종 텍스트를 모두 처리
#             if buffer:
#                 logging.info(f"[{sess.sid}] Final flush of buffer: '{buffer}'")
#                 yield buffer
            
#             logging.info(f"[{sess.sid}] Text generator finished.")

#         try:
#             logging.info(f"[{sess.sid}] Starting new TTS inference loop.")
#             # text_generator로부터 어절 단위 텍스트를 받아 실시간으로 음성 합성
#             for out in cosyvoice.inference_instruct2(
#                     tts_text=text_generator(),
#                     instruct_text="",
#                     prompt_speech_16k=prompt_speech_16k,
#                     zero_shot_spk_id="",
#                     stream=True,
#                     speed=1.0,
#                     text_frontend=True,
#             ):
#                 audio_chunk = out["tts_speech"].cpu()  # 생성된 오디오 조각을 CPU로 이동
#                 if audio_chunk.numel() > 0:  # 오디오 데이터가 비어있지 않다면
#                     buf = io.BytesIO()  # 오디오 데이터를 저장할 인메모리 바이너리 버퍼
#                     # 오디오 조각을 WAV 형식으로 버퍼에 저장
#                     torchaudio.save(buf, audio_chunk, SAMPLE_RATE, format="wav")
#                     buf.seek(0)  # 버퍼의 포인터를 처음으로 이동
#                     wav_chunk_bytes = buf.read()  # 버퍼의 모든 바이트를 읽음
#                     # 클라이언트(웹)에서 사용하기 위해 바이트를 base64 문자열로 인코딩
#                     b64 = base64.b64encode(wav_chunk_bytes).decode("utf-8")
#                     # SSE 큐에 JSON 형식으로 오디오 데이터 추가
#                     sess.sse_q.put(json.dumps({"type": "audio", "b64wav": b64}))

#         except Exception as e:
#             logging.error(f"[{sess.sid}] TTS (instruct2) Error: {e}", exc_info=True)
#             sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))

#     # 스레드가 완전히 종료되기 전에 클라이언트에 'end' 메시지를 보냄
#     sess.sse_q.put(json.dumps({"type": "end"}))
#     print(f"[{sess.sid}] TTS worker fully stopped.")
# ==============================
# 3) TTS Worker & Synthesizer - [핵심 수정]
# ==============================
def tts_worker(sess: Session):
    """백그라운드에서 TTS 변환 작업을 수행하는 함수"""
    print(f"[{sess.sid}] TTS worker started.")

    # stop_event가 설정될 때까지 외부 루프를 계속 실행하여 TTS 재시작을 가능하게 함
    while not sess.stop_event.is_set():
        # 루프가 새로 시작될 때마다 재시작 이벤트를 초기화
        sess.restart_tts.clear()

        # [수정됨] 텍스트를 받자마자 즉시 모델로 전달하는 제너레이터
        def text_generator() -> Generator[str, None, None]:
            """세션의 text_stream_q에서 텍스트를 받자마자 즉시 전달하는 제너레이터"""
            while not sess.stop_event.is_set():
                # 재시작 신호(force=true)가 오면 현재 추론을 중단
                if sess.restart_tts.is_set():
                    logging.info(f"[{sess.sid}] Restart signal received. Terminating text generator.")
                    break
                
                try:
                    # 텍스트 큐에서 새로운 텍스트 조각을 가져옴 (0.1초 타임아웃)
                    chunk = sess.text_stream_q.get(timeout=0.1)
                    
                    # [핵심 수정] 버퍼링이나 띄어쓰기 확인 없이 받은 즉시 모델로 전달!
                    if chunk:
                        logging.info(f"[{sess.sid}] Yielding immediately: '{chunk.strip()}'")
                        yield chunk

                except queue.Empty:
                    # 큐가 비어있으면 루프를 계속 돌며 새 텍스트를 기다림
                    continue
            
            logging.info(f"[{sess.sid}] Text generator finished.")

        try:
            logging.info(f"[{sess.sid}] Starting new TTS inference loop.")
            # text_generator로부터 텍스트를 받아 실시간으로 음성 합성
            for out in cosyvoice.inference_instruct2(
                    tts_text=text_generator(),
                    instruct_text="",
                    prompt_speech_16k=prompt_speech_16k,
                    zero_shot_spk_id="",
                    stream=True,
                    speed=1.0,
                    text_frontend=True,
            ):
                audio_chunk = out["tts_speech"].cpu()  # 생성된 오디오 조각을 CPU로 이동
                if audio_chunk.numel() > 0:  # 오디오 데이터가 비어있지 않다면
                    buf = io.BytesIO()  # 오디오 데이터를 저장할 인메모리 바이너리 버퍼
                    # 오디오 조각을 WAV 형식으로 버퍼에 저장
                    torchaudio.save(buf, audio_chunk, SAMPLE_RATE, format="wav")
                    buf.seek(0)  # 버퍼의 포인터를 처음으로 이동
                    wav_chunk_bytes = buf.read()  # 버퍼의 모든 바이트를 읽음
                    # 클라이언트(웹)에서 사용하기 위해 바이트를 base64 문자열로 인코딩
                    b64 = base64.b64encode(wav_chunk_bytes).decode("utf-8")
                    # SSE 큐에 JSON 형식으로 오디오 데이터 추가
                    sess.sse_q.put(json.dumps({"type": "audio", "b64wav": b64}))

        except Exception as e:
            logging.error(f"[{sess.sid}] TTS (instruct2) Error: {e}", exc_info=True)
            sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))

    # 스레드가 완전히 종료되기 전에 클라이언트에 'end' 메시지를 보냄
    sess.sse_q.put(json.dumps({"type": "end"}))
    print(f"[{sess.sid}] TTS worker fully stopped.")

# ==============================
# 4) HTTP Routes
# ==============================
@app.route("/")
def index():
    """웹사이트의 메인 페이지를 렌더링"""
    return render_template("index.html")

@app.route("/type", methods=["POST"])
def type_event():
    """클라이언트로부터 타이핑 이벤트를 받아 처리하는 엔드포인트"""
    data = request.get_json(force=True)  # 요청 본문에서 JSON 데이터를 추출
    sid = data.get("sid") or str(uuid.uuid4())  # 세션 ID를 가져오거나 없으면 새로 생성
    text = data.get("text", "")  # 전체 텍스트를 가져옴
    force = data.get("force", False)  # 즉시 재시작 여부를 나타내는 'force' 플래그

    sess = get_or_create_session(sid)
    
    # 새로 입력된 텍스트가 있는지 확인
    if len(text) > sess.last_sent_len:
        # 이전에 처리한 길이 이후의 새로운 텍스트 조각만 추출
        new_text_chunk = text[sess.last_sent_len:]
        # 새 텍스트 조각을 TTS 작업자 스레드의 텍스트 큐에 추가
        sess.text_stream_q.put(new_text_chunk)
        logging.info(f"[{sid}] Queued new text chunk: '{new_text_chunk.strip()}'")
        # 마지막으로 처리한 텍스트 길이를 현재 텍스트 길이로 업데이트
        sess.last_sent_len = len(text)
    
    # 'force' 플래그가 true이면(구두점 입력 시), TTS 재시작 이벤트를 설정
    if force:
        logging.info(f"[{sid}] Force flush requested. Setting restart event.")
        sess.restart_tts.set()
    
    return jsonify({"ok": True})

@app.route("/sse_audio")
def sse_audio():
    """생성된 오디오를 클라이언트로 스트리밍하는 Server-Sent Events(SSE) 엔드포인트"""
    sid = request.args.get("sid") or str(uuid.uuid4())
    sess = get_or_create_session(sid)

    def event_stream():
        """SSE 메시지를 생성하는 제너레이터 함수"""
        while not sess.stop_event.is_set():
            try:
                # SSE 큐에서 메시지를 가져옴 (0.5초 타임아웃)
                msg = sess.sse_q.get(timeout=0.5)
                # SSE 데이터 형식에 맞춰 클라이언트로 전송
                yield f"data: {msg}\n\n"
                # 'end' 메시지를 받으면 스트림 종료
                if '"type":"end"' in msg:
                    break
            except queue.Empty:
                # 큐가 비어있으면 'ping' 메시지를 보내 연결 유지
                yield 'data: {"type":"ping"}\n\n'

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # Nginx 같은 프록시 서버의 버퍼링 비활성화
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
    }
    return Response(event_stream(), headers=headers)

# ==============================
# 5) Run
# ==============================
if __name__ == "__main__":
    # Flask 웹 서버 실행
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)