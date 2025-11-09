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
# [수정] 1. 동적 임계값 변수 선언
SPACE_THRESHOLD_FIRST = 1     # 첫 청크는 공백 1개 (빠른 반응, 딜레이 감수)
SPACE_THRESHOLD_SUBSEQUENT = 4  # 이후 청크는 공백 4개 (더 큰 청크로 버퍼링 시간 확보)

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
# vLLM 및 FP16을 사용하려면 아래 주석을 해제하세요.
# cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=True, load_vllm=True)
cosyvoice = CosyVoice2(model_dir=MODEL_DIR, fp16=False) # 기본 로드
prompt_speech_16k = load_wav(PROMPT_WAV, 16000)
print("모델 로드 완료.")

SAMPLE_RATE = cosyvoice.sample_rate

# ==============================
# 1) Session Management
# ==============================
# [수정] 2. Session Dataclass에 상태 변수 추가
@dataclass
class Session:
    sid: str
    last_sent_len: int = 0
    sse_q: queue.Queue = field(default_factory=queue.Queue)
    text_stream_q: queue.Queue = field(default_factory=queue.Queue)
    worker_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    accumulated_audio: List[torch.Tensor] = field(default_factory=list)
    accumulated_text: str = ""  # 누적된 텍스트
    space_count: int = 0  # 공백 카운트
    is_first_chunk: bool = True  # 첫 번째 청크인지 추적
    current_threshold: int = field(default=SPACE_THRESHOLD_FIRST) # 현재 적용할 임계값

SESSIONS: Dict[str, Session] = {}
SESS_LOCK = threading.Lock()

def get_or_create_session(sid: str) -> Session:
    with SESS_LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            sess = Session(sid=sid)
            SESSIONS[sid] = sess
            t = threading.Thread(target=tts_worker_v2, args=(sess,), daemon=True)
            t.start()
            sess.worker_thread = t
    return sess

# ==============================
# 3) TTS Worker - Stream=False Version
# ==============================
# [수정] 3. tts_worker_v2 로직 수정
def tts_worker_v2(sess: Session):
    """
    동적 임계값을 사용하여 텍스트를 누적한 후 stream=False로 한 번에 합성
    """
    print(f"[{sess.sid}] TTS worker v2 started (stream=False mode, dynamic threshold)")

    while not sess.stop_event.is_set():
        try:
            chunk = sess.text_stream_q.get(timeout=1.0)
        except queue.Empty:
            continue

        if chunk is None:
            # EOS marker - 누적된 텍스트 모두 합성
            if sess.accumulated_text.strip():
                synthesize_accumulated_text(sess)
            sess.sse_q.put(json.dumps({"type": "end"}))
            # 세션 리셋 시, 상태 변수도 모두 초기화
            sess.accumulated_text = ""
            sess.space_count = 0
            sess.is_first_chunk = True
            sess.current_threshold = SPACE_THRESHOLD_FIRST # 초기 임계값으로 복원
            sess.last_sent_len = 0
            continue
        
        # 텍스트 누적
        sess.accumulated_text += chunk
        sess.space_count += chunk.count(' ')
        
        # 로그에 현재 임계값 표시
        logging.info(f"[{sess.sid}] Accumulated: '{sess.accumulated_text.strip()}' (spaces: {sess.space_count} / threshold: {sess.current_threshold})")
        
        # 동적 THRESHOLD 도달 시 합성
        if sess.space_count >= sess.current_threshold:
            # 마지막 공백까지만 합성
            last_space_idx = sess.accumulated_text.rfind(' ')
            if last_space_idx != -1:
                text_to_synthesize = sess.accumulated_text[:last_space_idx + 1]  # 공백 포함
                remaining = sess.accumulated_text[last_space_idx + 1:]
                
                # 합성 수행
                synthesize_text_chunk(sess, text_to_synthesize)
                
                # 리셋
                sess.accumulated_text = remaining
                sess.space_count = remaining.count(' ')

                # [핵심 로직]
                # 첫 번째 청크를 성공적으로 보냈다면,
                # 다음 임계값을 '더 크게' 변경하여 버퍼링 전략을 활성화합니다.
                if sess.is_first_chunk:
                    sess.is_first_chunk = False
                    sess.current_threshold = SPACE_THRESHOLD_SUBSEQUENT
                    logging.info(f"[{sess.sid}] First chunk sent. New threshold set to: {sess.current_threshold}")


def synthesize_text_chunk(sess: Session, text: str):
    """
    주어진 텍스트를 stream=False로 합성하고 결과 전송
    """
    if not text.strip():
        return
    
    # 띄어쓰기로 끝나도록 보장
    if not text.endswith(' '):
        text += ' '
    
    logging.info(f"[{sess.sid}] Synthesizing chunk (stream=False): '{text.strip()}'")
    
    try:
        # stream=False로 한 번에 합성
        result = None
        for out in cosyvoice.inference_instruct2(
                tts_text=text,
                instruct_text="",
                prompt_speech_16k=prompt_speech_16k,
                zero_shot_spk_id="",
                stream=False,  # 핵심: stream=False
                speed=1.0,
                text_frontend=True
        ):
            result = out
            break  # stream=False이므로 한 번만 실행됨
        
        if result:
            audio_chunk = result["tts_speech"].cpu()
            
            # 오디오 저장 및 전송
            buf = io.BytesIO()
            torchaudio.save(buf, audio_chunk, SAMPLE_RATE, format="wav")
            buf.seek(0)
            wav_chunk_bytes = buf.read()
            b64 = base64.b64encode(wav_chunk_bytes).decode("utf-8")
            
            # 청크 전송
            sess.sse_q.put(json.dumps({
                "type": "streaming_audio",  # 클라이언트와 호환되도록 변경
                "b64wav": b64,
                "text": text.strip()
            }))
            
            logging.info(f"[{sess.sid}] Audio chunk sent for: '{text.strip()}'")
            
    except Exception as e:
        logging.error(f"[{sess.sid}] TTS Error: {e}", exc_info=True)
        sess.sse_q.put(json.dumps({"type": "log", "msg": f"TTS error: {e}"}))

def synthesize_accumulated_text(sess: Session):
    """
    세션에 누적된 모든 텍스트를 합성 (EOS 마커 수신 시 호출)
    """
    if not sess.accumulated_text.strip():
        return
    
    text = sess.accumulated_text
    if not text.endswith(' '):
        text += ' '
    
    synthesize_text_chunk(sess, text)

# ==============================
# 4) HTTP Routes
# ==============================

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
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
    force = data.get("force", False)

    sess = get_or_create_session(sid)
    
    logging.info(f"[{sid}] /type received. len(text)={len(text)}, sess.last_sent_len={sess.last_sent_len}, force={force}")
    
    if len(text) > sess.last_sent_len:
        new_text_chunk = text[sess.last_sent_len:]
        sess.text_stream_q.put(new_text_chunk)
        logging.info(f"[{sid}] Queued new text chunk: '{new_text_chunk.strip()}'")
        sess.last_sent_len = len(text)
    
    if force:
        logging.info(f"[{sid}] Force flush requested. Queueing EOS marker (None).")
        sess.text_stream_q.put(None)
        # 리셋은 워커 스레드가 담당

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