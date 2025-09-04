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
# render_template과 render_template_string을 render_template으로 변경합니다.
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
PROMPT_WAV = os.path.join(PROJ_DIR, "asset", "zero_shot_prompt.wav")

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
# 1) 세션 관리 (변경 없음)
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

# 사용자 스레드 활당 부분임
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
# 2) 문장 추출 & 큐잉 (변경 없음)
# ==============================
def enqueue_flushable_sentences(sess: Session, force: bool = False):
    new_segment = sess.text[sess.last_flush_idx:]
    if not new_segment:
        return

    consumed = 0
    sentences: List[str] = []

    for m in re.finditer(r"[^\.!\?…。\！？]*[\.!\?…。\！？]", new_segment):
        end = m.end()
        chunk = new_segment[:end].strip() # 구두점 해당 부분까지 잘라서 청크로 저장하고
        if chunk:
            sentences.append(chunk) # 이부분에서 문장 리스트에 담아줌
        new_segment = new_segment[end:]
        consumed += end

    sess.last_flush_idx += consumed # 어디까지 소비했는지 그냥 체크하는 용도

    if force: # force 강제로 현재 텍스트를 만들어줘야할 경우가 있음
    # 공백 구두점, 사용자의 타임 아웃 이렇게 3가지 경우가 있음 이땐 바로 푸시해서 음성을 합성시키기 위함
    # 167 line에 코드 나와있음.
        rest = new_segment.strip()
        if rest:
            sentences.append(rest)
            sess.last_flush_idx = len(sess.text)

    for s in sentences: # 디버깅라인
        logging.debug(f"[{sess.sid}] enqueue sentence: {s[:80]}{'...' if len(s)>80 else ''}")
        sess.tts_q.put(s)
        sess.last_flush_ts = time.time()

# ==============================
# 3) TTS 워커 (변경 없음)
# ==============================
# tts_worker 메서드에서 이 메서드를 실행함
def synth_sentence_to_wav_bytes(sentence: str) -> bytes:
    # 이 함수는 변경할 필요 없음
    # 합성 코드 라인
    wav_parts = [] # 청크 단위로 만들어진 오디오를 붙여서 가지고 있음, 이를 사용
    # 이 cosy 메서드를 불러와서 사용함.
    # tts.model 메서드는 한/영 잘 나오는데 중간중간 bgm같은게 끼여져있음
    # cosyvoice.inference_zero_shot은 한국어가 중국어 처럼 나옴
    # cross 랭귀지가 bgm 문제가 젤 적은거 같음
    # 음악에 해당하는 토큰ID를 한번 검사, 그리고 특정 단어에 대해 뒤에 음성에 튀어나오냐?
    for out in cosyvoice.inference_zero_shot_typing( 
            text_stream=[sentence],
            prompt_text="<|endofprompt|>",
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
    return buf.read() # 서버가 읽을 수 있게 리턴해주기
    # for out in cosyvoice.inference_cross_lingual(
    #         tts_text=sentence,
    #         prompt_speech_16k=prompt_speech_16k,
    #         zero_shot_spk_id="",
    #         stream=False,
    #         speed=1.0,
    #         text_frontend=True
    #     ):
    #     wav_parts.append(out["tts_speech"].cpu())

    # if not wav_parts:
    #     return b""

    # wav_cat = torch.cat(wav_parts, dim=1)
    # buf = io.BytesIO()
    # torchaudio.save(buf, wav_cat, SAMPLE_RATE, format="wav")
    # buf.seek(0)
    # return buf.read() # 서버가 읽을 수 있게 리턴해주기


def tts_worker(sess: Session):
    last_keepalive = time.time()
    
    # 로그 추가: 워커 시작을 알림
    print(f"[{sess.sid}] TTS worker started.")

    while not sess.stop_event.is_set():
        now = time.time()

        # --- 혼합 전략 ---
        # 영어같은경우 she i/s my mam 일경우 i,s가 분단됨 이를 방지하고자 매우 작은 세컨드로 끝글자가 공백,구두점인지 판별 -> A
        # 사용자가 마지막 입력이 멈췄을 경우 타이핑이 전부 끝났다고 판단 -> B
        if (now - sess.last_flush_ts) >= FLUSH_INTERVAL_SEC and sess.last_flush_idx < len(sess.text):
            
            # 로그 추가: 플러시 조건 확인 시작
            print(f"[{sess.sid}] DEBUG: Checking flush conditions... (Full text: '{sess.text}')")
            
            last_char = sess.text[-1] if sess.text else ""
            
            if last_char.isspace() or PUNCT.match(last_char):
                # 로그 추가: 조건 A (공백/구두점) 충족
                print(f"[{sess.sid}] DEBUG: Condition A MET: Flushing due to space/punct ('{last_char}').")
                enqueue_flushable_sentences(sess, force=True) # 강제합성
            elif (now - sess.last_input_ts) >= WORD_TIMEOUT_SEC:
                # 로그 추가: 조건 B (타임아웃) 충족
                print(f"[{sess.sid}] DEBUG: Condition B MET: Flushing due to typing timeout ({WORD_TIMEOUT_SEC}s).")
                enqueue_flushable_sentences(sess, force=True) # 강제합성
            else:
                # 로그 추가: 대기 상태
                print(f"[{sess.sid}] DEBUG: WAITING: Last char ('{last_char}') is not space/punct, and timeout not met.")


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
    
    # 로그 추가: 워커 종료를 알림
    print(f"[{sess.sid}] TTS worker stopped.")
# ==============================
# 4) HTTP Routes
# ==============================
@app.route("/")
def index():
    # HTML 문자열 대신 render_template 함수를 사용하여 파일을 렌더링합니다.
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