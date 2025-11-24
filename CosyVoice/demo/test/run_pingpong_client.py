import requests
import uuid
import time
import os
import re

# --- 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 텍스트 파일 경로 확인 필수!
FILE_PATH = os.path.join(BASE_DIR, "sentences.txt") 
SERVER_URL = "http://localhost:8000/synthesize_chunk"
PUNCTUATION_REGEX = re.compile(r'[.!?]$')

def send_pingpong_request(session_id, text_chunk, worker_id, batch_id):
    payload = {
        "sid": session_id,
        "text_chunk": text_chunk,
        "worker_id": worker_id,
        "is_final": True,
        "batch_id": batch_id
    }
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 실패: {e}")

def run_pingpong_experiment():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 파일 없음: {FILE_PATH}")
        return

    session_id = str(uuid.uuid4())
    print(f"🏓 실험 시작 (SID: {session_id})")

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        words = re.sub(r'\s+', ' ', f.read()).strip().split(' ')

    current_worker = 'A'
    batch_id = 1
    buffer = ""

    for word in words:
        buffer += (" " + word if buffer else word)
        # 핑퐁은 문장 단위이므로 구두점 나올 때까지 대기
        if PUNCTUATION_REGEX.search(word):
            print(f"🚀 [Batch {batch_id}] 전송 -> Worker {current_worker}: {buffer[:20]}...")
            send_pingpong_request(session_id, buffer, current_worker, batch_id)
            
            current_worker = 'B' if current_worker == 'A' else 'A'
            batch_id += 1
            buffer = ""
            time.sleep(0.5) # 다음 문장 입력 딜레이

    if buffer:
        send_pingpong_request(session_id, buffer, current_worker, batch_id)

    print("✅ 실험 종료. 서버의 CSV 파일을 확인하세요.")

if __name__ == "__main__":
    run_pingpong_experiment()