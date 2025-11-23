import requests
import uuid
import time
import os
import re

# --- 1. 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "sentences.txt")
SERVER_URL = "http://localhost:8000/type"

# 구두점(.!?)으로 끝나는지 확인하는 정규식
PUNCTUATION_REGEX = re.compile(r'[.!?]$')

# --- 2. 헬퍼 함수 ---
def send_request(session_id, full_text_state, force=False):
    """서버로 HTTP POST 요청을 전송하는 함수"""
    payload = {
        "sid": session_id,
        "text": full_text_state,
        "force": force
    }
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 실패: {e}")
        print("🤔 'app_measure_delay.py' 서버가 실행 중인지 확인하세요.")
        raise # 에러 발생 시 중단

# --- 3. 스크립트 실행 ---
def run_real_stream_experiment():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 에러: 파일을 찾을 수 없습니다. 경로: {FILE_PATH}")
        return

    session_id = str(uuid.uuid4())
    print(f"🎉 'index.html 모방' 리얼 스트리밍 클라이언트 시작. 세션 ID (sid): {session_id}")

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # 1. 모든 줄바꿈과 연속 공백을 "단일 공백"으로 치환
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # 2. 텍스트를 "단어" 또는 "단어+구두점" 기준으로 분리
        # (숱 많은 머리칼을.) -> ['숱', '많은', '머리칼을.']
        words = full_text.split(' ')
        words = [w for w in words if w] # 빈 문자열 제거
            
        if not words:
            print("❌ 에러: 'sentences.txt' 파일에서 단어를 분리할 수 없습니다.")
            return
            
        print(f"\n--- 📄 총 {len(words)}개의 단어(또는 구두점)를 순차 전송 ---")

    except Exception as e:
        print(f"❌ 에러: 파일 읽기 또는 토큰 분리 실패: {e}")
        return

    current_text_state = "" # 서버에 보낼 누적 텍스트
    
    try:
        for i, word in enumerate(words):
            
            # [핵심] 텍스트 누적
            # (index.html에서 타이핑하듯이 단어를 추가)
            if current_text_state:
                current_text_state += " " + word
            else:
                current_text_state = word
            
            print(f"\n🚀 [단어 {i+1}/{len(words)}] 전송: \"{word}\"")
            
            # --- [핵심 로직] ---
            if PUNCTUATION_REGEX.search(word):
                # [A] 단어가 구두점으로 끝나는 경우 (예: "머리칼을.")
                # 'force=True' 신호를 보내 "최종 배치" (keep_context=False)를 요청
                print(f"▶ 구두점 감지. 'force=True' (최종 배치) 신호 전송.")
                send_request(session_id, current_text_state, force=True)
                
                # [수정] 텍스트 상태를 초기화하지 않습니다!
                # 서버가 last_sent_len을 갱신했으므로, 다음 단어는
                # if len(text) > sess.last_sent_len: 조건을 통과합니다.
                
                print("... (다음 문장 대기 0.5초) ...")
                time.sleep(0.5)
            
            else:
                # [B] 단어가 구두점으로 끝나지 않는 경우 (예: "숱", "많은")
                # [수정] index.html 처럼 단어 뒤에 "단일 공백"을 추가합니다.
                current_text_state += " "
                print(f"▶ 단일 공백 추가. 'force=False' 전송.")
                send_request(session_id, current_text_state, force=False)
                
                # 실제 타이핑처럼 각 단어 전송 사이에 약간의 딜레이를 줍니다.
                time.sleep(0.1) 

        # 모든 단어 전송 후, 마지막이 구두점이 아니었다면 강제 종료
        if not PUNCTUATION_REGEX.search(words[-1]):
             print("\n🚀 [!! 최종] 남은 텍스트 강제 종료 신호 전송...")
             send_request(session_id, current_text_state, force=True)


        print("\n---")
        print("✨ 모든 요청을 보냈습니다. 서버 콘솔에서 'delay_measurements.csv' 파일을 확인하세요.")
        print("---")

    except Exception as e:
        print(f"❌ 스크립트 중단: {e}")

if __name__ == "__main__":
    run_real_stream_experiment()