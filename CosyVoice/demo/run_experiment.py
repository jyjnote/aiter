import requests
import time
import uuid
import os
import sys

# --- 설정 ---
# app_stream_tts.py가 실행 중인 서버 주소
SERVER_BASE_URL = 'http://127.0.0.1:8000'
SENTENCES_FILE = 'test/sentences.txt'  # 읽어올 텍스트 파일
# -----------

TYPE_URL = f'{SERVER_BASE_URL}/type'
MEASUREMENT_PAGE_URL = f'{SERVER_BASE_URL}/measure'

def run_experiment():
    # 1. 이 실험을 위한 고유 세션 ID 생성
    session_id = str(uuid.uuid4())
    
    print("="*60)
    print(f" 실험 세션 ID (Session ID): {session_id}")
    print("="*60)
    print("\n실험 시작 방법:")
    print("1. (필수) 브라우저에서 아래 URL을 열어주세요:")
    print(f"   {MEASUREMENT_PAGE_URL}?sid={session_id}")
    print("\n2. 브라우저 페이지에 '✅ SSE 연결 성공!' 로그가 뜨는지 확인하세요.")
    print("3. 확인이 되었으면, 이 터미널로 돌아와 [Enter] 키를 누르세요...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n실험 취소됨.")
        return

    # 4. sentences.txt 파일 읽기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, SENTENCES_FILE)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 공백이 아닌 줄만 필터링
        sentences = [line.strip() for line in lines if line.strip()]
        
        if not sentences:
            print(f"오류: '{file_path}' 파일이 비어있거나 읽을 수 없습니다.")
            return
            
        print(f"총 {len(sentences)}개의 문장을 찾았습니다. 전송을 시작합니다.")
        
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        print(f"이 스크립트({os.path.basename(__file__)})와 동일한 위치에 {SENTENCES_FILE}이 있는지 확인하세요.")
        return
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return

    # 5. 각 문장을 'force=true' (완결된 문장)로 서버에 전송
    for i, sentence in enumerate(sentences):
        payload = {
            "sid": session_id,
            "text": sentence,
            "force": True  # 핵심: 이 문장이 하나의 완성된 오디오임을 서버에 알림
        }
        
        try:
            print(f"전송 중 ({i+1}/{len(sentences)}): \"{sentence}\"")
            response = requests.post(TYPE_URL, json=payload, timeout=10)
            response.raise_for_status() # 200-299 범위가 아니면 오류 발생
            
            # 서버가 요청을 처리할 최소한의 시간
            # 실제 오디오 생성 및 재생은 서버와 브라우저가 비동기로 처리합니다.
            time.sleep(0.1) 
            
        except requests.exceptions.ConnectionError:
            print("\n오류: 서버에 연결할 수 없습니다.")
            print(f"{SERVER_BASE_URL} 에서 app_stream_tts.py가 실행 중인지 확인하세요.")
            break
        except requests.exceptions.RequestException as e:
            print(f"\n요청 중 오류 발생: {e}")
            break
        except KeyboardInterrupt:
            print("\n사용자에 의해 전송이 중단되었습니다.")
            break
    
    print("="*60)
    print("모든 문장 전송 완료.")
    print("브라우저의 측정 페이지에서 결과를 확인하세요.")
    print("="*60)

if __name__ == "__main__":
    run_experiment()