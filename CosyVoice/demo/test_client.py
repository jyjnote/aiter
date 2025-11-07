import requests
import time
import uuid

# --- 설정 ---
BASE_URL = "http://127.0.0.1:8000"  # app_stream_tts.py가 실행 중인 주소
TEXT_TO_SEND = "지금은 테스트 얼마나 걸리는지 확인해 봅니다."
# --- 설정 끝 ---

def run_test(delay_seconds: float):
    """
    주어진 시간 간격(delay_seconds)으로 텍스트를 나누어 서버에 전송합니다.
    """
    test_sid = str(uuid.uuid4())
    chunks = TEXT_TO_SEND.split(' ')
    
    print("\n" + "="*50)
    print(f"▶️  테스트 시작 (SID: {test_sid})")
    print(f"▶️  전송 간격: {delay_seconds}초")
    print(f"▶️  전체 텍스트: '{TEXT_TO_SEND}'")
    print("="*50)
    
    accumulated_text = ""
    
    for i, chunk in enumerate(chunks):
        # [수정] 서버의 'last_forced_text' 로직에 맞춰 항상 전체 텍스트를 보냅니다.
        # 마지막 청크가 아니면 공백을 추가합니다.
        if i < len(chunks) - 1:
            accumulated_text += chunk + " "
        else:
            accumulated_text += chunk

        is_final_chunk = (i == len(chunks) - 1)
        
        # force=true는 마지막 청크에서만 보냅니다.
        # non-stream 모드이므로, 중간 전송은 force=false로 하여 서버가 버퍼링하게 합니다.
        payload = {
            "sid": test_sid,
            "text": accumulated_text,
            "force": is_final_chunk  
        }
        
        try:
            response = requests.post(f"{BASE_URL}/type", json=payload)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            
            if is_final_chunk:
                print(f"✅ FINAL 전송: '{accumulated_text}' (force=True)")
                print("\n🔥 서버 로그에서 RTF 값을 확인하세요! 🔥")
            else:
                print(f"➡️  버퍼링: '{accumulated_text}' (force=False)")
                time.sleep(delay_seconds)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 전송 실패: {e}")
            print("❌ app_stream_tts.py 서버가 실행 중인지 확인하세요.")
            break

if __name__ == "__main__":
    # 1. 1초 간격으로 테스트
    run_test(delay_seconds=1.0)
    
    # 다음 테스트를 위해 잠시 대기
    time.sleep(5) 
    
    # # 2. 0.5초 간격으로 테스트
    # run_test(delay_seconds=0.5)
    
    # # 다음 테스트를 위해 잠시 대기
    # time.sleep(5)
    
    # # 3. 딜레이 없이 최대한 빠르게 테스트
    # run_test(delay_seconds=0)