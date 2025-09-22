import os
import sys
import requests
import sseclient
import json
import base64
import uuid
import time
import logging
import threading
import queue
import re  # 정규식 모듈 임포트
from logging.handlers import QueueHandler
from tqdm import tqdm

# --- 설정 ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SENTENCES_PATH = "./demo/test/sentences.txt"
OUTPUT_DIR = "./demo/test/generated_audio_auto_non_bistream" # 비교를 위해 폴더명 변경
RESULTS_FILE = "./demo/test/experiment_results_non_bistream.txt" # 비교를 위해 파일명 변경
NUM_RUNS = 1
# ------------

# 서버 실행 설정
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PROJ_DIR))
from tester import app # non-bistream용 서버(tester.py) 사용

log_queue = queue.Queue()

# --- ✨ 1. 파일/폴더명으로 사용하기 안전한 텍스트로 변환하는 함수 추가 ---
def sanitize_filename(text: str, max_length: int = 60) -> str:
    """문자열을 안전한 파일/폴더명으로 변환합니다."""
    # 한글, 영어, 숫자, 공백, 하이픈(-)만 남기고 나머지 특수문자 제거
    sanitized = re.sub(r'[^A-Za-z0-9가-힣\s-]', '', text).strip()
    # 공백을 언더스코어(_)로 변경
    sanitized = sanitized.replace(' ', '_')
    # 길이를 제한
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized if sanitized else "empty_sentence"


def run_server():
    """백그라운드 스레드에서 Flask 서버를 실행하고 로그를 큐로 보냅니다."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print("백그라운드에서 Non-Bistream TTS 서버 시작...")
    app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True)

# 메인 실험 로직
def main():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(5)

    try:
        with open(SENTENCES_PATH, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"오류: {SENTENCES_PATH} 파일을 찾을 수 없습니다.")
        shutdown_server()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f_results:
        f_results.write("실험 순서|입력 텍스트|오디오 경로|로그 정보\n")

    print(f"총 {len(sentences)}개의 문장으로 {NUM_RUNS}회 Non-Bistream 자동 실험을 시작합니다.")

    total_tasks = NUM_RUNS * len(sentences)
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        for run_num in range(1, NUM_RUNS + 1):
            run_folder = os.path.join(OUTPUT_DIR, f"run_{run_num}")
            os.makedirs(run_folder, exist_ok=True)

            for i, sentence in enumerate(sentences):
                session_id = str(uuid.uuid4())
                audio_chunks = []

                requests.post(f"http://{SERVER_HOST}:{SERVER_PORT}/type", json={"sid": session_id, "text": sentence})

                sse_url = f"http://{SERVER_HOST}:{SERVER_PORT}/sse_audio?sid={session_id}"
                response = requests.get(sse_url, stream=True)
                client = sseclient.SSEClient(response)

                for event in client.events():
                    if not event.data: continue
                    data = json.loads(event.data)
                    if data.get("type") == "audio":
                        audio_chunks.append(base64.b64decode(data['b64wav']))
                    elif data.get("type") == "end":
                        break

                final_wav_data = b"".join(audio_chunks)

                # --- ✨ 2. 문장 텍스트를 기반으로 파일명 생성 ---
                safe_name = sanitize_filename(sentence)
                filename_with_index = f"{i+1:03d}_{safe_name}.wav"
                audio_path = os.path.join(run_folder, filename_with_index)

                with open(audio_path, 'wb') as f_audio:
                    f_audio.write(final_wav_data)

                log_info_block = ""
                start_capture = False
                while not log_queue.empty():
                    log_record = log_queue.get()
                    log_message = log_record.getMessage()
                    if "[FINAL-MAP" in log_message: start_capture = True
                    if start_capture: log_info_block += log_message + "\n"
                    if "[LLM-END]" in log_message and start_capture: break

                with open(RESULTS_FILE, 'a', encoding='utf-8') as f_results:
                    log_info_cleaned = log_info_block.strip().replace('\n', ' // ')
                    result_line = f"{run_num}-{i+1}|{sentence}|{audio_path}|{log_info_cleaned}\n"
                    f_results.write(result_line)

                pbar.update(1)

    print("\n실험 완료. 서버를 종료합니다.")
    shutdown_server()
    print(f"✅ 모든 결과가 '{RESULTS_FILE}' 파일과 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")


def shutdown_server():
    try:
        requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/shutdown")
    except requests.exceptions.ConnectionError:
        pass

if __name__ == "__main__":
    main()