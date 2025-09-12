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
from logging.handlers import QueueHandler
from tqdm import tqdm

# --- 설정 ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SENTENCES_PATH = "./demo/test/sentences.txt"
OUTPUT_DIR = "./demo/test/generated_audio_auto"
RESULTS_FILE = "./demo/test/experiment_results.txt"
NUM_RUNS = 1
# ------------

# 1. 서버 실행 및 로그 캡처를 위한 설정
# app_stream_tts에서 app 객체를 가져옵니다.
# 이를 위해 sys.path에 프로젝트 디렉토리를 추가합니다.
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PROJ_DIR))
from tester import app

log_queue = queue.Queue()

def run_server():
    """백그라운드 스레드에서 Flask 서버를 실행하고 로그를 큐로 보냅니다."""
    # 모든 로거의 핸들러를 QueueHandler로 교체
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO) # INFO 레벨 이상만 캡처

    # Werkzeug 로그는 비활성화
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    print("백그라운드에서 TTS 서버 시작...")
    app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True)

# 2. 메인 실험 로직
def main():
    # 서버 스레드 시작
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(5) # 서버가 완전히 시작될 때까지 대기

    # 문장 파일 읽기
    try:
        with open(SENTENCES_PATH, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"오류: {SENTENCES_PATH} 파일을 찾을 수 없습니다.")
        shutdown_server()
        return

    # 결과 파일 및 출력 폴더 준비
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f_results:
        f_results.write("실험 순서|입력 텍스트|오디오 경로|로그 정보\n")

    print(f"총 {len(sentences)}개의 문장으로 {NUM_RUNS}회 자동 실험을 시작합니다.")

    total_tasks = NUM_RUNS * len(sentences)
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        for run_num in range(1, NUM_RUNS + 1):
            run_folder = os.path.join(OUTPUT_DIR, f"run_{run_num}")
            os.makedirs(run_folder, exist_ok=True)

            for i, sentence in enumerate(sentences):
                session_id = str(uuid.uuid4())
                audio_chunks = []

                # 텍스트 전송
                requests.post(f"http://{SERVER_HOST}:{SERVER_PORT}/type", json={"sid": session_id, "text": sentence})

                # 오디오 수신
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

                # 오디오 파일 저장
                final_wav_data = b"".join(audio_chunks)
                safe_filename = "".join([c for c in sentence if c.isalnum() or c in (' ', '-')]).rstrip()[:20]
                audio_path = os.path.join(run_folder, f"sentence_{i+1}_{safe_filename}.wav")
                with open(audio_path, 'wb') as f_audio:
                    f_audio.write(final_wav_data)

                # 로그 큐에서 [FINAL-MAP] 정보 찾기
                log_info_block = ""
                start_capture = False
                while not log_queue.empty():
                    log_record = log_queue.get()
                    log_message = log_record.getMessage()

                    if "[FINAL-MAP" in log_message:
                        start_capture = True

                    if start_capture:
                        log_info_block += log_message + "\n"

                    if "[LLM-END]" in log_message and start_capture:
                        break

                # 결과 파일에 기록
                with open(RESULTS_FILE, 'a', encoding='utf-8') as f_results:
                    # 파이프(|) 문자를 다른 것으로 대체하여 CSV 형식 유지
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
        # 서버가 이미 종료되었을 수 있음
        pass

if __name__ == "__main__":
    main()