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
import torch
import torchaudio
import re  # 정규식 모듈 임포트
from logging.handlers import QueueHandler
from tqdm import tqdm

# --- 설정 ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SENTENCES_PATH = "./demo/test/sentences.txt"

OUTPUT_DIR = "./demo/test/generated_audio_auto_bistream_chunks"
RESULTS_FILE = "./demo/test/experiment_results_bistream.txt"
NUM_RUNS = 1
# ------------

# 서버 실행 및 로그 캡처 설정
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PROJ_DIR))
from tester_bistream import app

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


def concatenate_wav_files(chunk_paths, output_path):
    """디스크에 저장된 WAV 청크 파일들을 읽어 하나로 합칩니다."""
    audio_tensors = []
    sample_rate = -1

    if not chunk_paths:
        logging.warning("병합할 청크 파일이 없습니다.")
        return

    try:
        chunk_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    except (ValueError, IndexError):
        logging.warning("청크 파일명 정렬에 실패하여 기본 정렬을 사용합니다.")
        chunk_paths.sort()

    for path in chunk_paths:
        try:
            waveform, sr = torchaudio.load(path)
            if sample_rate == -1:
                sample_rate = sr
            elif sample_rate != sr:
                logging.warning(f"샘플링 레이트 불일치: {sr} vs {sample_rate}. {path} 건너뜁니다.")
                continue
            audio_tensors.append(waveform)
        except Exception as e:
            logging.error(f"청크 파일 로드 실패 {path}: {e}")
            continue

    if not audio_tensors:
        logging.error("병합할 유효한 오디오 텐서가 없습니다.")
        return

    try:
        combined_waveform = torch.cat(audio_tensors, dim=1)
        torchaudio.save(output_path, combined_waveform, sample_rate)
        logging.info(f"{len(audio_tensors)}개의 청크를 성공적으로 병합하여 '{output_path}'에 저장했습니다.")
    except Exception as e:
        logging.error(f"최종 오디오 파일 저장 실패: {e}")


def run_server():
    """백그라운드 스레드에서 Flask 서버를 실행하고 로그를 큐로 보냅니다."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print("백그라운드에서 Bistream TTS 서버 시작...")
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
        f_results.write("실험 순서|입력 텍스트|병합된 오디오 경로|로그 정보\n")

    print(f"총 {len(sentences)}개의 문장으로 {NUM_RUNS}회 Bistream 자동 실험을 시작합니다.")
    print(f" - 개별 청크와 최종 병합본은 '{OUTPUT_DIR}'에 저장됩니다.")

    total_tasks = NUM_RUNS * len(sentences)
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        for run_num in range(1, NUM_RUNS + 1):
            run_folder = os.path.join(OUTPUT_DIR, f"run_{run_num}")
            os.makedirs(run_folder, exist_ok=True)

            for i, sentence in enumerate(sentences):
                session_id = str(uuid.uuid4())
                saved_chunk_paths = []

                # --- ✨ 2. 문장 텍스트를 기반으로 폴더명 생성 ---
                safe_folder_name = sanitize_filename(sentence)
                # 폴더 이름에 순번을 붙여 정렬 및 중복 방지 (001, 002, ...)
                folder_name_with_index = f"{i+1:03d}_{safe_folder_name}"
                sentence_chunk_folder = os.path.join(run_folder, folder_name_with_index)
                os.makedirs(sentence_chunk_folder, exist_ok=True)

                requests.post(f"http://{SERVER_HOST}:{SERVER_PORT}/type", json={"sid": session_id, "text": sentence})

                sse_url = f"http://{SERVER_HOST}:{SERVER_PORT}/sse_audio?sid={session_id}"
                response = requests.get(sse_url, stream=True)
                client = sseclient.SSEClient(response)

                for chunk_idx, event in enumerate(client.events()):
                    if not event.data: continue
                    data = json.loads(event.data)
                    
                    if data.get("type") == "audio":
                        decoded_chunk = base64.b64decode(data['b64wav'])
                        chunk_path = os.path.join(sentence_chunk_folder, f"chunk_{chunk_idx}.wav")
                        with open(chunk_path, 'wb') as f_chunk:
                            f_chunk.write(decoded_chunk)
                        saved_chunk_paths.append(chunk_path)

                    elif data.get("type") == "end":
                        break
                
                combined_audio_path = "N/A"
                if saved_chunk_paths:
                    combined_audio_path = os.path.join(sentence_chunk_folder, "combined.wav")
                    concatenate_wav_files(saved_chunk_paths, combined_audio_path)
                else:
                    logging.warning(f"'{sentence}' 문장에 대해 생성된 오디오 청크가 없습니다.")

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
                    result_line = f"{run_num}-{i+1}|{sentence}|{combined_audio_path}|{log_info_cleaned}\n"
                    f_results.write(result_line)

                pbar.update(1)

    print("\n실험 완료. 서버를 종료합니다.")
    shutdown_server()
    print(f"✅ 모든 결과가 '{OUTPUT_DIR}' 폴더와 '{RESULTS_FILE}' 파일에 저장되었습니다.")

def shutdown_server():
    try:
        requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/shutdown")
    except requests.exceptions.ConnectionError:
        pass

if __name__ == "__main__":
    main()