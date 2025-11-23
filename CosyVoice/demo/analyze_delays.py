import pandas as pd
import os
import sys

# --- 1. 설정 (Configuration) ---
# 현재 스크립트가 있는 디렉토리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 서버가 생성한 CSV 파일 경로
CSV_PATH = os.path.join(BASE_DIR, 'delay_measurements.csv')

def analyze_delay_stats():
    print(f"'{CSV_PATH}' 파일에서 데이터를 읽는 중...")
    
    # 1-1. CSV 파일 존재 여부 확인
    if not os.path.exists(CSV_PATH):
        print(f"❌ 에러: '{CSV_PATH}' 파일을 찾을 수 없습니다.")
        print("먼저 'app_measure_delay.py' 서버를 실행하고, 클라이언트 스크립트를 실행하여 데이터를 생성하세요.")
        return

    # 1-2. CSV 파일 읽기 (인코딩 시도)
    try:
        try:
            df = pd.read_csv(CSV_PATH, encoding='utf-8-sig') # 한글 헤더용
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_PATH, encoding='utf-8') # 영어 헤더용
            
    except pd.errors.EmptyDataError:
        print(f"❌ 에러: '{CSV_PATH}' 파일이 비어있습니다.")
        return
    except Exception as e:
        print(f"❌ 에러: CSV 파일 읽기 실패: {e}")
        return

    # 1-3. 딜레이 컬럼명 확인 (한글/영어 모두 지원)
    if '재생 딜레이(초)' in df.columns:
        delay_column_name = '재생 딜레이(초)'
    elif 'playback_gap_seconds' in df.columns:
        delay_column_name = 'playback_gap_seconds'
    else:
        print(f"❌ 에러: '재생 딜레이(초)' 또는 'playback_gap_seconds' 열을 찾을 수 없습니다.")
        print(f"   감지된 열: {df.columns.tolist()}")
        return
        
    if len(df) < 2:
        print("⚠️ 경고: 데이터가 2개 미만이라 '워밍업' 배치를 제외한 통계를 낼 수 없습니다.")
        return

    # --- 3. 통계 계산 ---
    # [중요] 첫 번째 측정값(index 0)은 캐시 워밍업이므로 제외
    data_to_analyze = df.iloc[1:].copy()
    
    # 딜레이(Gap) 데이터만 선택
    delay_data = data_to_analyze[delay_column_name]
    
    # 4. 통계치 계산
    stats = delay_data.describe()
    
    count = int(stats['count'])
    mean = stats['mean']
    std_dev = stats['std']
    min_val = stats['min']
    median = stats['50%'] # 중앙값 (Median)
    max_val = stats['max']
    
    # 딜레이가 0보다 큰 (묵음 발생) 배치의 비율
    positive_delay_ratio = (delay_data > 0).mean() * 100

    # --- 5. 결과 출력 ---
    print("\n--- 📈 스트리밍 딜레이 통계 (첫 배치 제외) ---")
    print(f"  총 측정 배치 수: {count} 개")
    print("---------------------------------------------")
    print(f"  평균 딜레이 (Mean):   {mean:.4f} 초")
    print(f"  중앙값 딜레이 (Median): {median:.4f} 초")
    print(f"  표준 편차 (Std Dev):  {std_dev:.4f} 초 (낮을수록 안정적)")
    print("---------------------------------------------")
    print(f"  최소 딜레이 (Min):    {min_val:.4f} 초 (가장 빠른 응답, 마이너스는 '미리 준비됨')")
    print(f"  최대 딜레이 (Max):    {max_val:.4f} 초 (가장 긴 묵음 발생)")
    print("---------------------------------------------")
    print(f"  '묵음' 발생 비율 (딜레이 > 0초): {positive_delay_ratio:.2f} %")
    print("---")

if __name__ == "__main__":
    # --- 라이브러리 설치 확인 ---
    try:
        import pandas
    except ImportError:
        print("---")
        print("⚠️ 'pandas' 라이브러리가 필요합니다.")
        print("터미널에서 'pip install pandas'를 실행해주세요.")
        print("---")
        sys.exit()
        
    analyze_delay_stats()