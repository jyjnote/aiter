import pandas as pd
import glob
import os
import sys

def analyze_timeline():
    # 1. 가장 최근 CSV 파일 찾기
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(base_dir, 'delay_measurements_*.csv'))
    
    if not csv_files:
        print("❌ CSV 파일이 없습니다.")
        return
    
    # 최신 파일 선택
    csv_path = max(csv_files, key=os.path.getctime)
    print(f"📂 분석 대상 파일: {os.path.basename(csv_path)}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return

    # 2. Batch ID 기준으로 정렬 (이게 핵심!)
    # 작업이 늦게 끝나서 로그가 뒤에 찍혀도, 여기서 순서를 바로잡습니다.
    df = df.sort_values(by='BatchID').reset_index(drop=True)
    
    print(f"\n--- 📊 총 {len(df)}개 배치 분석 (정렬됨) ---")

    # 3. 타임라인 시뮬레이션
    # "이전 오디오가 언제 끝나는지"를 추적합니다.
    
    previous_audio_end_time = 0.0
    
    print(f"{'ID':<4} {'Worker':<6} {'RTF':<6} {'오디오길이':<10} {'생성완료시각':<15} {'재생가능시각':<15} {'GAP(딜레이)':<10}")
    print("-" * 80)

    delays = []
    rtfs = []

    for idx, row in df.iterrows():
        batch_id = int(row['BatchID'])
        worker = row['Worker']
        gen_end_time = row['Gen_End_Time(Epoch)'] # 생성이 끝난 절대 시각
        audio_len = row['Audio_Duration(Sec)']
        rtf = row['RTF']
        
        # [시뮬레이션 로직]
        # 재생 가능 시각 = max(내 생성이 끝난 시간, 앞 오디오가 끝난 시간)
        
        if idx == 0:
            # 첫 번째 배치는 "생성 끝나자마자" 재생
            play_start_time = gen_end_time
            gap = 0.0 # 첫 배치는 비교 대상 없음
        else:
            # 이전 오디오가 아직 재생 중이면 기다렸다가(Gap=0) 이어서 재생
            # 이전 오디오가 이미 끝났으면(Gap>0) 바로 재생
            
            if gen_end_time > previous_audio_end_time:
                # 버퍼링 발생! (생성이 늦음)
                play_start_time = gen_end_time
                gap = gen_end_time - previous_audio_end_time
            else:
                # 미리 준비됨 (끊김 없음)
                play_start_time = previous_audio_end_time
                gap = 0.0 # 사용자 입장에서는 끊김 없음 (음수 딜레이는 0으로 처리)
                # 참고: 실제 여유 시간(Margin)을 보고 싶으면 음수 값 그대로 사용 가능
                # margin = gen_end_time - previous_audio_end_time (음수면 여유)

        # 다음 배치를 위해 "이 오디오가 끝나는 시간" 갱신
        previous_audio_end_time = play_start_time + audio_len
        
        delays.append(gap)
        rtfs.append(rtf)

        # 출력 (생성완료 시간은 보기 좋게 소수점 뒤만 표시)
        t_gen = f"{gen_end_time % 1000:.2f}"
        t_play = f"{play_start_time % 1000:.2f}"
        
        print(f"{batch_id:<4} {worker:<6} {rtf:.2f}   {audio_len:.2f}s      {t_gen:<15} {t_play:<15} {gap:.4f}s")

    # 4. 최종 통계
    avg_gap = sum(delays[1:]) / len(delays[1:]) if len(delays) > 1 else 0 # 첫 배치는 제외
    avg_rtf = sum(rtfs) / len(rtfs)
    max_gap = max(delays)
    
    print("-" * 80)
    print(f"✅ 평균 딜레이(Gap): {avg_gap:.4f} 초 (첫 문장 제외)")
    print(f"✅ 최대 딜레이(Gap): {max_gap:.4f} 초")
    print(f"✅ 평균 RTF:       {avg_rtf:.4f}")
    print("-" * 80)

if __name__ == "__main__":
    analyze_timeline()