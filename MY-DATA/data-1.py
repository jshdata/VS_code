import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data.csv')

# 2. 데이터 처리
# 필요한 데이터 전처리 작업을 수행합니다.
# 여기서는 간단하게 날짜별 데이터의 합계를 계산한다고 가정합니다.
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
monthly_data = df.resample('M').sum()  # 월별 합계 계산

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 월별 데이터 시각화
plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['value'], marker='o')  # bar를 plot으로 변경하고 marker 추가
plt.title('월별 데이터 합계')
plt.xlabel('날짜')
plt.ylabel('합계 값')
plt.xticks(rotation=45)
plt.grid(True)  # 그리드 추가로 가독성 향상
plt.tight_layout()
plt.show()