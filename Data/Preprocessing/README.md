### step2-1. 결측치 처리
- .isnull() : 결측치 발견
- .dropna() : 결측치가 포함된 행 삭제(axis=로 방향 지정)
- .fillna() :  수치형(mean or median) / 범주형(mode)[0]
- inplace=True: 원본에 덮어쓰기

### step2-2. 인코딩 
1) .get_dummies() : 원핫 인코딩
- columns=[] : 변환할 칼럼 지정
- drop_first=True : 더비 변수 함정 회피(다중공선성 방지)
- dtype=int : 모델 입력 오류 예방
- df_dncoded : 이 변수에 결과 분리 저장
- 시험에서 가장 빈번하게 출제

2) LabelEncoder() : 라벨 인코딩
- from sklearn.preprocessing import LabelEncoder
- le = LabelEncoder()
- for col in [칼럼1, 칼럼2]:
    df[col] = le.fit_transform(df[col]): : 학습과 변환을 동시에 수행함.
- 여러 칼럼을 적용할 때는 for루프 사용
- 범주를 0,1,2.. 순서대로 정수로 변환

3) replace() + astype() : 매핑 인코딩
- df_01 = df['sex'].replace({'male': 1, 'female': 0}).astype(int)
- mapping = {'a':0, 'b':1} : 딕셔너리로 매핑 관계 정의
  for col in ['col1', 'col2']:
    df[col] = df[col].replace(mapping).astype(int) : A/B, Yes/Np, T/F와 같은 이진 변수에 적합

### step3 데이터 분할
- from sklearn.model_selection import train_test_split : 인포트하기
- test_size=0.2 : 학습/테스트 데이터 분할
- random_state=42 : 결과 재현성을 위해 고정값
- stratify=y : 정답 클래스 비율 유지하며 분할(분류모델에서 필수)
