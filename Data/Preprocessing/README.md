# 핵심 단계 
- 결측치 처리 -> 이상치 처리 -> 인코딩 -> 데이터 분할 -> 스케일링
## 1. 결측치 처리 
### 1) dropna()
- 결측치 소수일 때
- 분석에 큰 영향이 없는 행일 때
### 2) fillna()
- df.clean = df.copy로 원본 데이터 보존해서 오류 방지
- 수치형: .mean() or .median
- 범주형: .made()[0]
- 
## 2. 이상치 처리
### 1) 조건 기준 제거 (시험point!)
- df = df[df["컬럼명"] < 기준값]
### 2) IOR 기반 제거
### 3) clip 값 제한

## 3. 인코딩 
### 1) 원-핫 인코딩
- 범주 간 순서가 없을 때
- pd.get_dummies(df, columns=["컬럼명"])
### 2) 라벨 인코딩
- from sklearn.preprocessing import LabelEncoder
- le = LabelEncoder()
- df["칼럼명"] = le.fit_transform(df["칼럼명"])
- 시험 point!) 라벨 인코딩 대상 칼럼을 지정하여 for문 활용, fit하기 전에 데이터 copy해서 따로 지정.
### 3) replace() (시험 point!)
- 딕셔너리로 정의
### 4) astype() 
- .astype(int)와 같이 직접 형 변환

## 3. 데이터 분할
### 1) 타겟(y)과 입력(X) 분리
- X = df_enc.drop("타겟", axis=1)
- y = df_enc["타겟"]
### 2) 훈련과 검증 데이터 분할
- from sklearn.model_selection import train_test_split
- X_train, X_valid, y_train, y_valid = train_test_split()
- test_size=0.2으로 검증용 20% 분할
- random_state=42으로 분할 방식 고정해서 재현성 유지
- stratify=y : 정답 클래스 비율 유지하며 분할(분류모델에서 필수)

## 4. 스케일링
### 1) StandardScaler
- from sklearn.preprocessing import StandardScaler
- 평균 0, 표준편차1
- 조건이 없을 때
### 2) MinMaxScaler
- from sklearn.preprocessing import MinMaxScaler
- 최소0, 최대1
- 값 범위를 0~1로 변환
### 3) RobustScaler
- 중앙값/IOQ 기준
- 이상치가 많을 때














