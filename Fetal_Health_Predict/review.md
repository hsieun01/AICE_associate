# 태아 건강 상태 예측
## 틀린거 복습
### 10번. 데이터 분할
- target = df_na['fetal_health']
- features = df_na.drop('fetal_health', axis=1)
- X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.25, random_state=100, stratify=target)
  - 분할 할 때, (피처, 타겟, 사이즈...) 순서 주의
 
### 11번. RobustScaler 스케일링
- rs = RobustScaler()
- X_train_scaled = rs.fit_transform(X_train)
- X_valid_scaled = rs.transform(X_valid)
  - fit은 train에서 한 번만 해야함.

### 13번. 모델 성능 평가
- y_pred_lr = lr.predict(X_valid_scaled)
- y_pred_xgbrf = xgbrf.predict(X_valid_scaled)
  - 예측 값을 구하기 위해 필요한 것은 테스트 입력값 하나임.
- print(classification_report(y_valid, y_pred_lr))
- print(classification_report(y_valid, y_pred_xgbrf))
  - 분류 리포트는 모델 함수가 아니기 때문에 '모델.repert' 이런식으로 쓰지 않음.
  - 리포트(실제값, 예측값) 순서 주의
 
### 15번. 딥러닝 모델 생성
- Bulid단계(모델 구조): model = Sequential([~~])
  - from tensorflow.keras.models import Sequential
  - from tensorflow.keras.layers import Dense
    - 임포트(아마 시험에서는 줄듯)
  - Dense(수, 활성화함수 지정 activation='')
  - Dropout()은 과적합 방지
    - 학습 중 뉴런을 무작위로 끔
  -  BatchNormalization()은 학습 안정화
    - 각 층 입력을 정규화
- Complie 단계(학습 설정): model.coplie(~~)
  - optimizer=' '
    - 손실함수를 최적화하기 위한 알고리즘 지정
  - loss=' '
    - 손실함수 지정
  - metrics=[' ']
    - 각 에포크마다 검증 데이터 셋 지정
- 조기 종료 설정: EarlyStopping(~~)
  - es = 
  - monitor= 'val_loss'
    - val_loss를 보고 
  - patience= 5
    - 5번 이상 반복했는데 성능이 향상되지 않으면 종료시킴
  - restore_best_weights=True
    - 가장 낮은 검증 손실을 낸 모델로 복구
- 모델 저장 설정: ModelCheckpoint(~~)
  - mc = 
  - save_best_only=True
- 학습 실행 단계: history = model.fit(~~)
  - x_train, y_train
  - 하이파라미터
  - validation_data = (x_test, y_test)
  - callbacks = [es,mc]
    
