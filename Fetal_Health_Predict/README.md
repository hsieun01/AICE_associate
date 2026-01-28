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
- print(classification_report(y_pred_lr, y_valid))
- print(classification_report(y_pred_xgbrf, y_valid))
  - 분류 리포트는 모델 함수가 아니기 때문에 '모델.repert' 이런식으로 쓰지 않음.
  - 리포트(예측값, 실제값) 순서 주의
 
### 15번. 
