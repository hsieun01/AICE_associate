# 핵심 단계
- fit() -> predict() -> 성능평가 -> 피처 중요도
## 종류
1. 분류 모델: LogisticRegression / DecisionTreeClassifier / RandomForestClassifier / XGBClassifier
2. 회귀 모델: LinearRegression / DecisionTreeRegressor / RandomForestRegressor / XGBRegressor

### 1. LogisticRegressio - 이진분류
from sklearn.linear_model import LogisticRegression
# 모델 생성
lr = LogisticRegression(
C=1.0,            # 규제 강도 (작을수록 강한 규제)
max_iter=100,       # 최대 반복 횟수
random_state=42)

# 학습
lr.fit(X_train, y_train)

# 예측
y_pred = lr.predict(X_test)

### 2) DecisionTreeClassifier - 해석하기 좋음 (투명한 모델)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)

### 3. RandomForestClassifier - 배깅 트리 기반 분류
from sklearn.ensemble import RandomForestClassifier
# 모델 생성
rfc = RandomForestClassifier(
n_estimators=100,    # 트리 개수
max_depth=None,      # 트리 깊이
min_samples_split=2, # 분할 최소 샘플
random_state=42) 

# 학습
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

# 특성 중요도 (지원!)
importances = rfc.feature_importances_

### 4. XGBClassifier
from xgboost import XGBClassifier
# 모델 생성
xgb = XGBClassifier(
n_estimators=100,    # 트리 개수
max_depth=6,         # 트리 깊이
learning_rate=0.1,   # 학습률
random_state=42)

# 학습
xgb.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 특성 중요도 (지원!)
importances = xgb.feature_importances_

### 1. LinearRegression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
피처 중요도 없음

### 2. RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
importances = model.feature_importances_

### 3. DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
importances = model.feature_importances_

### 4. XGBRegressor
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
importances = model.feature_importances_


