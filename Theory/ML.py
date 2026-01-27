# 핵심 단계
- fit() -> predict() -> 성능평가 -> 피처 중요도
## 종류
1. 분류 모델: LogisticRegression / DecisionTreeClassifier / RandomForestClassifier / XGBClassifier
2. 회귀 모델: LinearRegression / DecisionTreeRegressor / RandomForestRegressor / XGBRegressor

### 1. LogisticRegressio - 이진분류
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0,max_iter=100,random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

### 2) DecisionTreeClassifier - 해석하기 좋음 (투명한 모델)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
importances = dt.feature_importances_    # 특성 중요도 (지원!)

### 3. RandomForestClassifier - 배깅 트리 기반 분류
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(
n_estimators=100, max_depth=None, min_samples_split=2, random_state=42) 
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
importances = rfc.feature_importances_    # 특성 중요도 (지원!)

### 4. XGBClassifier
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred = model.predict(X_test)
importances = xgb.feature_importances_     # 특성 중요도 (지원!)

### 1. LinearRegression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

### 2. RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
importances = model.feature_importances_     # 특성 중요도 (지원!)

### 3. DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
importances = model.feature_importances_    # 특성 중요도 (지원!)

### 4. XGBRegressor
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
importances = model.feature_importances_    # 특성 중요도 (지원!)


