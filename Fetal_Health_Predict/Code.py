import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rc("font", family="NanumGothic")

#  데이터 파일을 읽고 데이터프레임 변수명 df에 할당
df = pd.read_csv('fetal_health.csv')
print(df.shape)
print(df.describe())
print(df['fetal_health'].value_counts())

#태아 건강 상태(fetal_health)에 따라 기본 심박수(base_hr)의 분포를 비교
import seaborn as sns
sns.boxplot(df, x='fetal_health', y='base_hr')

#  태아 건강 상태(fetal_health)별로 평균
df.groupby('fetal_health').mean()

# 태아 건강 상태(fetal_health)의 분포
sns.countplot(df, x='fetal_health')

# 범주형 변수의 값을 수치형으로 변환
df['fetal_health'] = df['fetal_health'].replace({'Normal':0, 'Suspect':1, 'Pathological':2})
df_enc = df
# ! df_enc = df.copy()
# df_enc['fetal_health'] = df_enc['fetal_health'].replace({'Normal':0, 'Suspect':1, 'Pathological':2}) 와 같이 카피를 하고 하는게 더 안정적!                

# 불필요한 컬럼 삭제
df_del= df_enc.drop(columns = ['decel_l', 'decel_s', 'hist_zero'], axis=1) 

#수치형 변수들의 상관관계 확인
cols = ['accel', 'decel_p', 'var_st_abn', 'per_var_lt_abn', 'var_lt_mean', 
        'hist_mode', 'hist_mean', 'hist_median', 'hist_var', 'fetal_health']

sns.heatmap(df[cols].corr(), annot=True, fmt='.2f')

#결측치 처리
df_del.isnull().sum()
df_na = df_del.fillna(df_del.mean())
df_na.isnull().sum()

#  훈련데이터와 검증데이터를 분리
#  !분할 할 때 피처, 타겟 순서 주의, 오타주의!
from sklearn.model_selection import train_test_split
target = df_na['fetal_health']
features = df_na.drop('fetal_health', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(
features, target, test_size=0.25, random_state=100, stratify=target)

# 훈련데이터셋과 검증데이터셋에 스케일링
# fit은 train에서만 함.
from sklearn.preprocessing import RobustScaler 
rs = RobustScaler()

X_train_scaled = rs.fit_transform(X_train)
X_valid_scaled = rs.transform(X_valid)

# 태아 건강 상태를 예측하는 머신러닝 모델생
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier

lr = LogisticRegression(max_iter = 1000, random_state=120)
xgbrf = XGBRFClassifier(max_depth=5, min_child_weight=3, random_state=120)

lr.fit(X_train_scaled, y_train)
xgbrf.fit(X_train_scaled, y_train)

# 모델의 성능 평가
# 헷갈리지 말자 예측값을 구하기 위해 필요한건 입력값이다.
# 분류 리포트는 모델 함수가 아니기 때문에 '모델.repert' 이런식으로 쓰지 않음
# 성능평가 중 classificatio_report 또는 metrix_confusion 로 평가하면 분류 모델임
from sklearn.metrics import classification_report
y_pred_lr = lr.predict(X_valid_scaled)
y_pred_xgbrf = xgbrf.predict(X_valid_scaled)

print(classification_report(y_pred_lr, y_valid))
print(classification_report(y_pred_xgbrf, y_valid))

# 중요도
feature_importance = xgbrf.feature_importances_
feature_name = features.columns

df_importance = pd.DataFrame({'importance': feature_importance,
                              'features': feature_name})
                              
df_importance.sort_values(by='importance', ascending=False, inplace=True)
display(df_importance)

# 딥러닝 라이브러리
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
# print(tf.config.list_physical_devices('CPU')) 

y_train = to_categorical(y_train, num_classes=3)
y_valid = to_categorical(y_valid, num_classes=3)

# 딥러닝 모델
model = Sequential([
    Dense(32, activation='selu', input_shape=(features.shape[1],)),
    Dense(64, activation='selu'),
    Dropout(0.3),
    Dense(16, activation='selu'),
    BatchNormalization(),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
mc = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(X_train_scaled, y_train, epochs=80, batch_size=128, validation_data=(X_valid_scaled, y_valid), callbacks=[es, mc])
요.
# 성능평가
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()
plt.title('Model Categorical Crossentropy')
plt.xlabel('Epochs')
plt.ylabel('Loss')

simul = np.array([0.13333333,0.8,0.5,0.91490668,0,0,0,0.58939436,-0.3,0,0.31746032,-0.44444444,0.50943396,-0.09090909,0.36842105,0.31578947,0.26315789,-0.04597701]).reshape(1, -1)

# 시뮬레이션
best_dl = load_model('best_model.h5')
print(best_dl.predict(simul))












