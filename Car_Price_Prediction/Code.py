import padas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rc("font", family="NanumGothicCoding")

# 파일 읽기
df = pd.read_csv('mini_car_prices.csv')
print(df.describe())
print(df['color_group'].value_counts())
print(df['interior'].value_counts())

# 주행 거리(odometer)와 거래 가격(sellingprice)에 대한 관계 시각화
import seaborn as sns
sns.scatterplot(df, x='odometer', y='sellingprice')

# 차체 유형(body_type) 별 거래 가격(sellingprice)의 사분위 분포 시각화
sns.boxplot(df, x='body_type', y='sellingprice'])

# 자동차 브랜드 원산지(make)와 차체 유형(body_type)별로 차량 거래 가격(sellingprice)의 평균
df.groupby('make', 'body_type')['sellingprice'].mean()

# 차량 거래 가격(sellingprice)과 MMR(mmr)의 관계
sns.lmplot(df, x='sellingprice', y='mmr')

# 불필요하거나 정보를 누설하는 컬럼 삭제
df_del = df.drop(cloumns=['trim', 'vin', 'seller', 'mmr', 'saledate'], axis=1)

# 수치형 변수들의 상관관계 시각화
sns.heatmap(df_del.corr(), annot=True, fmt='.4f')

# 결측치 처리
df_na = df_del.copy()
print(df_na.isnull().sum())
df_na = df_na.fillna(df_na.mean(numeric_only=True))
df_na = df_na.fillna(df_na.mode()iloc[0])
print(df_na.isnull().sum())

# 레이블 인코딩(Label encoding)과 원-핫 인코딩(One-hot encoding) 범주형 변수를 수치형 변수로 변환
from sklearn.preprocessing import LabelEncoder
df_enc = df_na.copy()
le =  LabelEncoder()
df_enc['model'] = le.fit_transform(df_enc['model'])
df_enc['interior'] = le.fit_transform(df_enc['interior'])

cols = df_enc.select_dtypes(include='object').columns
df_enc = pd.get_dummies(df_enc, columns=cols, drop_first=True, dtype=int)

display(df_enc.head())

# 훈련데이터와 검증데이터 분리
from sklearn.model_selection import train_test_split
X=df_enc.drop(columns='sellingprice', axis=1)
y= df_enc['sellingprice']

X_train, X_test, y_train, y_test = train_test_split(
X,y, test_size=0.3, random_state=120)

# 스케일링
x_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
x_test = (X_test - X_train.min()) / (X_train.max() - X_train.min())

print(x_train)

# 차량 거래 가격 예측 머신러닝 모델
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

lr = LinearRegression(n_jobs=-1)
gbr = GradientBoostingRegressor(max_depth=7, n_estimators=40, random_state=120 )

lr.fit(x_train, y_train)
gbr.fit(x_train, y_train)

# 성능 평가(r2_score)
from sklearn.metrics import r2_score
y_pred_ly = ly.predict(x_test)
y_pred_gbr = gbr.predict(x_test)
print(r2_score(y_test, y_pred_ly))
print(r2_score(y_test, y_pred_gbr))

# 피처 중요도
ip = pd.DataFrame({'Feature':x_train.columns, 'Importance':gbs.feature_importances_})
ip.sort_values('importance', ascending=False)

# 딥러닝 인포트
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

# 딥러닝 모델
# 모델 구조 설계
model = Seqeuntial([
  Dense(64, 'relu'),
  Dense(128, 'relu'),
  Dense(256, 'relu'),
  BatchNormalization(),
  Dense(1, 'linear')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metics=['mse', 'mae'])

# 조기종료 설정
es = EarlyStopping(
  monitor = 'val_loss'
  patience=5
  restory_best_weights=True
)

# 모델 저장 설정
mc = ModelCheckpoint('best_model.h5', save_best_only=True)

# 학습실행
history = model.fit(
  x_train,
  y_train,
  epochs= 150,
  batch_size= 32,
  validation_data = (x_test, y_test),
  callbacks=[es, mc]
)

# 딥러닝 성능 평가
plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='test_mae')
plt.legend()
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')

# 시뮬레이션
simul = np.array([0.70833333,0.74590164,1.,0.0202109,0.07142857,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.]).reshape(1, -1)
best_dl = load_model('best_model.h5')
print(best_dl.predict(simul))


