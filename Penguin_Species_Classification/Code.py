# 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
import warnings
warnings.filterwarnings('ignore')

# 파일 읽기 및 분석
df = pd.read_csv('penguins.csv')
len(df.columns)
df.isnull().sum()
df['species'].value_counts()
df['body_mass_g'].mean()
df['body_mass_g'].median()

# 종별 펭귄 수 시각화
sns.countplot(df, x='species')

# 종별 채중 분포 시각화
sns.boxplot(df, x='species', y='body_mass_g')

# 종별 평균 부리 길이
df.groupby('species')['bill_length_mm'].mean()

# 섬 별 종의 분포 시각화
sns.countplot(df, x='island', hue='species')

# 결측치 확인
df.isnull().sum()

# 히트맵으로 수치형 변수들끼리 상관관계 확인
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')

# 결측치 제거
df.dropna(inplace=True)

# 인코딩
df['sex'] = df['sex'].replace({'Male':1, 'Female':0})
df = pd.get_dummies(df, columns=['island'], drop_first=True)

# 인코딩2
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# 데이터 분할
from sklearn.model_selection import train_test_split
X = df.drop('species', axis=1) #!!!
y = df['species']
X_train, X_valid, y_train, y_valid = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_valid.shape)

# 스케일링
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_valid_s = sc.fit_transform(X_valid)

# 랜덤포레스트로 학습시키고 예측 정확도 확인
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X_train_s, y_train)
pred_rfc = rfc.predict(X_valid_s)
print(accuracy_score(y_valid, pred_rfc))

# 분류 리포트와 혼동행렬 출력
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_valid, pred_rfc))
print(confusion_matrix(y_valid, pred_rfc))

# 피처 중요도
importance = pd.Series(rfc.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))

# 딥러닝 모델 시퀀셜
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
Dense(64, 'relu'),
Dense(32, 'relu'),
Dense(3, 'softmax')])

model.compile('adam', 'sparse_categorical_crossentropy',
             metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train_s, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_valid_s, y_valid),
    callbacks=[checkpoint],
    verbose=1
)

# 선 그래프
plt.plot(history.history['loss'], label='Train Loss')
# 선 그래프
plt.plot(history.history['val_loss'], label='Validation Loss')
# 범례 표시
plt.legend()
# 그래프 제목
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# 그래프 출력
plt.show()

# 저장된 모델 시뮬레이션
from tensorflow.keras.models import load_model
best_model = load_model('best_model.h5')
# 예측 수행
pred_simul = best_model.predict(simul_s)
print('예측 확률:', pred_simul)
print(f'Adelie: {pred_simul[0][0]:.4f}')
print(f'Chinstrap: {pred_simul[0][1]:.4f}')
print(f'Gentoo: {pred_simul[0][2]:.4f}')
