# AICE Associate 모의고사
## 복습
### 8번) 변수들의 상관관계 확인
- plt.figure(figsize=(10,5))  # 창 띄우기
- sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
  - df.corr()  # 상관관계 함수
  - numeric_only=Ture  # 수치형 변수들만
  - annot=True  # 셀 안에 숫자
  - cmap='coolwarm'  # 높을 수록 빨강, 낮을 수록 파랑
  - fmt='.2f'  # 소수점 두번째 자리까지만
  
### 10번) 범주형 변수 인코딩
- df['sex'] = df['sex'].replace({'Male':1, 'Female':0})
  - replace할 때는 특정 칼럼 지정해서 바꿔야함.
- df = pd.get_dummies(df, columns=['island'], drop_first=True)
  - 원래는 겟 더미스 하기 전에 카피 데이터로 만들고 해야함
- from sklearn.preprocessing import LabelEncoder  #임포트
- le = LabelEncoder()  #()필수임
- df['species'] = le.fit_transform(df['species'])
  - .fit_transform()  #형식 암기
 
### 12번) 스케일링
- from sklearn.preproessing import StandardScaler  #임포트
- sc = StandardScale()   #()필수
- X_train_s = sc.fit_transform(X_train)
- X_valid_s = sc.fit_transform(X_valid)

### 13번) 랜덤포레스트 학습
- from sklearn.ensemble import RandomForestClassifier
  - .ensemble 암기
- from sklearn.matrics import accuracy_score
  - .matrics 암기
- rfc = RandomForestClassifier(n_estimators=100, random_state=42)
- rfc.fit(X_train_s, y_train)
  - 학습 시키기
- pred_rfc = rfc.predict(X_valid_s)
  - 예측 시키기
- accuracy_score(y_valid, pred_rfc)
  - '실제 값, 예측 값' 으로 정확도 출력
  
### 14번) 분류 리포트와 혼동행렬
- from sklearn.metrics import classification_report, confusion_matrix
  - 성능평가 라이브러리는 다 .metrics 인가보다
- classification_report(y_valid, pred_rfc)
- confusion_matrix(y_valid, pred_rfc)

















