# 핵심 단계:
- 환경 준비 -> 구조확인 -> 기초 통계 -> 결측/이상치 탐색 -> 시각화 -> 상관관계 분석

## 구조환경
### head()
- df.head()
- 기본 처음 5개 행 불러오기
### info()
- df.info()
- 칼럼 갯수, 결측치 여부, 자료형 확인
### describe()
- df.describe() : 수치형
- df.describe(include="object") : 범주형
- 평균, 표준편차, 최소최대값, 사분위 수 확인

## 시각화
### countplot
- sns.countplot(data=df, x="컬럼명")
- 범주형 데이터 갯수 비교
- 시험point: "불균형 클래스인가요?"
### hisplot
- sns.hisplot(df["컬럼명"], kde=True)
- 숫자 데이터, 왜도 분포 형태 확인
- kde=True : 곡선까지 같이 표시
### boxplot
- sns.boxplot(x=df["컬럼명"])
- 이상치 찾기
### heatmap
- corr = df.corr(numeric_only=True
- sns.heatmap(corr, annot=True, cmap="coolwarm")
- 상관관계 확인
- annot=True : 셀 안에 상관관계 숫자 표시
### implot
- sns.implot(data=df, x="칼럼", y="칼럼")
- 두 변수 관계를 회귀선으로 표시
### jointplot
- sns.jointplot(data=df, x="", y="", kind="scatter")
- 산점도, 양쪽 분포 한번에 확인







