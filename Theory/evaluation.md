# 종류 

## 혼동 행렬 (Confusion Matrix)
- from sklearn.metrics import confusion_matrix
- cm = confusion_matrix(y_test, y_pred)
  - 중요: (y_true, y_pred) 순서!
[[TN  FP]
[FN  TP]]
- TN (True Negative)
: 0으로 예측, 실제 0 - 정답!
- FP (False Positive) - 1종 오류
: 1로 예측, 실제 0 - 오답!
- FN (False Negative) - 2종 오류
: 0으로 예측, 실제 1 - 오답!
- TP (True Positive)
: 1로 예측, 실제 1 - 정답!

## 정밀도, 재현율, F1-score
- from sklearn.metrics import classification_report
- print(classification_report(y_test, y_pred))

## ROC 곡선과 AUC
- from sklearn.metrics import roc_curve, roc_auc_score
- import matplotlib.pyplot as plt
- FPR = FP / (FP + TN)
- TPR = TP / (TP + FN) = 재현율
- X축: FPR (오탐률)
- Y축: TPR (탐지율)
