import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

df1 = pd.read_csv('./半监督96/imbalanced_data.csv')
df2 = pd.read_csv('./ADASYN95/ADASYN_data.csv')
df3 = pd.read_csv('./RL99/RL_data.csv')

test_pred = df3['pre']
y_test = df3['label']

pre = precision_score(y_test,test_pred,average=None)
print(np.round(pre, 3))
print(round(np.mean(pre), 3))

rec = recall_score(y_test,test_pred,average=None)
print(np.round(rec, 3))
print(round(np.mean(rec), 3))

f1 = f1_score(y_test,test_pred,average=None)
print(np.round(f1, 3))
print(round(np.mean(f1), 3))