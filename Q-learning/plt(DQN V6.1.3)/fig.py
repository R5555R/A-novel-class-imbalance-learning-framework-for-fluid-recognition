import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score


df = pd.read_csv('./RL99/RL_data.csv')
y_test = df[['label']].values
test_pred = df[['pre']].values

from sklearn.metrics import recall_score,  f1_score, precision_score
print(np.mean(precision_score(y_test, test_pred, average=None)))
print(np.mean(recall_score(y_test, test_pred, average=None)))
print(np.mean(f1_score(y_test, test_pred, average=None)))

C2 = confusion_matrix(y_test, test_pred, labels=[0, 1, 2, 3, 4, 5], sample_weight=None)
print(C2)
sns.set(font="simhei")  # 遇到标签需要汉字的可以在绘图前加上这句
f, ax = plt.subplots()
C3 = sns.heatmap(C2, annot=True, cmap="Blues", ax=ax, fmt='g',
                 xticklabels=["气层", "气水同层", "差气层", "含气水层", "水层", "干层"],
                 yticklabels=["气层", "气水同层", "差气层", "含气水层", "水层", "干层"])
ax.set_title('XGBoost+ADASYN')  # 标题
ax.set_xlabel('Output Class')  # x轴
ax.set_ylabel('Target Class')  # y轴
# plt.savefig('./RL_data.svg',bbox_inches='tight')
plt.show()
print(recall_score(y_test, test_pred, average=None) * 100)
print(np.mean(recall_score(y_test, test_pred, average=None) * 100))