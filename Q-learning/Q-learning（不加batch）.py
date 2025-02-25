import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier # GDBT梯度提升树
from sklearn.ensemble import ExtraTreesClassifier # ET极限森林
from catboost import CatBoostClassifier # CatBoost
from xgboost import XGBClassifier # XGBoost
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mode
from sklearn.metrics import log_loss
from tqdm import tqdm

# # 载入 iris 数据集
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# load data
train_data = pd.read_csv('..//data//精度1//E-train-1.csv')
train_data1 = pd.read_csv('..//data//精度1//无标签126层数据（8口井）.csv')
test_data = pd.read_csv('..//data//精度1//E-test.csv')
# 合并所有数据
data = pd.concat((train_data,train_data1,test_data), axis=0)
features = ['SP','PE','GR','U','TH','K','AC','CNL','DEN','RLLS','RLLD']
train_x = data[features]
le = LabelEncoder()
LABEL = le.fit_transform(data['LABEL'])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data[features].values, LABEL, test_size=0.2)

# 训练三个决策树基分类器
estimator0 = GradientBoostingClassifier()
estimator1 = ExtraTreesClassifier()
estimator2 = CatBoostClassifier()
estimator3 = XGBClassifier()

estimator = [estimator0, estimator1, estimator2, estimator3]
base_classifiers = []

for i in estimator:
    clf = i
    clf.fit(X_train, y_train)
    base_classifiers.append(clf)
print([accuracy_score(y_test, k.predict(X_test)) for k in base_classifiers])

# max_depth = [4,5,6]
# for i in max_depth:
#     clf = DecisionTreeClassifier(max_depth=i)
#     clf.fit(X_train, y_train)
#     # print(clf.predict_proba(X_train))
#     base_classifiers.append(clf)
# print(accuracy_score(y_test, base_classifiers[0].predict(X_test)))
# print('end')

# 定义 Q-learning 算法
num_classes = 6
Q = np.zeros((num_classes, 3))  # Q 表
epsilon = 0.9  # ε-greedy 策略的 ε 值
alpha = 0.01  # 学习率
gamma = 0.9  # 折扣因子

# 初始化权重和偏差
weights = 0
bias = 0
one_hot = OneHotEncoder(sparse=False)
one_hot.fit(y_train.reshape(-1, 1))

# ε-greedy 策略选择动作
def epsilon_greedy(Q, state):
    if np.random.uniform() < epsilon:
        action = np.random.choice([0, 1, 2])
    else:
        action = np.argmax(Q[state, :])
    return action


# 训练 Q-learning 模型
for episode in tqdm(range(100)):
    state = 0  # 初始状态
    # state = np.random.randint(3)  # 初始状态
    for i in range(len(X_train)):
        action = epsilon_greedy(Q, state)  # 选择动作

        if action == 0:
            probs = [clf.predict_proba([X_train[i]])[0] for clf in base_classifiers]
            # result = np.argmax(probs, axis=1)
            result = mode(np.argmax(np.array(probs),axis=1), keepdims=True)[0][0]
        elif action == 1:
            probs = [clf.predict_proba([X_train[i]])[0] for clf in base_classifiers]
            # result = np.argsort(-np.array(probs), axis=1)[1]
            result = mode(np.argsort(-np.array(probs), axis=1)[:, 1], keepdims=True)[0][0]
        else:
            probs = [clf.predict_proba([X_train[i]])[0] for clf in base_classifiers]
            result = np.random.choice([0, 1, 2])

        # reward = 1 if result == y_train[i] else -1  # 计算奖励
        # 重新写奖励
        # 计算基分类器的预测结果和交叉熵损失
        y_true = one_hot.transform(y_train[i].reshape(-1, 1)).reshape(-1,)
        # print('----------')
        # print(y_true)
        # print(probs[0])
        loss_clf = [log_loss(y_true, pred) for pred in probs]
        loss_clf = sum(loss_clf)
        # print(loss_clf)

        # 计算强化学习模型的预测结果和交叉熵损失
        y_pred_rl = one_hot.transform(result.reshape(-1, 1)).reshape(-1,)
        # y_pred_rl = weights * np.array(y_pred_rl) + bias
        loss_rl = log_loss(y_true, y_pred_rl)
        reward = -loss_clf - loss_rl

        next_state = result  # 计算下一个状态
        Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * np.max(Q[next_state, :]) - Q[state, action])  # 更新 Q 表

        # # 更新权重和偏差
        # y_pred_base = np.array(probs)
        # y_pred_base_weighted = y_pred_base * weights
        # y_pred_rl = np.sum(y_pred_base_weighted, axis=1) + bias
        # error = y_test - y_pred_rl
        # weights = weights + alpha * np.mean(np.multiply(error[:, np.newaxis], y_pred_base), axis=0)

        state = next_state  # 跳转到下一个状态

# 测试 Q-learning 模型
print(Q)
correct = 0
state = 0
pred = []
true = []
for i in range(len(X_test)):
    action = np.argmax(Q[state, :])  # 选择动作
    # print(action)

    if action == 0:
        probs = [clf.predict_proba([X_test[i]])[0] for clf in base_classifiers]
        # result = np.argmax(probs, axis=1)
        result = mode(np.argmax(np.array(probs), axis=1), keepdims=True)[0][0]
    elif action == 1:
        probs = [clf.predict_proba([X_test[i]])[0] for clf in base_classifiers]
        # result = np.argsort(-np.array(probs), axis=1)[1]
        result = mode(np.argsort(-np.array(probs), axis=1)[:, 1], keepdims=True)[0][0]
    else:
        probs = [clf.predict_proba([X_test[i]])[0] for clf in base_classifiers]
        result = np.random.choice([0, 1, 2])
    pred.append(result)
    if result == y_test[i]:
        correct += 1
import sklearn.metrics as metrics
print("评价指标-1：")
print(metrics.classification_report(y_test,pred))
print("Accuracy:", correct / len(X_test))
