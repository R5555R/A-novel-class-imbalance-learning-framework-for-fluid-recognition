import os

import gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier  # GDBT梯度提升树
from sklearn.ensemble import ExtraTreesClassifier  # ET极限森林
from catboost import CatBoostClassifier  # CatBoost
from xgboost import XGBClassifier  # XGBoost
from imblearn.over_sampling import ADASYN
from collections import deque
import xgboost as xgb
import sklearn.metrics as metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time


le = LabelEncoder()

# load data
train_data = pd.read_csv('..//data//精度1//E-train-1.csv')
train_data1 = pd.read_csv('..//data//精度1//无标签126层数据（8口井）.csv')
test_data = pd.read_csv('..//data//精度1//E-test.csv')

# LABEL = le.fit_transform(train_data['LABEL'])

# # 合并所有数据
# data = pd.concat((train_data, train_data1), axis=0)
data = train_data
features = ['SP', 'PE', 'GR', 'U', 'TH', 'K', 'AC', 'CNL', 'DEN', 'RLLS', 'RLLD']
#
LABEL = le.fit_transform(data['LABEL'])

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(data[features].values, LABEL, test_size=0.2)

X_train = x_train
X_test = x_test
Y_train = y_train
Y_test = y_test

# # 制作顺序数据集
# X_train = train_data[features].values
# X_test = test_data[features].values
# Y_train = le.fit_transform(train_data['LABEL'])
# Y_test= le.fit_transform(test_data['LABEL'])

one_hot = OneHotEncoder(sparse=False)
one_hot.fit(y_train.reshape(-1, 1))

def imbalanced_data_model():
    # 合并所有数据
    data = pd.concat((train_data, test_data), axis=0)
    features = ['SP', 'PE', 'GR', 'U', 'TH', 'K', 'AC', 'CNL', 'DEN', 'RLLS', 'RLLD']
    le = LabelEncoder()
    LABEL = le.fit_transform(data['LABEL'])

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data[features].values, LABEL, test_size=0.2)

    weights = []
    # [1, 9, 18, 3, 15, 9]
    for label in y_train:
        if label == 0:
            weights.append(1)  # 类别0的样本权重为10
        elif label == 1:
            weights.append(9)  # 类别1的样本权重为1
        elif label == 2:
            weights.append(18)  # 类别2的样本权重为2
        elif label == 3:
            weights.append(3)  # 类别3的样本权重为3
        elif label == 4:
            weights.append(15)  # 类别4的样本权重为1
        else:
            weights.append(9)  # 类别5的样本权重为1

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)

    # 设置XGBoost参数
    params = {'objective': 'multi:softprob', 'num_class': 6}

    # 训练模型
    num_rounds = 50
    model = xgb.train(params, dtrain, num_rounds)

    # 在测试集上进行预测
    # dtest = xgb.DMatrix(X_test)
    # test_pred = model.predict(dtest)
    # test_pred = np.argmax(test_pred, axis=1)
    # print(accuracy_score(y_test, test_pred))
    return model

def ADASYN_model():
    # 合并所有数据
    data = pd.concat((train_data, test_data), axis=0)
    features = ['SP', 'PE', 'GR', 'U', 'TH', 'K', 'AC', 'CNL', 'DEN', 'RLLS', 'RLLD']
    le = LabelEncoder()
    LABEL = le.fit_transform(data['LABEL'])

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data[features].values, LABEL, test_size=0.2)
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X_train, y_train)

    xgb = XGBClassifier()
    xgb.fit(X_res, y_res)
    print(accuracy_score(y_test, xgb.predict(X_test)))
    return xgb

# Train each decision tree on the training data
models = []
estimator0 = GradientBoostingClassifier()
estimator1 = ExtraTreesClassifier()
estimator2 = CatBoostClassifier(verbose=False)
estimator3 = XGBClassifier()

estimator = [estimator0, estimator1, estimator2, estimator3]
for clf in estimator:
    clf.fit(x_train, y_train)
    models.append(clf)
print([accuracy_score(y_test, k.predict(x_test)) for k in models])
models.append(imbalanced_data_model())
models.append(ADASYN_model())

# 读取李31井的数据
# li31_path = '../data/精度1/31.csv'
# li31 = pd.read_csv(li31_path)

li27_path = './li20.csv'
li31 = pd.read_csv(li27_path)

features = ['SP', 'PE', 'GR', 'U', 'TH', 'K', 'AC', 'CNL', 'DEN', 'RLLS', 'RLLD']
x_li31 = li31[features].values
y_li31 = li31['LABEL']-1
blind = li31[features]
blind.insert(loc=0, column='Depth', value=li31[['DEPT']])
blind['LABEL'] = y_li31

#GBDT
yhat_blind_GBDT = models[0].predict(x_li31)
blind['GBDT_Pred'] = yhat_blind_GBDT

#ET
yhat_blind_ET = models[1].predict(x_li31)
blind['ET_Pred'] = yhat_blind_ET

#CAT
yhat_blind_CAT = models[2].predict(x_li31)
blind['CATBOOST_Pred'] = yhat_blind_CAT

#XGB
yhat_blind_XGB = models[3].predict(x_li31)
blind['XGBOOST_Pred'] = yhat_blind_XGB

#Weight
yhat_blind_WEIGHT = xgb.DMatrix(x_li31)
yhat_blind_WEIGHT = models[4].predict(yhat_blind_WEIGHT)
yhat_blind_WEIGHT = np.argmax(yhat_blind_WEIGHT, axis=1)
blind['WEIGHT_Pred'] = yhat_blind_WEIGHT

#ADASYN
yhat_blind_ADASYN = models[5].predict(x_li31)
blind['ADASYN_Pred'] = yhat_blind_ADASYN

blind.to_excel('reasult_li20.xlsx',index=False)
print('结果已导出！！！！！！！！！！！！！！！！！')



class Environment(object):
    def __init__(self):
        self.predictions = np.zeros(6).reshape(1, -1)
        self.pred = []

    def reset(self):
        # 创建包含 10 个长度为 18 的全 0 子列表的列表
        lst = [[0] * 18 for _ in range(len_deque)]
        return deque(lst, maxlen=len_deque)

    def step(self, state, weights, model, X, y):
        # Compute the predictions using the weighted model
        self.predictions = np.zeros(6).reshape(1, -1)
        self.pred = []
        y_true = one_hot.transform(y.reshape(-1, 1)).reshape(1, -1)
        for i, clf in enumerate(model):
            if i == 4:
                dtest = xgb.DMatrix(X)
                clf_pred = clf.predict(dtest)
                self.predictions += weights[i] * clf_pred
                self.pred.append(clf_pred.tolist())
            else:
                clf_pred = clf.predict_proba(X)
                self.predictions += weights[i] * clf_pred
                self.pred.append(clf_pred.tolist())

        # Compute the accuracy of each model's predictions
        loss_clf = [log_loss(y_true, _) for _ in self.pred]
        loss_clf = sum(loss_clf)/len(loss_clf)

        # Compute the accuracy of the predictions
        loss_clfs = log_loss(y_true, self.predictions)

        # Compute the reward
        reward = -loss_clfs-loss_clf



        y_pred = np.argmax(self.predictions, axis=1)
        # # 设计loss为预测正确和错误
        # if y_pred == 1 or y_pred ==2 or y_pred ==4:
        #     if y_pred == y:
        #         reward = 1
        #     else:
        #         reward = -1
        # else:
        #     if y_pred == y:
        #         reward = Lambda
        #     else:
        #         reward = -Lambda


        accuracy = accuracy_score(y, y_pred)

        # 将新的特征、预测结果、标准差和性能打包放入state
        d = []
        d.extend(X.tolist()[0])
        d.extend(self.predictions.tolist()[0])
        d.append(loss_clf)
        # 将数据添加到队列的末尾
        state.append(d)
        return state, reward, accuracy, y_pred

# 定义Q-learning代理
class NormalizeOutput(nn.Module):
    def forward(self, x):
        x = nn.functional.softmax(x, dim=1)
        x = x / torch.sum(x, dim=1, keepdim=True)
        return x

class OUTPUT(nn.Module):
    def forward(self, x):
        print(x.shape)
        print(x)
        return x

class QLearningAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def _build_network(self):
        network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_dim * len_deque, self.action_dim),
            # nn.Linear(self.hidden_dim * 10, self.action_dim),
            NormalizeOutput(),  # 添加标准化输出的自定义层
            # OUTPUT()
        )
        return network

    def act(self, state, istrain):
        # if istrain:
        #     if np.random.rand() < self.epsilon:
        #         x = np.random.rand(6)
        #         return x/sum(x)
        #     else:
        #         state = torch.tensor(list(state), dtype=torch.float32).to(self.device).unsqueeze(0)
        #         with torch.no_grad():
        #             q_values = self.q_network(state)
        #             return q_values[0].cpu().numpy()
        # else:
        #     state = torch.tensor(list(state), dtype=torch.float32).to(self.device).unsqueeze(0)
        #     with torch.no_grad():
        #         q_values = self.q_network(state)
        #         return q_values[0].cpu().numpy()

        if np.random.rand() < self.epsilon:
            x = np.random.rand(6)
            return x
        else:
            state = torch.tensor(list(state), dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
                return q_values[0].cpu().numpy()

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(list(state), dtype=torch.float32).to(self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(list(next_state), dtype=torch.float32).to(self.device).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.q_network(state)
        q_value = torch.sum(q_values * action)

        with torch.no_grad():
            next_q_values = self.q_network(next_state)
            next_q_value = torch.max(next_q_values)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 定义训练函数
def train(agent, env, episodes, max_steps, X_train):
    scores = []
    for episode in range(episodes):
        state = env.reset()
        score = 0
        for step in tqdm(range(len(X_train))):
            action = agent.act(state, istrain=False)
            next_state, reward, accuracy, y_pred = env.step(state, action, models, X_train[step:step + 1], Y_train[step:step + 1])
            agent.learn(state, action, reward, next_state, done=False)

            score += reward
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"Episode {episode + 1}/{episodes}, Score: {score:.2f}, Average Score: {avg_score:.2f}")
        # if avg_score >= 195:
        #     print(f"Solved in {episode + 1} episodes!")
        #     break
    return scores


# 创建CartPole环境和Q-learning代理，然后开始训练
env = Environment()
len_deque = 10
state_dim = 18
action_dim = 6
hidden_dim = 128
lr = 0.001
gamma = 0.99
epsilon = 0.1
Lambda = 0.5
agent = QLearningAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon)
episodes = 1
max_steps = 1000
scores = train(agent, env, episodes, max_steps, X_train)

# # 可视化训练得分
# import matplotlib.pyplot as plt
#
# plt.plot(scores)
# plt.xlabel("Episode")
# plt.ylabel("Score")
# plt.title("Training Scores")
# plt.show()

# 使用训练好的代理玩一次游戏
correct = 0
total = 0
teacher_pred = []
y_predictions = []
state = env.reset()
for i in range(len(X_test)):
    weights = agent.act(state, istrain=False)
    pred = []
    predictions = np.zeros(6).reshape(1, -1)
    for j, clf in enumerate(models):
        if j == 4:
            dtest = xgb.DMatrix(X_test[i].reshape(1, -1))
            clf_pred = clf.predict(dtest)
            predictions += weights[j] * clf.predict(dtest)
            pred.append(clf_pred.tolist())
        else:
            clf_pred = clf.predict_proba(X_test[i].reshape(1, -1))
            predictions += weights[j] * clf.predict_proba(X_test[i].reshape(1, -1))
            pred.append(clf_pred.tolist())

    # Compute the accuracy of the predictions
    y_true = one_hot.transform(Y_train[i].reshape(-1, 1)).reshape(1, -1)

    # Compute the accuracy of each model's predictions
    loss_clf = [log_loss(y_true, _) for _ in pred]
    loss_clf = sum(loss_clf) / len(loss_clf)

    # Compute the accuracy of the predictions
    loss_clfs = log_loss(y_true, predictions)


    # 将新的特征、预测结果、标准差和性能打包放入state
    d = []
    d.extend(X_test[i:i + 1].tolist()[0])
    d.extend(predictions.tolist()[0])
    d.append(loss_clf)
    # 将数据添加到队列的末尾
    state.append(d)

    y_pred = np.argmax(predictions, axis=1)
    if y_pred == Y_test[i]:
        correct += 1
    total += 1

    y_predictions.append(y_pred)
    teacher_pred.append(predictions.tolist()[0])

# Print the accuracy on the test set
print("评价指标-1：(X_test)")
print(metrics.classification_report(Y_test, y_predictions))
print("Accuracy on test set: {:.2f}%".format(correct / total * 100))


li31_path = '../data/精度1/31.csv'
li31 = pd.read_csv(li31_path)

features = ['SP', 'PE', 'GR', 'U', 'TH', 'K', 'AC', 'CNL', 'DEN', 'RLLS', 'RLLD']
x_li31 = li31[features].values
y_li31 = le.fit_transform(li31['LABEL'])
li31_pred = []

state = env.reset()
correct = 0
total = 0
for i in range(len(li31)):
    weights = agent.act(state, istrain=False)
    print(weights)
    pred = []
    predictions = np.zeros(6).reshape(1, -1)
    for j, clf in enumerate(models):
        if j == 4:
            dtest = xgb.DMatrix(x_li31[i].reshape(1, -1))
            clf_pred = clf.predict(dtest)
            predictions += weights[j] * clf.predict(dtest)
            pred.append(clf_pred.tolist())
        else:
            clf_pred = clf.predict_proba(x_li31[i].reshape(1, -1))
            predictions += weights[j] * clf.predict_proba(x_li31[i].reshape(1, -1))
            pred.append(clf_pred.tolist())

    # Compute the accuracy of the predictions
    y_true = one_hot.transform(y_li31[i].reshape(-1, 1)).reshape(1, -1)

    # Compute the accuracy of each model's predictions
    loss_clf = [log_loss(y_true, _) for _ in pred]
    loss_clf = sum(loss_clf) / len(loss_clf)

    # Compute the accuracy of the predictions
    loss_clfs = log_loss(y_true, predictions)


    # 将新的特征、预测结果、标准差和性能打包放入state
    d = []
    d.extend(X_test[i:i + 1].tolist()[0])
    d.extend(predictions.tolist()[0])
    d.append(loss_clf)
    # 将数据添加到队列的末尾
    state.append(d)

    y_pred = np.argmax(predictions, axis=1)
    if y_pred == y_li31[i]:
        correct += 1
    total += 1

    li31_pred.append(y_pred.tolist()[0])

li31.to_excel('li31.xlsx', index=False)
print(li31_pred)
# Print the accuracy on the test set
print("评价指标-1：(li31)")
print(metrics.classification_report(y_li31, li31_pred))
print("Accuracy on test set: {:.2f}%".format(correct / total * 100))