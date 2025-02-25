from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor  # GDBT梯度提升树
from sklearn.ensemble import ExtraTreesRegressor  # ET极限森林
from catboost import CatBoostRegressor  # CatBoost
# from torch.distributions.constraints import one_hot
from xgboost import XGBRegressor  # XGBoost
from collections import deque
import xgboost as xgb
import sklearn.metrics as metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time
# from train_model import sourcePre
from makedata import makedata,getdata


le = LabelEncoder()

x_train, x_test, y_train, y_test = getdata()

X_train = x_train
X_test = x_test
Y_train = y_train
Y_test = y_test


# one_hot = OneHotEncoder(sparse=False)
# one_hot.fit(y_train.reshape(-1, 1))


# Train each decision tree on the training data
models = []
estimator0 = GradientBoostingRegressor()
estimator1 = ExtraTreesRegressor()
estimator2 = CatBoostRegressor(verbose=False)
estimator3 = XGBRegressor()

estimator = [estimator0, estimator1, estimator2, estimator3]
for reg in estimator:
    reg.fit(x_train, y_train)
    models.append(reg)
# print([accuracy_score(y_test, k.predict(x_test)) for k in models])

class Environment(object):
    def __init__(self):
        self.predictions = np.zeros(6).reshape(1, -1)
        self.pred = []

    def reset(self):
        # 创建包含 10 个长度为 8 的全 0 子列表的列表
        lst = [[0] * 9 for _ in range(len_deque)]
        return deque(lst, maxlen=len_deque)

    def step(self, state, weights, model, X, y):
        # Compute the predictions using the weighted model
        self.predictions = np.zeros(1).reshape(1, -1)
        self.pred = []
        y_true = y
        for i, clf in enumerate(model):
            if i == 4:
                dtest = xgb.DMatrix(X)
                clf_pred = clf.predict(dtest)
                self.predictions += weights[i] * clf_pred
                predictions_ = self.predictions
                self.pred.append(clf_pred.tolist())
                pred_ = self.pred
            else:
                # clf_pred = clf.predict_proba(X)
                clf_pred = clf.predict(X)
                self.predictions += weights[i] * clf_pred
                predictions_ = self.predictions
                self.pred.append(clf_pred.tolist())
                pred_ = self.pred
        # Compute the accuracy of each model's predictions
        loss_clf = [mean_squared_error(y_true, _) for _ in self.pred]
        loss_clf = sum(loss_clf)/len(loss_clf)

        # Compute the accuracy of the predictions
        loss_clfs = mean_squared_error(y_true, self.predictions)

        # Compute the reward
        # reward = (-loss_clfs - loss_clf)*apha
        reward = -loss_clfs - loss_clf
        # y_pred = np.argmax(self.predictions, axis=1)
        accuracy = mean_squared_error(y, predictions_)     #均方误差


        # 将新的特征、预测结果、标准差和性能打包放入state
        d = []
        d.extend(X.tolist()[0])
        d.extend(self.predictions.tolist()[0])
        d.append(loss_clf)
        # 将数据添加到队列的末尾
        state.append(d)
        return state, reward, accuracy, predictions_

# 定义Q-learning代理
class NormalizeOutput(nn.Module):
    def forward(self, x):
        x = nn.functional.softmax(x, dim=1)
        x = x / torch.sum(x, dim=1, keepdim=True)
        return x

class OUTPUT(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x

class QLearningAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            NormalizeOutput()  # 添加标准化输出的自定义层
        )
        return network

    def save_model(self, file_path='./'):
        torch.save(self.q_network.state_dict(), file_path)

    def load_model(self, file_path='./'):
        self.q_network.load_state_dict(torch.load(file_path))
        self.q_network.eval()  # 切换到评估模式

    def act(self, state):
        if np.random.rand() < self.epsilon:
            x = np.random.rand(4)
            # return x / sum(x)
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
# def train(agent, env, episodes, X_train):
#     state = env.reset()
#     score = 0
#     for step in tqdm(range(len(X_train))):
#         action = agent.act(state)
#         next_state, reward, accuracy, y_pred = env.step(state, action, models, X_train[step:step + 1], Y_train[step:step + 1])
#         agent.learn(state, action, reward, next_state, done=False)
#         score += reward
#     # avg_score = np.mean(scores[-100:])
#     print(f"Episode {episode + 1}/{episodes}, Score: {score:.2f}")
#     return score


def train(agent, env, episodes, X_train):
    scores = []
    for episode in range(episodes):
        state = env.reset()
        score = 0
        for step in tqdm(range(len(X_train))):
            action = agent.act(state)
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
len_deque = 1
state_dim = 9   #21
action_dim = 4
hidden_dim = 64
lr = 0.001
gamma = 0.9
epsilon = 0.1
apha = 0.05
Lambda = 0.5
agent = QLearningAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon)
episodes = 10
scores = []
acc = []

start_time = time.time()
scores = train(agent, env, episodes, X_train)
end_time = time.time()
execution_time = end_time - start_time
print(f"运行时间: {execution_time:.2f} seconds")
##保存模型
agent.save_model('model_weights.pth')


scores = pd.DataFrame(scores)
scores.to_csv('return.csv',index=False)

#回执return图
import matplotlib.pyplot as plt
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("return")
plt.title("Training return")
plt.show()
plt.savefig("training_return.png")



#################################使用训练好的代理玩一次游戏####################################
#加载模型
agent.load_model('model_weights.pth')

correct = 0
total = 0
teacher_pred = []
y_predictions = []
y_th = []
state = env.reset()
for i in range(len(X_test)):
    weights = agent.act(state)
    pred = []
    predictions = np.zeros(1)
    for j, clf in enumerate(models):
        if j == 4:
            dtest = xgb.DMatrix(X_test[i].reshape(1,-1))
            clf_pred = clf.predict(dtest)
            predictions += weights[j] * clf_pred
            pred.append(clf_pred.tolist())
        else:
            clf_pred = clf.predict(X_test[i].reshape(1,-1))
            predictions += weights[j] * clf_pred
            pred.append(clf_pred.tolist())

    # Compute the accuracy of the predictions
    # y_true = one_hot.transform(Y_test[i].reshape(1,-1))
    y_true=Y_test[i]

    # Compute the accuracy of each model's predictions
    loss_clf = [mean_squared_error(y_true, _) for _ in pred]
    loss_clf = sum(loss_clf) / len(loss_clf)

    # Compute the accuracy of the predictions
    loss_clfs = mean_squared_error(y_true, predictions)

    # Compute the reward
    # reward = (-loss_clfs - loss_clf)*apha
    reward = -loss_clfs - loss_clf

    # 将新的特征、预测结果、标准差和性能打包放入state
    d = []
    d.extend(X_test[i:i + 1].tolist()[0])
    d.append(predictions[0])
    d.append(loss_clf)
    # 将数据添加到队列的末尾
    state.append(d)

    y_pred = predictions
    y_predictions.append(y_pred.tolist()[0])
    y_th.append(Y_test[i].tolist())

    acc.append(mean_squared_error(y_th, y_predictions))
    # print(mean_squared_error(y_th, y_predictions))
acc = pd.DataFrame(acc)
acc.to_csv('test_acc.csv',index=False)
print("ture:",y_th)
print("predict:",y_predictions)


print("mse :",mean_squared_error(y_th,y_predictions))
print("mae :",mean_absolute_error(y_th,y_predictions))
print("r2  :",r2_score(y_th,y_predictions))
# 可视化训练得分
import matplotlib.pyplot as plt
plt.plot(acc)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Training Scores")
plt.show()
plt.savefig("acc.png")



plt.plot(y_th, label="y_th", marker='o')
plt.plot(y_predictions, label="y_predictions", marker='*')
plt.legend()
plt.title("Comparison of y_th and y_predictions")
plt.xlabel("Index")
plt.ylabel("Values")
plt.show()
plt.savefig("true and predict.png")




