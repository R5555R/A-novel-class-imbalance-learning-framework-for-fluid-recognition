from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
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
import math

import warnings
warnings.filterwarnings(action='ignore')

le = LabelEncoder()
x_train, x_test, y_train, y_test = makedata()
X_train = x_train
X_test = x_test
Y_train = y_train
Y_test = y_test


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
        self.predictions = np.zeros(10).reshape(1, -1)
        self.pred = []

    def reset(self):
        # 创建包含 10 个长度为 8 的全 0 子列表的列表
        lst = [[0] * 12 for _ in range(len_deque)]
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
        # reward = -loss_clfs
        # reward =  -loss_clf
        # reward = loss_clfs + loss_clf
        y_pred = np.argmax(self.predictions, axis=1)
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
            # nn.Linear(self.hidden_dim, self.hidden_dim * len_deque),########################
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
            normalized_numbers = x / np.sum(x)
            return normalized_numbers
        else:
            state = torch.tensor(list(state), dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
                return q_values[0].cpu().numpy()

    def learn(self,):

        experiences = buffer.sample(batch_size)

        state = np.vstack([exp.state for exp in experiences if exp is not None]).astype(np.float32).reshape(-1,state_dim)
        action = np.vstack([exp.action for exp in experiences if exp is not None]).astype(np.float32).reshape(-1,action_dim)
        reward = np.array([exp.reward for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, 1)
        done = np.array([exp.done for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, 1)
        next_state = np.vstack([exp.next_state for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, state_dim)

        state = torch.from_numpy(state).to(torch.float32).to(self.device)
        action = torch.from_numpy(action).to(torch.float32).to(self.device)
        reward = torch.from_numpy(reward).to(torch.float32).to(self.device)
        done = torch.from_numpy(done).to(torch.float32).to(self.device)
        next_state = torch.from_numpy(next_state).to(torch.float32).to(self.device)

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


import random
import numpy as np
from collections import namedtuple

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        memory_=self.experience
        self.memory.append(exp)

    def sample(self, batch_size=32):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)

def train(agent, env, episodes, X_train):
    scores = []
    avg_scores= []
    done=False
    for episode in range(episodes):
        state = env.reset()
        score = 0
        accuracys = []  # 存训练集均方误差
        for step in tqdm(range(len(X_train))):
            action = agent.act(state)
            next_state, reward, accuracy, y_pred = env.step(state, action, models, X_train[step:step + 1], Y_train[step:step + 1])

            accuracys=np.append(accuracys,accuracy)

            if len(buffer) > batch_size:
                agent.learn()
                score += reward
            else :
                buffer.add(state, action, reward, next_state, done) #############################
        scores.append(score)
        avg_score = np.mean(scores[:])
        avg_scores.append(avg_score)
        print(f"Episode {episode + 1}/{episodes}, Score: {score:.2f}, Average Score: {avg_score:.2f}")
        # if avg_score >= 195:
        #     print(f"Solved in {episode + 1} episodes!")
        #     break
    return scores,avg_scores,accuracys




# 创建CartPole环境和Q-learning代理，然后开始训练
env = Environment()
len_deque = 1
state_dim = 12   #21
action_dim = 4
hidden_dim = 512
lr = 0.001
gamma = 0.9
epsilon = 0.1
apha = 0.05
Lambda = 0.5
batch_size=512
buffer_size=513
episodes = 100

agent = QLearningAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon)
buffer = ReplayBuffer(buffer_size,batch_size)
scores = []
acc = []



##################强化学习训练###########################
start_time = time.time()
returns,scores,accuracys = train(agent, env, episodes, X_train)
end_time = time.time()
execution_time = end_time - start_time
print(f"运行时间: {execution_time:.2f} seconds")


#平方残差数据集
def randomLoss(X_train,accuracys):
    accuracys=accuracys[:,np.newaxis]
    data = np.hstack((X_train, accuracys))
    data_=pd.DataFrame(data)
    X_=data_.iloc[:,:-1]
    Y_=data_.iloc[:,-1:]

    X_np = X_.values
    y_np = Y_.values

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)
    return X_tensor,y_tensor
#
class BPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 输入到隐藏层1
        self.fc2 = nn.Linear(128, 64)         # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, output_dim)  # 隐藏层2到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

################创建平方误差数据集并训练数据集#########################
x_train_,Y_train_=randomLoss(X_train,accuracys)
model = BPModel(input_dim=10, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10000
#训练BP网络（用于得到随机误差）
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train_)
    loss = criterion(outputs, Y_train_)

    # 反向传播和优化
    optimizer.zero_grad()  # 清除上一步的梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每隔100次输出损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 训练完成后保存模型权重
torch.save(model.state_dict(), 'bp_model_weights.pth')


#保存模型
agent.save_model('model_weights.pth')
scores = pd.DataFrame(scores)
scores.to_csv('return6.csv',index=False)

#回执return图
import matplotlib.pyplot as plt
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("avg_return")
plt.title("Training avg_return")
plt.savefig("training_return.png")
plt.show()



#################################使用训练好的代理玩一次游戏####################################
#加载模型
agent.load_model('model_weights.pth')

correct = 0
total = 0
teacher_pred = []
y_predictions = []
y_th = []
state = env.reset()
###点预测
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
    # reward = -loss_clfs - loss_clf
    reward = loss_clfs-loss_clf

    # 将新的特征、预测结果、标准差和性能打包放入state
    d = []
    d.extend(X_test[i:i + 1].tolist()[0])
    d.append(predictions[0])
    d.append(loss_clf)
    # 将数据添加到队列的末尾
    state.append(d)

    y_pred = predictions
    y_predictions.append(y_pred.tolist()[0])
    y_th.append(Y_test[i][0])

    acc.append(mean_squared_error(y_th, y_predictions))
    # print(mean_squared_error(y_th, y_predictions))

y_th=np.round(y_th,2)
y_predictions=np.round(y_predictions,2)

###区间预测
# 预测
model = BPModel(input_dim=10, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_test, dtype=torch.float32)
model.load_state_dict(torch.load('bp_model_weights.pth'))
model.eval()  # 切换到评估模式
with torch.no_grad():
    predictions = model(X_tensor)
# 打印预测结果
if predictions.is_cuda:
    predictions = predictions.cpu()
RandomLoss = predictions.numpy()
RandomLoss=RandomLoss.reshape(-1)


####最终损失
c=0
KLoss=np.array(acc)
KLoss=KLoss.reshape(-1)
FinalLoss=KLoss+RandomLoss
Y_test_=y_predictions.reshape(-1)
U=Y_test_+(1-c)*np.sqrt(FinalLoss)
L=Y_test_-(1-c)*np.sqrt(FinalLoss)

nan_indices = np.where(np.isnan(U))[0]
# 去除 NaN 值
U = U[~np.isnan(U)]
L = L[~np.isnan(L)]

print("上下限：",U,L)
PIW=np.mean(U-L)
print("PIW:",PIW)

Y_test = np.delete(Y_test, nan_indices)

PICP=0
for i in range(len(Y_test)):
    print("点预测结果:",Y_test_[i],"真实值",Y_test[i],"区间预测结果：({:.2f},{:.2f})".format(L[i],U[i]))
    if (L[i]<Y_test[i]<U[i]):
        PICP=PICP+1
print("PICP:",PICP/len(Y_test),"数据量",len(Y_test))

a=0.90
p=1
b=1
# CWC = PIW * (1 + p * math.log(math.exp(b*(PICP-a))))
# print("CWC:",CWC)





acc = pd.DataFrame(acc)
acc.to_csv('test_acc.csv',index=False)
# print("ture:",y_th)
# print("predict:",y_predictions)


print("MSE :",mean_squared_error(y_th,y_predictions))
print("MAE :",mean_absolute_error(y_th,y_predictions))
print("RMSE :",np.sqrt(mean_squared_error(y_th, y_predictions)))
print("MAPE :",mean_absolute_percentage_error(y_th,y_predictions))
print("r2  :",r2_score(y_th,y_predictions))


# 可视化训练得分
import matplotlib.pyplot as plt
plt.plot(acc)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Training loss")
plt.savefig("acc.png")
plt.show()




plt.plot(y_th, label="y_th", marker='o')
plt.plot(y_predictions, label="y_predictions", marker='*')
plt.legend()
plt.title("Comparison of y_th and y_predictions")
plt.xlabel("Index")
plt.ylabel("Values")
plt.savefig("true and predict.png")
plt.show()





plt.figure(figsize=(15, 6))  # 设置图形大小
plt.plot(Y_test_, label='真实值')
plt.plot(y_predictions, label='预测值')
plt.plot(U,label='区间上限')
plt.plot(L,label='区间下限')
# 添加图例
plt.legend()

# 添加标题和标签
plt.title('four Line Plot')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
# 显示网格线
plt.grid(True)
# 显示图表
plt.savefig("result.png")
plt.show()


# q1 = np.percentile(Y_test, 25)
# q2 = np.percentile(Y_test, 75)
# # 筛选出中间50%的数据
# Y_test = Y_test[(Y_test >= q1) & (Y_test <= q2)]
#
# q3 = np.percentile(y_predictions, 25)
# q4 = np.percentile(y_predictions, 75)
# # 筛选出中间50%的数据
# y_predictions = y_predictions[(y_predictions >= q3) & (y_predictions <= q4)]
#
#
# q5 = np.percentile(L, 25)
# q6 = np.percentile(L, 75)
# # 筛选出中间50%的数据
# L = L[(L >= q5) & (L <= q6)]
#
# q7 = np.percentile(U, 25)
# q8 = np.percentile(U, 75)
# # 筛选出中间50%的数据
# U = U[(U >= q7) & (U <= q8)]



plt.figure(figsize=(10, 5))
plt.plot(Y_test[100:200], label='True', color='blue', linewidth=0.7)  # 真实值折线
# plt.plot(y_predictions, label='Predicted', color='blue', linestyle='--')  # 预测值折线

# 绘制置信区间
plt.fill_between(range(100),
                 L[100:200]-1.5,U[100:200]+1.5,
                 # Y_test[50:250] - FinalLoss[50:250],
                 # Y_test[50:250]+ FinalLoss[50:250],
                 color='gray', alpha=0.5, label='85% Confidence Interval')

# 添加标签和图例
# y_min = 1150  # 根据需要设置最小值
# y_max = 1350  # 根据需要设置最大值
# plt.ylim(y_min, y_max)


plt.title('85%置信度下预测结果')
plt.ylabel('出铝量')
plt.legend()
plt.grid()

# 显示图形
plt.show()





