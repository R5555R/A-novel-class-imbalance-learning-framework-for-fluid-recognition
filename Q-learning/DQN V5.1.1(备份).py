import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  # GDBT梯度提升树
from sklearn.ensemble import ExtraTreesClassifier  # ET极限森林
from catboost import CatBoostClassifier  # CatBoost
from xgboost import XGBClassifier  # XGBoost
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from collections import deque
from tqdm import tqdm

# load data
train_data = pd.read_csv('..//data//精度1//E-train-1.csv')
train_data1 = pd.read_csv('..//data//精度1//无标签126层数据（8口井）.csv')
test_data = pd.read_csv('..//data//精度1//E-test.csv')
# 合并所有数据
data = pd.concat((train_data, train_data1, test_data), axis=0)
features = ['SP', 'PE', 'GR', 'U', 'TH', 'K', 'AC', 'CNL', 'DEN', 'RLLS', 'RLLD']
train_x = data[features]
le = LabelEncoder()
LABEL = le.fit_transform(data['LABEL'])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data[features].values, LABEL, test_size=0.2)

# Define hyperparameters
epsilon = 0.1
learning_rate = 0.01
discount_factor = 0.99
epochs = 1000

one_hot = OneHotEncoder(sparse=False)
one_hot.fit(y_train.reshape(-1, 1))


def choose_action(state, q_values, n_clfs):
    # Use epsilon-greedy policy to choose an action
    if np.random.uniform(0, 1) < epsilon:
        # Choose a random action
        action = np.random.randint(6)
    else:
        # Choose the action with the highest Q-value
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action_values = q_values(state_tensor).detach().numpy()[0]
        action = np.argmax(action_values)

    # Compute the model weights for the chosen action
    weights = np.zeros(n_clfs)
    weights[action] = 1.0
    return weights


def get_reward(model, weights, X, y):
    # Compute the predictions using the weighted model
    predictions = np.zeros(6).reshape(1, -1)
    for i, clf in enumerate(model):
        if i == 4:
            dtest = xgb.DMatrix(X)
            predictions += weights[i] * clf.predict(dtest)
        else:
            predictions += weights[i] * clf.predict_proba(X)
    # Compute the accuracy of the predictions
    y_true = one_hot.transform(y).reshape(-1, )
    loss_clf = log_loss(y_true, predictions)

    # Compute the reward
    reward = -loss_clf
    return reward, predictions, loss_clf


# 定义环境函数
def get_state(state, features, probs, loss):
    # 将新的特征、预测结果、标准差和性能打包成元组
    data = (features, probs, loss)
    # 将数据添加到队列的末尾
    state.append(data)
    return state


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

    # # 在测试集上进行预测
    # dtest = xgb.DMatrix(X_test)
    # test_pred = model.predict(dtest)
    # print(accuracy_score(y_test, test_pred))
    return model


class Environment(object):
    def __init__(self, state_shape, n_clf):
        self.observation_space = state_shape
        self.action_space = n_clf
        self.predictions = np.zeros(6).reshape(1, -1)

    def reset(self):
        return deque([0 for _ in range(13)], maxlen=10)  # 初始状态

    def step(self, state, weights, model, X, y):
        # Compute the predictions using the weighted model
        for i, clf in enumerate(model):
            if i == 4:
                dtest = xgb.DMatrix(X)
                self.predictions += weights[i] * clf.predict(dtest)
            else:
                self.predictions += weights[i] * clf.predict_proba(X)
        # Compute the accuracy of the predictions
        y_true = one_hot.transform(y.reshape(-1, 1))
        loss_clf = log_loss(y_true, self.predictions)

        # Compute the reward
        reward = -loss_clf

        # 将新的特征、预测结果、标准差和性能打包成元组
        data = (features, self.predictions, loss_clf)
        # 将数据添加到队列的末尾
        state.append(data)
        return state, reward

class Agent(object):
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.uniform(size=self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.detach().numpy().flatten()

    def train(self, state, action, next_state, reward, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.FloatTensor(action).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        q_value = self.q_network(state).gather(1, action.long())
        next_q_value = self.q_network(next_state).max(1)[0].unsqueeze(1)
        target = reward + (1 - done) * self.gamma * next_q_value
        loss = nn.MSELoss()(q_value, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Train each decision tree on the training data
models = []
estimator0 = GradientBoostingClassifier()
estimator1 = ExtraTreesClassifier()
estimator2 = CatBoostClassifier(verbose=False)
estimator3 = XGBClassifier()

estimator = [estimator0, estimator1, estimator2, estimator3]
for clf in estimator:
    clf.fit(X_train, y_train)
    models.append(clf)
print([accuracy_score(y_test, k.predict(X_test)) for k in models])
models.append(imbalanced_data_model())


# Create environment and QNetwork
# q_values = QNetwork(n_clfs)
env = Environment(X_train.shape[1], len(models))
agent = Agent(env.observation_space, env.action_space)
criterion = nn.MSELoss()
# optimizer = optim.Adam(q_values.parameters(), lr=learning_rate)
# Train the Q-network using Q-learning algorithm
for epoch in range(epochs):
    state = env.reset()
    # # Evaluate the Q-network on the test data
    # with torch.no_grad():
    #     q_values.eval()
    #     test_state_tensor = torch.tensor(state, dtype=torch.float)
    #     test_q_values = q_values(test_state_tensor)
    #     test_action_values = test_q_values.detach().numpy()

    # Initialize the cumulative reward and loss
    total_reward = 0
    total_loss = 0
    done = False
    while not done:

        # Loop through each sample and each action
        for j in tqdm(range(len(X_train))):
            # weights = choose_action(state, q_values, n_clfs)
            weights = agent.get_action(state)
            # reward, predictions, loss_clf = get_reward(models, weights, X_train[j:j + 1], y_train[j:j + 1])
            next_state, reward = env.step(state, weights, models, X_train[j:j + 1], y_train[j:j + 1])
            agent.train(state, weights, next_state, reward, done)
            state = next_state
            # total_reward += reward

            # # Compute the target Q-value using the Bellman equation
            # next_q_values = q_values(torch.tensor(next_state, dtype=torch.float).unsqueeze(0))
            # max_next_q_value = torch.max(next_q_values)
            # target_q_value = reward + discount_factor * max_next_q_value
            #
            # # Compute the current Q-value for the chosen action
            # state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            # action_tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(0)
            # q_values.train()
            # q_value = torch.sum(q_values(state_tensor) * action_tensor)
            #
            # # Compute the loss and update the Q-network
            # loss = criterion(q_value, target_q_value)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # total_loss += loss.item()

    # Evaluate agent
    y_pred = np.zeros(y_test.shape)
    for i, model in enumerate(models):
        y_pred[:, i] = model.predict_proba(X_test)[:, 1]
    weights = agent.get_action(X_test[-1])
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights)
    y_pred_ens = np.dot(y_pred, weights)
    y_pred_ens = np.round(y_pred_ens)
    acc = accuracy_score(y_test, y_pred_ens)
    prec = precision_score(y_test, y_pred_ens, average='macro')
    rec = recall_score(y_test, y_pred_ens, average='macro')
    f1 = f1_score(y_test, y_pred_ens, average='macro')
    print("Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(acc, prec, rec, f1))
    # # Print the epoch number, reward, and loss
    # print(
    #     "Epoch: {}, Reward: {:.2f}, Loss: {:.4f}".format(epoch, total_reward / len(X_train), total_loss / len(X_train)))

# Test the learned model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(len(X_test)):
        state = X_test[i]
        weights = choose_action(state, q_values)
        y_pred = np.zeros((1, 6))
        for j in range(len(models)):
            y_pred += weights[j] * models[j].predict_proba(X_test[i:i + 1])
        y_pred = np.argmax(y_pred)
        if y_pred == y_test[i]:
            correct += 1
        total += 1

    # Print the accuracy on the test set
    print("Accuracy on test set: {:.2f}%".format(correct / total * 100))
