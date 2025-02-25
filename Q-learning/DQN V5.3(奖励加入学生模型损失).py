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
import sklearn.metrics as metrics
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
x_train, x_test, y_train, y_test = train_test_split(data[features].values, LABEL, test_size=0.2)

# 制作顺序数据集
X_train = train_data[features].values
X_test = test_data[features].values
Y_train = le.fit_transform(train_data['LABEL'])
Y_test= le.fit_transform(test_data['LABEL'])

# Define hyperparameters
epsilon = 0.2
learning_rate = 0.01
discount_factor = 0.99
epochs = 1

one_hot = OneHotEncoder(sparse=False)
one_hot.fit(y_train.reshape(-1, 1))


def choose_action(state, q_values):
    # Use epsilon-greedy policy to choose an action
    action = []
    if np.random.uniform(0, 1) < epsilon:
        # Choose a random action
        action = np.random.rand(5)
    else:
        # Choose the action with the highest Q-value
        state_tensor = torch.tensor(list(state), dtype=torch.float).unsqueeze(0)
        action = q_values(state_tensor).detach().numpy()  # 前向推理
        # action = np.argmax(action_values)

    # Compute the model weights for the chosen action
    # weights = np.zeros(n_clfs)
    # weights[action] = 1.0

    return action


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
    def __init__(self):
        self.predictions = np.zeros(6).reshape(1, -1)
        self.pred = []

    def reset(self):
        # 创建包含 10 个长度为 18 的全 0 子列表的列表
        lst = [[0] * 18 for _ in range(10)]
        return deque(lst, maxlen=10)

    def step(self, state, weights, model, X, y):
        # Compute the predictions using the weighted model
        self.predictions = np.zeros(6).reshape(1, -1)
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
        y_true = one_hot.transform(y.reshape(-1, 1)).reshape(1, -1)
        loss_clf = [log_loss(y_true, _) for _ in self.pred]
        loss_clf = sum(loss_clf)/len(loss_clf)

        # Compute the accuracy of the predictions
        loss_clfs = log_loss(y_true, self.predictions)

        # Compute the reward
        reward = -loss_clfs-loss_clf
        accuracy = accuracy_score(y, np.argmax(self.predictions, axis=1))

        # 将新的特征、预测结果、标准差和性能打包放入state
        d = []
        d.extend(X.tolist()[0])
        d.extend(self.predictions.tolist()[0])
        d.append(loss_clf)
        # 将数据添加到队列的末尾
        state.append(d)
        return state, reward, accuracy


class QNetwork(nn.Module):
    def __init__(self, n_clfs):
        super(QNetwork, self).__init__()
        self.n_clfs = n_clfs
        self.fc1 = nn.Linear(180, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, n_clfs)

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(x)
        print(x)
        return x


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

n_clfs = len(models)
# Create environment and QNetwork
q_values = QNetwork(n_clfs)
env = Environment()
criterion = nn.MSELoss()
optimizer = optim.Adam(q_values.parameters(), lr=learning_rate)
# Train the Q-network using Q-learning algorithm
for epoch in range(epochs):
    state = env.reset()  # deque:10*18 0

    # Initialize the cumulative reward and loss
    reward = 0
    total_reward = 0
    total_loss = 0
    accuracy = 0

    # Loop through each sample and each action
    for j in tqdm(range(len(X_train))):
        weights = choose_action(state, q_values)
        next_state, reward, accuracy = env.step(state, weights, models, X_train[j:j + 1], Y_train[j:j + 1])
        total_reward += reward

        # Compute the target Q-value using the Bellman equation
        next_q_values = q_values(torch.tensor(next_state, dtype=torch.float).unsqueeze(0))
        max_next_q_value = torch.max(next_q_values)
        target_q_value = reward + discount_factor * max_next_q_value

        # Compute the current Q-value for the chosen action
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action_tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(0)
        q_values.train()
        q_value = torch.sum(q_values(state_tensor) * action_tensor)

        # Compute the loss and update the Q-network-
        loss = criterion(q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if j % 100 == 0:
            print()
            print('reward',reward)
            print('loss',loss)

    # Print the epoch number, reward, and loss
    print(
        "Epoch: {}, Reward: {:.2f}, Loss: {:.4f}, Accuracy: {:.2f}".format(epoch, total_reward / len(X_train), total_loss / len(X_train), accuracy))

# Test the learned model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    teacher_pred = []
    y_predictions = []
    state = env.reset()
    for i in tqdm(range(len(X_train))):
        weights = choose_action(state, q_values)
        probs = np.zeros(6).reshape(1, -1)
        for j, clf in enumerate(models):
            if j == 4:
                dtest = xgb.DMatrix(X_train[i].reshape(1, -1))
                probs += weights[j] * clf.predict(dtest)
            else:
                probs += weights[j] * clf.predict_proba(X_train[i].reshape(1, -1))

        # Compute the accuracy of the predictions
        y_true = one_hot.transform(Y_train[i].reshape(-1, 1)).reshape(1, -1)
        loss_clf = log_loss(y_true, probs)

        # 将新的特征、预测结果、标准差和性能打包放入state
        d = []
        d.extend(X_train[i:i + 1].tolist()[0])
        d.extend(probs.tolist()[0])
        d.append(loss_clf)
        # 将数据添加到队列的末尾
        state.append(d)

        y_pred = np.argmax(probs, axis=1)
        if y_pred == Y_train[i]:
            correct += 1
        total += 1

        y_predictions.append(y_pred)
        teacher_pred.append(probs)

    # Print the accuracy on the test set
    print(y_predictions)
    print("评价指标-1：(X_train)")
    print(metrics.classification_report(Y_train, y_predictions))
    # print("Accuracy on train set: {:.2f}%".format(correct / total * 100))

# Test the learned model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    teacher_pred = []
    y_predictions = []
    state = env.reset()
    for i in tqdm(range(len(X_test))):
        weights = choose_action(state, q_values)
        probs = np.zeros(6).reshape(1, -1)
        for j, clf in enumerate(models):
            if j == 4:
                dtest = xgb.DMatrix(X_test[i].reshape(1, -1))
                probs += weights[j] * clf.predict(dtest)
            else:
                probs += weights[j] * clf.predict_proba(X_test[i].reshape(1, -1))

        # Compute the accuracy of the predictions
        y_true = one_hot.transform(y_train[i].reshape(-1, 1)).reshape(1, -1)
        loss_clf = log_loss(y_true, probs)

        # 将新的特征、预测结果、标准差和性能打包放入state
        d = []
        d.extend(X_test[i:i + 1].tolist()[0])
        d.extend(probs.tolist()[0])
        d.append(loss_clf)
        # 将数据添加到队列的末尾
        state.append(d)

        y_pred = np.argmax(probs, axis=1)
        if y_pred == Y_test[i]:
            correct += 1
        total += 1

        y_predictions.append(y_pred)
        teacher_pred.append(probs.tolist()[0])

    # Print the accuracy on the test set
    print("评价指标-1：(X_test)")
    print(metrics.classification_report(Y_test, y_predictions))
    # print("Accuracy on test set: {:.2f}%".format(correct / total * 100))


