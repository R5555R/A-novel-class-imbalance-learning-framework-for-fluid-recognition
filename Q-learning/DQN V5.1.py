import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  # GDBT梯度提升树
from sklearn.ensemble import ExtraTreesClassifier  # ET极限森林
from catboost import CatBoostClassifier  # CatBoost
from xgboost import XGBClassifier  # XGBoost
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor # 回归树
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
X_train, X_test, y_train, y_test = train_test_split(data[features].values, LABEL, test_size=0.2)

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
        action = q_values(state_tensor).detach().numpy()
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
                self.predictions += weights[i] * clf.predict(dtest)
            else:
                self.predictions += weights[i] * clf.predict_proba(X)
        # Compute the accuracy of the predictions
        y_true = one_hot.transform(y.reshape(-1, 1)).reshape(1, -1)
        loss_clf = log_loss(y_true, self.predictions)

        # Compute the reward
        reward = -loss_clf
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

n_clfs = len(models)
# Create environment and QNetwork
q_values = QNetwork(n_clfs)
env = Environment()
criterion = nn.MSELoss()
optimizer = optim.Adam(q_values.parameters(), lr=learning_rate)
# Train the Q-network using Q-learning algorithm
# X_train = X_train[0:10]
for epoch in range(epochs):
    state = env.reset()

    # Initialize the cumulative reward and loss
    total_reward = 0
    total_loss = 0
    accuracy = 0

    # Loop through each sample and each action
    for j in tqdm(range(len(X_train))):
        weights = choose_action(state, q_values)
        next_state, reward, accuracy = env.step(state, weights, models, X_train[j:j + 1], y_train[j:j + 1])
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

        # Compute the loss and update the Q-network
        loss = criterion(q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Print the epoch number, reward, and loss
    print(
        "Epoch: {}, Reward: {:.2f}, Loss: {:.4f}, Accuracy: {:.2f}".format(epoch, total_reward / len(X_train), total_loss / len(X_train), accuracy))

# Test the learned model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    teacher_pred = []
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
        y_true = one_hot.transform(y_train[i].reshape(-1, 1)).reshape(1, -1)
        loss_clf = log_loss(y_true, probs)

        # 将新的特征、预测结果、标准差和性能打包放入state
        d = []
        d.extend(X_train[i:i + 1].tolist()[0])
        d.extend(probs.tolist()[0])
        d.append(loss_clf)
        # 将数据添加到队列的末尾
        state.append(d)

        y_pred = np.argmax(probs, axis=1)
        if y_pred == y_train[i]:
            correct += 1
        total += 1

        teacher_pred.append(probs.tolist()[0])

    # Print the accuracy on the test set
    print("Accuracy on train set: {:.2f}%".format(correct / total * 100))


teacher_output = pd.DataFrame(teacher_pred)
teacher_output.to_csv('teacher_output.csv',index=False)

# Test the learned model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    teacher_pred = []
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
        y_true = one_hot.transform(y_train[i].reshape(-1, 1)).reshape(1, -1)
        loss_clf = log_loss(y_true, probs)

        # 将新的特征、预测结果、标准差和性能打包放入state
        d = []
        d.extend(X_train[i:i + 1].tolist()[0])
        d.extend(probs.tolist()[0])
        d.append(loss_clf)
        # 将数据添加到队列的末尾
        state.append(d)

        y_pred = np.argmax(probs, axis=1)
        if y_pred == y_train[i]:
            correct += 1
        total += 1

        teacher_pred.append(probs.tolist()[0])

    # Print the accuracy on the test set
    print("Accuracy on test set: {:.2f}%".format(correct / total * 100))


# teacher_output = pd.DataFrame(teacher_pred)
# teacher_output.to_csv('teacher_output.csv',index=False)

# 定义学生模型
model_student = KNeighborsRegressor()
# 使用教师模型中的软目标训练学生模型
model_student.fit(X_train, teacher_pred)
# 评估学生模型
y_pred_student = model_student.predict(X_test)
acc_student = accuracy_score(y_test, np.argmax(y_pred_student, axis=1))

print("评价指标-1：(KNeighborsRegressor)")
print(metrics.classification_report(y_test, np.argmax(y_pred_student, axis=1)))


# 定义学生模型
model_student = DecisionTreeRegressor()
# 使用教师模型中的软目标训练学生模型
model_student.fit(X_train, teacher_pred)
# 评估学生模型
y_pred_student = model_student.predict(X_test)
acc_student = accuracy_score(y_test, np.argmax(y_pred_student, axis=1))

print("评价指标-1：(DecisionTreeRegressor)")
print(metrics.classification_report(y_test, np.argmax(y_pred_student, axis=1)))


# 定义学生模型
model_student = RandomForestRegressor()
# 使用教师模型中的软目标训练学生模型
model_student.fit(X_train, teacher_pred)
# 评估学生模型np.argmax(np.array(probs),axis=1)
y_pred_student = model_student.predict(X_test)
acc_student = accuracy_score(y_test, np.argmax(y_pred_student, axis=1))

print("评价指标-1：(RandomForestRegressor)")
print(metrics.classification_report(y_test, np.argmax(y_pred_student, axis=1)))


# 定义软标签训练数据集
class SoftLabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, running_loss / len(train_loader)))



# 定义软标签训练数据集
train_dataset = SoftLabelDataset(X_train, teacher_pred)
test_dataset = SoftLabelDataset(X_test, y_test)
# 定义训练数据集的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# 定义神经网络模型、损失函数和优化器
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练神经网络
train(model, train_loader, criterion, optimizer, num_epochs=50)

# 定义预测函数
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
    return predictions

# 在测试集上进行预测
predictions = predict(model, test_loader)


print("评价指标-1：(BP)")
print(metrics.classification_report(y_test, predictions))

