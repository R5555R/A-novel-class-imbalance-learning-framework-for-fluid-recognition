import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier  # GDBT梯度提升树
from sklearn.ensemble import ExtraTreesClassifier  # ET极限森林
from catboost import CatBoostClassifier  # CatBoost
from xgboost import XGBClassifier  # XGBoost
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# # Load the Iris dataset
# iris = load_iris()
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

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
def choose_action(state, q_values):
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
    weights = np.zeros(6)
    weights[action] = 1.0
    return weights
def get_reward(model, weights, X, y):
    # Compute the predictions using the weighted model
    predictions = np.zeros(6).reshape(1, -1)
    for i, tree in enumerate(model):
        predictions += weights[i] * tree.predict(X).reshape(-1,1)
    predictions = np.round(predictions).astype(int)

    # Compute the accuracy of the predictions
    accuracy = accuracy_score(y, predictions)

    # Compute the reward
    reward = accuracy
    return reward
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

q_values = QNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(q_values.parameters(), lr=learning_rate)
# Train the Q-network using Q-learning algorithm
for epoch in range(epochs):
    # Train each decision tree on the training data
    models = []
    # for i in range(6):
    #     tree = DecisionTreeClassifier()
    #     tree.fit(X_train, y_train)
    #     models.append(tree)
    # 训练三个决策树基分类器
    estimator0 = GradientBoostingClassifier()
    estimator1 = ExtraTreesClassifier()
    estimator2 = CatBoostClassifier(verbose=False)
    estimator3 = XGBClassifier()

    estimator = [estimator0, estimator1, estimator2, estimator3]
    for clf in estimator:
        clf.fit(X_train, y_train)
        models.append(clf)
    print([accuracy_score(y_test, k.predict(X_test)) for k in models])

    # Evaluate the Q-network on the test data
    with torch.no_grad():
        q_values.eval()
        test_state_tensor = torch.tensor(X_test, dtype=torch.float)
        test_q_values = q_values(test_state_tensor)
        test_action_values = test_q_values.detach().numpy()

    # Initialize the cumulative reward and loss
    total_reward = 0
    total_loss = 0

    # Loop through each sample and each action
    for j in tqdm(range(len(X_train))):
        state = X_train[j]
        weights = choose_action(state, q_values)
        reward = get_reward(models, weights, X_train[j:j + 1], y_train[j:j + 1])
        total_reward += reward

        # Compute the target Q-value using the Bellman equation
        next_state = X_train[(j + 1) % len(X_train)]
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
        "Epoch: {}, Reward: {:.2f}, Loss: {:.4f}".format(epoch, total_reward / len(X_train), total_loss / len(X_train)))

# Test the learned model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(len(X_test)):
        state = X_test[i]
        weights = choose_action(state, q_values)
        y_pred = np.zeros((1, 6))
        for j in range(len(models)):
            y_pred += weights[j] * models[j].predict_proba(X_test[i:i+1])
        y_pred = np.argmax(y_pred)
        if y_pred == y_test[i]:
            correct += 1
        total += 1

    # Print the accuracy on the test set
    print("Accuracy on test set: {:.2f}%".format(correct / total * 100))
