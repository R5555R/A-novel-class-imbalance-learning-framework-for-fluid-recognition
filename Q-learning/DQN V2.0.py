import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 加载数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
input_dim = X_train.shape[1]
num_classes = len(set(y_train))


# 定义基学习器
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# 定义Q函数网络
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_classifiers):
        super(QNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classifiers = num_classifiers
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_classifiers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class RLAlgorithm(object):
    def __init__(self, num_classifiers, learning_rate, gamma, epsilon):
        self.num_classifiers = num_classifiers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_net = QNetwork(input_dim, num_classes, num_classifiers)
        self.classifiers = nn.ModuleList([Classifier(input_dim, num_classes) for _ in range(num_classifiers)])
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        # epsilon-greedy策略
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_classifiers - 1)
        else:
            q_values = self.q_net(state)
            action = torch.argmax(q_values).item()
        return action

    def train(self, experiences):
        # 抽取经验并计算Q值
        states, actions, rewards, next_states = experiences
        q_values = self.q_net(states)
        next_q_values = self.q_net(next_states)
        target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0].unsqueeze(1)
        target_q_values = target_q_values.detach()
        expected_q_values = q_values.gather(1, actions)

        # 计算TD误差并更新Q网络
        loss = nn.functional.smooth_l1_loss(expected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_classifiers(self, experiences):
        # 抽取经验并更新每个基学习器的权重
        states, actions, rewards, next_states = experiences
        q_values = self.q_net(states)
        classifier_probs = torch.stack([c(state) for c, state in zip(self.classifiers, states)])
        classifier_probs = classifier_probs.detach()
        action_probs = classifier_probs.gather(1, actions.unsqueeze(1).unsqueeze(2).repeat(1, 1, num_classes))
        action_probs = action_probs.squeeze(1)
        log_probs = torch.log(action_probs)
        rewards = rewards.repeat(self.num_classifiers, 1).transpose(0, 1)
        advantage = rewards - q_values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        policy_loss = (-log_probs * advantage.detach()).mean()
        entropy_loss = (-classifier_probs * torch.log(classifier_probs + 1e-8)).sum(dim=1).mean()
        loss = policy_loss - 0.001 * entropy_loss
        for c in self.classifiers:
            c.zero_grad()
        loss.backward()
        for c in self.classifiers:
            c.optimizer.step()


# 训练过程
num_episodes = 500
max_steps = 200
batch_size = 32
rl_algo = RLAlgorithm(num_classifiers=6, learning_rate=0.001, gamma=0.99, epsilon=0.1)

for episode in range(num_episodes):
    state = torch.from_numpy(X_train).float()
    state = state.unsqueeze(0)
    done = False
    step = 0
    total_reward = 0
    while not done and step < max_steps:
        action = rl_algo.select_action(state)
        next_state = torch.from_numpy(X_train).float()
        next_state = next_state.unsqueeze(0)
        reward = -1 * np.abs(y_train - action)
        total_reward += reward
        done = (total_reward >= 0)
        rl_algo.train((state, torch.tensor([action]), torch.tensor([reward]), next_state))
        state = next_state
        step += 1
    print("Episode {}: Total reward = {}".format(episode, total_reward))

# 测试过程
rl_algo.q_net.eval()
with torch.no_grad():
    state = torch.from_numpy(X_test).float()
    q_values = rl_algo.q_net(state)
    action = torch.argmax(q_values, dim=1).numpy()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, action)
print("Accuracy:", accuracy)
