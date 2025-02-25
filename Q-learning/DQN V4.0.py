import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class Environment(object):
    def __init__(self, X, y, models):
        self.X = X
        self.y = y
        self.models = models
        self.observation_space = self.X.shape[1]
        self.action_space = len(self.models)
        self.n_models = len(models)

    def reset(self):
        self.idx = 0
        return self.X[self.idx]

    def step(self, action, X_train, y_train):
        y_pred = np.zeros((X_train.shape[0], self.n_models))
        for i, model in enumerate(self.models):
            y_pred[:, i] = model.predict(X_train)
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights)
        y_pred_ens = np.dot(y_pred, weights)
        print(y_pred_ens)
        reward = accuracy_score(y_train, y_pred_ens)
        done = self.idx == len(X_train) - 1
        self.idx += 1
        next_state = self.X[self.idx]
        return next_state, reward, done


def main():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train decision tree models
    models = []
    for i in range(6):
        model = DecisionTreeClassifier(max_depth=2)
        model.fit(X_train, y_train)
        models.append(model)

    # Create environment and agent
    env = Environment(X_train, y_train, models)
    agent = Agent(env.observation_space, env.action_space)

    # Train agent using Q-learning
    num_episodes = 1000
    for i in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            for j in range(len(X_train)):
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.train(state, action, next_state, reward, done)
                state = next_state



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


if __name__ == '__main__':
    main()

