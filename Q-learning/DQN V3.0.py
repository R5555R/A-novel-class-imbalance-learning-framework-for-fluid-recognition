import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define state space and action space
STATE_DIM = 4
ACTION_DIM = 6

# Define Q-learning parameters
BATCH_SIZE = 32
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1000
LR = 0.001
EPOCHS = 50

# Define neural network model
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 64)
        self.fc2 = nn.Linear(64, ACTION_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define Q-learning algorithm
class QLearning:
    def __init__(self):
        self.q_net = QNet()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = []
        self.steps = 0
        self.epsilon = EPSILON_START

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(ACTION_DIM)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        self.steps += 1
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1 * self.steps / EPSILON_DECAY)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = np.array(self.memory)[np.random.choice(len(self.memory), BATCH_SIZE, replace=False), :]
        state_batch = torch.tensor(batch[:, 0].tolist(), dtype=torch.float32)
        action_batch = torch.tensor(batch[:, 1].tolist(), dtype=torch.int64)
        reward_batch = torch.tensor(batch[:, 2].tolist(), dtype=torch.float32)
        next_state_batch = torch.tensor(batch[:, 3].tolist(), dtype=torch.float32)

        q_values = self.q_net(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        next_q_values = self.q_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + GAMMA * next_q_values

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Train Q-learning algorithm
q_learning = QLearning()

for epoch in range(EPOCHS):
    for i in range(len(X_train)):
        state = X_train[i]
        action = q_learning.act(state)
        next_state = state
        reward = int(y_train[i] == action)

        q_learning.remember(state, action, reward, next_state)
        q_learning.learn()

    # Test the model after each epoch
    q_learning.q_net.eval()
    y_pred = []
    for state in X_test: 
        state = torch.tensor(state, dtype=torch.float32)
        q_values = q_learning.q_net(state)
        action = torch.argmax(q_values).item()
        y_pred.append(action)
    accuracy = accuracy_score(y_test, y_pred)
    print("Epoch {}/{}: Test accuracy = {:.3f}".format(epoch+1, EPOCHS, accuracy))
    q_learning.q_net.train()
