import random
from collections import deque
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model:
    def __init__(self, name, model, accuracy):
        self.name = name
        self.model = model
        self.accuracy = accuracy


class Environment:
    def __init__(self, models, X, y):
        self.models = models
        self.observation_space = len(models)
        self.action_space = 6
        self.current_observation = np.zeros(self.observation_space)
        self.done = False
        self.X = X
        self.y = y

    def reset(self):
        self.done = False
        self.current_observation = np.zeros(self.observation_space)

    def step(self, action):
        selected_models = np.random.choice(self.observation_space, action, replace=False)
        selected_models = [self.models[idx] for idx in selected_models]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        predictions = []
        for model in selected_models:
            model.model.fit(X_train, y_train)
            # y_pred = model.model.predict(X_test)
            # predictions.append(y_pred.reshape(-1,))
            y_pred = model.model.predict_proba(X_test)
            y_pred += y_pred
            predictions.append(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        # y_pred = np.mean(predictions, axis=0)
        accuracy = accuracy_score(y_test, y_pred)
        reward = accuracy - 0.5
        self.current_observation = np.array([model.accuracy for model in self.models])
        if accuracy >= 0.9:
            self.done = True
        return self.current_observation, reward, self.done


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # state, action, reward, next_state, done = zip(*np.random.choice(self.buffer, batch_size, replace=False))
        # return state, action, reward, next_state, done
        batch = random.sample(self.buffer, batch_size)
        state = np.array([transition[0] for transition in batch])
        action = np.array([transition[1] for transition in batch])
        reward = np.array([transition[2] for transition in batch])
        next_state = np.array([transition[3] for transition in batch])
        done = np.array([transition[4] for transition in batch])
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def train(agent, environment, replay_buffer, batch_size, gamma, optimizer, loss_fn, num_episodes):
    score = []
    for episode in range(num_episodes):
        state = environment.current_observation
        total_reward = 0
        done = False
        while not done:
            action = agent.forward(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = torch.argmax(action, axis=1).item() + 1
            next_state, reward, done = environment.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) > batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
                state_batch = torch.tensor(state_batch, dtype=torch.float32)
                action_batch = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
                next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
                done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)

                q_values = agent(state_batch)
                q_values = q_values.gather(1, action_batch)
                next_q_values = agent(next_state_batch).detach()
                max_next_q_values = torch.max(next_q_values, axis=1)[0].unsqueeze(1)
                expected_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)
                loss = loss_fn(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("Episode:", episode, "Total reward:", total_reward)
        score.append(total_reward)
    print(np.mean(score))

def evaluate(agent, models, X, y):
    model_scores = np.zeros(len(models))
    for i, model in enumerate(models):
        if agent.act(0, model.accuracy) == 1:
            model.fit(X, y)
            y_pred = model.predict(X)
            model_scores[i] = accuracy_score(y, y_pred)
    return np.mean(model_scores)

def plot_learning_curve(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()



def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    models = [
        Model("Logistic Regression", LogisticRegression(solver='lbfgs', max_iter=3000), 0.8),
        Model("Decision Tree", DecisionTreeClassifier(), 0.9),
        Model("Random Forest", RandomForestClassifier(), 0.95),
        Model("Gradient Boosting", GradientBoostingClassifier(), 0.9),
        Model("SVM", SVC(probability=True), 0.85),
        Model("KNN", KNeighborsClassifier(), 0.7)
    ]
    environment = Environment(models, X, y)
    replay_buffer = ReplayBuffer(10000)
    agent = DQNAgent(environment.action_space, environment.observation_space)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()
    num_episodes = 200

    train(agent, environment, replay_buffer, 32, 0.99, optimizer, loss_fn, num_episodes)

    accuracies = []
    for i in range(100):
        accuracy = evaluate(agent, models, X, y)
        accuracies.append(accuracy)
    print("Final accuracy:", np.mean(accuracies))

    # plot_learning_curve(replay_buffer.rewards)


if __name__ == "__main__":
    main()




