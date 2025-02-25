import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# 定义Q-learning代理
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
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        return network

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.q_network(state)
        q_value = q_values[action]

        with torch.no_grad():
            next_q_values = self.q_network(next_state)
            next_q_value = torch.max(next_q_values)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 定义训练函数
def train(agent, env, episodes, max_steps):
    scores = []
    for episode in range(episodes):
        state = env.reset()[0]
        score = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"Episode {episode + 1}/{episodes}, Score: {score:.2f}, Average Score: {avg_score:.2f}")
        if avg_score >= 195:
            print(f"Solved in {episode + 1} episodes!")
            break
    return scores


# 创建CartPole环境和Q-learning代理，然后开始训练
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128
lr = 0.001
gamma = 0.99
epsilon = 0.1
agent = QLearningAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon)
episodes = 1000
max_steps = 1000
scores = train(agent, env, episodes, max_steps)

# 可视化训练得分
import matplotlib.pyplot as plt

plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Training Scores")
plt.show()

# 使用训练好的代理玩一次游戏
state = env.reset()[0]
score = 0
while True:
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action)
    score += reward
    env.render()
    if done:
        break
print(f"Final Score: {score:.2f}")
env.close()
