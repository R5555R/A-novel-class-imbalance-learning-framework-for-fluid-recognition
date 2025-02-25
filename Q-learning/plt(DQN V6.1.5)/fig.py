import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Arial']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False

precision = pd.read_excel('./precision.xlsx')
recall = pd.read_excel('./recall.xlsx')
f1_score = pd.read_excel('./f1-score.xlsx')
cost = pd.read_excel('./cost.xlsx')

reward1 = precision['reward1']
reward2 = precision['reward2']
reward3 = precision['reward3']
reward4 = precision['reward4']

x_tick = ['5', '10', '20', '50']
labels = ['$Reward_{1}$','$Reward_{2}$','$Reward_{3}$','$Reward_{4}$']
labels = ['$R_{a}$','$R_{b}$','$R_{c}$','$R_{d}$']

fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
plt.grid(True,axis = "y",zorder=0,alpha=0.3)    #只打开y轴的网格线，横线

plt.plot(np.arange(len(reward1)), reward1, color='#F9C499', marker='P', linestyle='--', label=labels[0], markersize=10)
plt.plot(np.arange(len(reward2)), reward2, color='#C4D6A0', marker='s', linestyle='--', label=labels[1], markersize=10)
plt.plot(np.arange(len(reward3)), reward3, color='#92CDDC', marker='D', linestyle='--', label=labels[2], markersize=10)
plt.plot(np.arange(len(reward4)), reward4, color='#D9958F', marker='o', linestyle='--', label=labels[3], markersize=10)

plt.xticks(np.arange(len(x_tick)),labels=x_tick)
plt.xticks(size=15)
plt.yticks(size=15)

plt.legend(loc="best", frameon=False, fontsize=12)  # set legend location
ax.set_ylabel('Precision', color='k', fontsize=15)
ax.set_xlabel('The length of the state', color='k', fontsize=15)
plt.ylim(0.9,1.0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 调整布局
plt.tight_layout()

path = r'F:\OneDrive\Project\Logging\V3.0\Q-learning\plt(DQN V6.1.5)'+'//'+'precision.svg'
plt.savefig(path, bbox_inches='tight')
plt.show()




# 绘制召回率的图
reward1 = recall['reward1']
reward2 = recall['reward2']
reward3 = recall['reward3']
reward4 = recall['reward4']

x_tick = ['5', '10', '20', '50']
labels = ['$Reward_{1}$','$Reward_{2}$','$Reward_{3}$','$Reward_{4}$']
labels = ['$R_{a}$','$R_{b}$','$R_{c}$','$R_{d}$']

fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
plt.grid(True,axis = "y",zorder=0,alpha=0.3)    #只打开y轴的网格线，横线

plt.plot(np.arange(len(reward1)), reward1, color='#F9C499', marker='P', linestyle='--', label=labels[0], markersize=10)
plt.plot(np.arange(len(reward2)), reward2, color='#C4D6A0', marker='s', linestyle='--', label=labels[1], markersize=10)
plt.plot(np.arange(len(reward3)), reward3, color='#92CDDC', marker='D', linestyle='--', label=labels[2], markersize=10)
plt.plot(np.arange(len(reward4)), reward4, color='#D9958F', marker='o', linestyle='--', label=labels[3], markersize=10)

plt.xticks(np.arange(len(x_tick)),labels=x_tick)
plt.xticks(size=15)
plt.yticks([0.84, 0.88, 0.92, 0.96, 1.0], ['0.84', '0.88', '0.92', '0.96', '1.0'])
plt.yticks(size=15)

plt.legend(loc="best", frameon=False, fontsize=12)  # set legend location
ax.set_ylabel('Recall', color='k', fontsize=15)
ax.set_xlabel('The length of the state', color='k', fontsize=15)
plt.ylim(0.84,1.0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 调整布局
plt.tight_layout()

path = r'F:\OneDrive\Project\Logging\V3.0\Q-learning\plt(DQN V6.1.5)'+'//'+'recall.svg'
plt.savefig(path, bbox_inches='tight')
plt.show()




# 绘制f1的图
reward1 = f1_score['reward1']
reward2 = f1_score['reward2']
reward3 = f1_score['reward3']
reward4 = f1_score['reward4']

x_tick = ['5', '10', '20', '50']
labels = ['$Reward_{1}$','$Reward_{2}$','$Reward_{3}$','$Reward_{4}$']
labels = ['$R_{a}$','$R_{b}$','$R_{c}$','$R_{d}$']

fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
plt.grid(True,axis = "y",zorder=0,alpha=0.3)    #只打开y轴的网格线，横线

plt.plot(np.arange(len(reward1)), reward1, color='#F9C499', marker='P', linestyle='--', label=labels[0], markersize=10)
plt.plot(np.arange(len(reward2)), reward2, color='#C4D6A0', marker='s', linestyle='--', label=labels[1], markersize=10)
plt.plot(np.arange(len(reward3)), reward3, color='#92CDDC', marker='D', linestyle='--', label=labels[2], markersize=10)
plt.plot(np.arange(len(reward4)), reward4, color='#D9958F', marker='o', linestyle='--', label=labels[3], markersize=10)

plt.xticks(np.arange(len(x_tick)),labels=x_tick)
plt.xticks(size=15)
plt.yticks([0.84, 0.88, 0.92, 0.96, 1.0], ['0.84', '0.88', '0.92', '0.96', '1.0'])
plt.yticks(size=15)

plt.legend(loc="best", frameon=False, fontsize=12)  # set legend location
ax.set_ylabel('F1-score', color='k', fontsize=15)
ax.set_xlabel('The length of the state', color='k', fontsize=15)
plt.ylim(0.84,1.0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 调整布局
plt.tight_layout()

path = r'F:\OneDrive\Project\Logging\V3.0\Q-learning\plt(DQN V6.1.5)'+'//'+'f1-score.svg'
plt.savefig(path, bbox_inches='tight')
plt.show()



# 绘制cost的图
reward1 = cost['reward1']
reward2 = cost['reward2']
reward3 = cost['reward3']
reward4 = cost['reward4']

x_tick = ['5', '10', '20', '50']
labels = ['$Reward_{1}$','$Reward_{2}$','$Reward_{3}$','$Reward_{4}$']
labels = ['$R_{a}$','$R_{b}$','$R_{c}$','$R_{d}$']

fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
plt.grid(True,axis = "y",zorder=0,alpha=0.3)    #只打开y轴的网格线，横线

plt.plot(np.arange(len(reward1)), reward1, color='#F9C499', marker='P', linestyle='--', label=labels[0], markersize=10)
plt.plot(np.arange(len(reward2)), reward2, color='#C4D6A0', marker='s', linestyle='--', label=labels[1], markersize=10)
plt.plot(np.arange(len(reward3)), reward3, color='#92CDDC', marker='D', linestyle='--', label=labels[2], markersize=10)
plt.plot(np.arange(len(reward4)), reward4, color='#D9958F', marker='o', linestyle='--', label=labels[3], markersize=10)

plt.xticks(np.arange(len(x_tick)),labels=x_tick)
plt.xticks(size=15)
plt.yticks([60.0, 65.0, 70.0, 75.0, 80.0], ['60.0', '65.0', '70.0', '75.0', '80.0'])
plt.yticks(size=15)

plt.legend(loc="best", frameon=False, fontsize=12)  # set legend location
ax.set_ylabel('Run Time', color='k', fontsize=15)
ax.set_xlabel('The length of the state', color='k', fontsize=15)
plt.ylim(60.0, 84.0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 调整布局
plt.tight_layout()

path = r'F:\OneDrive\Project\Logging\V3.0\Q-learning\plt(DQN V6.1.5)'+'//'+'cost.svg'
plt.savefig(path, bbox_inches='tight')
plt.show()