import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Arial']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False

path = './绘图.xlsx'
df = pd.read_excel(path)

reward = df['reward']
precision = df['precision']
recall = df['recall']
f1_score = df['f1-score']
x_tick = ['$Reward_{1}$','$Reward_{2}$','$Reward_{3}$','$Reward_{4}$']
x_tick = ['$r_{t}^a$','$r_{t}^b$','$r_{t}^c$','$r_{t}^d$']

list_name = ['Precision', 'Recall', 'F1-score', 'Score']
list_metrics = [precision, recall, f1_score, reward]
for i, metrics in enumerate(list_metrics):

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    # plt.grid(b = "True",axis = "y",zorder=0,alpha=0.3)    #只打开y轴的网格线，横线
    colors = ['#D9958F', '#C4D6A0', '#92CDDC', '#F9C499']
    ax.bar(np.arange(len(metrics)), metrics, tick_label=x_tick, width=0.5, color=colors)


    ax.set_ylabel(list_name[i], color='k', fontsize=15)
    ax.set_xlabel('Reward Function', color='k', fontsize=15)
    # plt.xticks(size=15, rotation=45)
    plt.xticks(size=15)
    plt.yticks(size=15)
    if i == 3:
        # plt.ylim(0.8, 1.0)
        # plt.yticks([0.80, 0.85, 0.90, 0.95, 1.0], ['0.80', '0.85', '0.90', '0.95', '1.0'])
        pass
    else:
        plt.ylim(0.8, 1.0)
        plt.yticks([0.80, 0.85, 0.90, 0.95, 1.0], ['0.80', '0.85', '0.90', '0.95', '1.0'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 调整布局
    plt.tight_layout()
    ax.legend(loc='upper left',ncol=2,fontsize='large', frameon=False)

    path = r'F:\OneDrive\Project\Logging\V3.0\Q-learning\plt(DQN V6.1.4)'+'//'+list_name[i]+'.pdf'
    plt.savefig(path, bbox_inches='tight')
    plt.show()


