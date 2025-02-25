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

pre = precision['reward4']
rec = recall['reward4']
f1 = f1_score['reward4']
run_time = cost['reward4']

x_tick = ['5', '10', '20', '50']

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
plt.grid(axis="y",zorder=0,alpha=0.3)    #只打开y轴的网格线，横线
width = 0.2
bar1 = ax.bar(np.arange(len(pre))-width, pre, width=width, tick_label=x_tick, label="Precision",color='#D9958F',edgecolor='black', zorder=100,linewidth=0.2)
bar2 = ax.bar(np.arange(len(rec)), rec, width=width, tick_label=x_tick, label="Recall",color='#92CDDC',edgecolor='black', zorder=100,linewidth=0.2)
bar3 = ax.bar(np.arange(len(f1)) + width, f1, width=width, tick_label=x_tick, label="F1-score",color='#F9C499',edgecolor='black', zorder=100,linewidth=0.2)
ax.set_ylim(0.85,1.1)
ax.tick_params(axis='y', labelsize=15)
# 创建第二个 Y 轴并绘制折线图
ax1 = ax.twinx()
line = ax1.plot(np.arange(len(run_time)), run_time, color='k', marker='o',markersize=10,linestyle='--',label="Runnning time")
ax1.set_ylim(55,85)
ax1.tick_params(axis='y', labelsize=15)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
labels = ['Precision', 'Y2']

# 在 plt.legend() 中添加柱状图和折线图的标签
plt.legend([bar1, bar2, bar3, line[0]],
           ['Precision', 'Recall', 'F1-score', 'Running time'],
           loc='best', fontsize=12)


ax.set_ylabel('Value(%)', color='k', fontsize=15)
ax1.set_ylabel('Running Time(s)', color='k', fontsize=15)
ax.set_xlabel('The length of the state', color='k', fontsize=15)

ax.tick_params(axis='x', labelsize=15)
plt.xticks(np.arange(len(x_tick)),x_tick, size=15)
plt.yticks(size=15)
path = r'F:\OneDrive\Project\Logging\V3.0\Q-learning\plt(DQN V6.1.5)\双y轴图.pdf'
plt.savefig(path, bbox_inches='tight',dpi=200)
plt.tight_layout()
plt.show()