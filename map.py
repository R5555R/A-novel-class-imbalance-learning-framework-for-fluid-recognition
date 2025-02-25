import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
warnings.filterwarnings(action='ignore')
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取Excel文件
file_path = r'C:\Users\rmw\Desktop\附件一（训练集）.xlsx'  # 替换为您的文件路径
xls = pd.read_excel(file_path, sheet_name=None)

# 将所有子表按行合并
df_list = []
for sheet_name, df in xls.items():
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# selected_columns = df[['温度，oC', '励磁波形', '磁芯材料', '磁芯损耗，w/m3']]

label = '磁芯损耗，w/m3'  # 替换为你的标签列名称

corr_matrix = df.corr()

correlation_with_label = corr_matrix[label].abs()

# 排除标签自身并排序
correlation_with_label = correlation_with_label.drop(label)
sorted_correlation = correlation_with_label.sort_values(ascending=False)

# 选择相关性绝对值最大的前十个特征
top_10_features = sorted_correlation.head(50)
top_10_features_index = sorted_correlation.head(50).index

# top_10_features = sorted_correlation.head(15)
# top_10_features_index = sorted_correlation.head(15).index

# 提取数据
data_ = df[top_10_features_index.tolist() + [label]]





# # 4. 绘制热力图
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr_matrix, annot=True, cmap='viridis', linewidths=0.5)
# plt.title('指定字段的热力图')
# plt.show()

# 可视化前十个特征与标签的相关性
plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh')
plt.title("Top 10 Features with Highest Absolute Correlation to Label")
plt.xlabel("Absolute Correlation")
plt.ylabel("Feature")
plt.show()


file_path = r'C:\Users\rmw\Desktop\附件一（训练集）.xlsx'  # 替换为您的文件路径
df = pd.read_excel(file_path)

from scipy import stats

# 假设 "励磁波形" 是离散变量，"磁芯损耗" 是连续变量
F, p = stats.f_oneway(df[df['温度，oC'] == 1]['磁芯损耗，w/m3'],
                      df[df['温度，oC'] == 2]['磁芯损耗，w/m3'],
                      df[df['温度，oC'] == 3]['磁芯损耗，w/m3'])

print(f"F值: {F}, p值: {p}")


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 假设 df 是已经存在的 DataFrame
plt.figure(figsize=(10, 6))

# 绘制箱线图
ax = sns.boxplot(x='励磁波形', y='磁芯损耗，w/m3', data=df)

# 计算并标注相关统计数值
for i in range(len(df['励磁波形'].unique())):
    # 获取当前箱体的数据
    data = df[df['励磁波形'] == df['励磁波形'].unique()[i]]['磁芯损耗，w/m3']

    # 计算四分位数和中位数
    q1 = np.percentile(data, 25)
    q2 = np.median(data)
    q3 = np.percentile(data, 75)

    # 获取当前箱体的 x 坐标
    x = i

    # 在箱线图上标注
    ax.text(x, q1, f'Q1: {q1:.2f}', horizontalalignment='center', color='black')
    ax.text(x, q2, f'Median: {q2:.2f}', horizontalalignment='center', color='red')
    ax.text(x, q3, f'Q3: {q3:.2f}', horizontalalignment='center', color='black')

# 添加标题和标签
plt.title('励磁波形对磁芯损耗的影响')
plt.ylabel('磁芯损耗')
plt.xlabel('励磁波形')

# 显示图表
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取Excel文件
file_path = r'C:\Users\rmw\Desktop\附件一（训练集）.xlsx'  # 替换为您的文件路径
df = pd.read_excel(file_path)

# 2. 确保温度字段和磁芯损耗字段的名称正确
# 假设温度字段名为 '温度，oC'，磁芯损耗字段名为 '磁芯损耗，w/m3'
temperature_column = '温度，oC'
loss_column = '磁芯损耗，w/m3'

# 3. 根据温度字段进行分组
grouped = df.groupby(temperature_column)

# 4. 为每个组绘制独立的磁芯损耗折线图
for temperature, group in grouped:
    plt.figure(figsize=(10, 6))
    plt.plot(group.index, group[loss_column], marker='o', label=f'温度: {temperature} °C')

    # 添加图形细节
    plt.title(f'温度: {temperature} °C 下的磁芯损耗')
    plt.xlabel('样本索引')
    plt.ylabel('磁芯损耗 (w/m3)')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # 保存图形为文件（可选）
    plt.savefig(f'磁芯损耗_温度_{temperature}.png')  # 将图形保存为 PNG 文件

    # 显示图形
    plt.show()







