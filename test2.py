import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 读取 Excel 文件的第八个子表
file_path = r'C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\天泰300KA电解槽\天泰300KA电解槽\2021年综合数据\2021年1-3月日报表.xlsx'  # 替换为实际文件路径
sheet_name = 8  # 第八个子表（索引从 0 开始）

# 读取数据
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 指定要分析的字段列表
columns_to_analyze = ["实际出铝量","工作电压","Fe超标数量","超标分子比","超标铝水平","超标电解质水平","超标槽温","超标平均电压"]  # 替换为实际字段名

# 筛选指定字段数据
data_selected = data[columns_to_analyze]

# 处理空值和非数值内容
# 如果字段包含空值或非数值，将这些值转换为NaN并处理
data_selected = data_selected.apply(pd.to_numeric, errors='coerce')  # 将无法转换的内容替换为NaN
data_selected = data_selected.dropna()  # 删除包含NaN的行

# 检查是否还有足够数据
if data_selected.empty:
    print("数据为空，请检查输入数据是否正确！")
else:
    # 计算相关性矩阵
    correlation_matrix = data_selected.corr()

    # 绘制相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()



