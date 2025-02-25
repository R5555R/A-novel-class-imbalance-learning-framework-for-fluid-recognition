# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings(action='ignore')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# # # 读取Excel文件
# # file_path = r'C:\Users\rmw\Desktop\附件一（训练集）.xlsx'  # 替换为你的xlsx文件路径
# # df = pd.read_excel(file_path)
# #
# # # 查看"磁通密度"字段的基本信息
# # print(df['0（磁通密度B，T）'].describe())
# #
# # # 绘制"磁通密度"字段的直方图
# # plt.figure(figsize=(10,6))
# # sns.histplot(df['0（磁通密度B，T）'], bins=30, kde=True, color='blue')
# # plt.title('磁通密度分布直方图')
# # plt.xlabel('磁通密度')
# # plt.ylabel('频率')
# # plt.grid(True)
# # plt.show()
# #
# # # 绘制"磁通密度"字段的箱线图
# # plt.figure(figsize=(10,6))
# # sns.boxplot(x=df['0（磁通密度B，T）'], color='lightgreen')
# # plt.title('磁通密度箱线图')
# # plt.xlabel('磁通密度')
# # plt.grid(True)
# # plt.show()
# #
# # ##########################
# #
# # # 查看“磁力波形”字段的独特值及其计数
# # print(df['励磁波形'].value_counts())
# #
# # # 绘制“磁力波形”字段的饼图来表示各类波形的比例
# # plt.figure(figsize=(8,8))
# # df['励磁波形'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
# # plt.title('励磁波形分布')
# # plt.ylabel('')
# # plt.show()
# #
# # # 绘制“磁力波形”字段的柱状图
# # plt.figure(figsize=(8,6))
# # sns.countplot(data=df, x='励磁波形', palette='Set2')
# # plt.title('励磁波形频率分布')
# # plt.xlabel('励磁波形类型')
# # plt.ylabel('频数')
# # plt.grid(True)
# # plt.show()
# #
# #
# #
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.linear_model import LinearRegression
# # from scipy.optimize import minimize
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import StandardScaler
# # import warnings
# # warnings.filterwarnings(action='ignore')
# # from mpl_toolkits.mplot3d import Axes3D
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
# # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# #
# # # 1. 读取Excel文件
# # file_path = r'C:\Users\rmw\Desktop\附件一（训练集）.xlsx'  # 替换为您的文件路径
# # xls = pd.read_excel(file_path, sheet_name=None)
# #
# # # 将所有子表按行合并
# # df_list = []
# # for sheet_name, df in xls.items():
# #     df_list.append(df)
# #
# # # 合并所有子表为一个大的DataFrame
# # df = pd.concat(df_list, ignore_index=True)
# # # df['励磁波形'] = pd.factorize(df['励磁波形'])[0]
# # # df['磁芯材料'] = pd.factorize(df['磁芯材料'])[0]
# # # print(df.shape)
# # # features_to_standardize = ['温度，oC','励磁波形', '磁芯材料']
# # # # 打印合并后的DataFrame
# # # print(combined_df.head())
# # scaler = StandardScaler()
# # # df[features_to_standardize] = scaler.fit_transform(df[features_to_standardize])
# # # 2. 提取自变量（温度、励磁波形、磁芯材料）和因变量（磁芯损耗）
# # X = df[['温度，oC', '励磁波形', '磁芯材料']]
# # y = df['磁芯损耗，w/m3']
# #
# # # 3. 使用 PolynomialFeatures 构建二次回归模型
# # poly = PolynomialFeatures(degree=2, include_bias=False)
# # X_poly = poly.fit_transform(X)
# #
# # # 4. 线性回归拟合二次项特征
# # model = LinearRegression()
# # model.fit(X_poly, y)
# #
# # # 5. 定义损耗函数，用于最小化
# # def loss_function(variables):
# #     variables_poly = poly.transform([variables])
# #     return model.predict(variables_poly)[0]
# #
# # # 6. 优化寻找最优条件，初始猜测为数据的平均值
# # initial_guess = [df['温度，oC'].mean(), df['励磁波形'].mean(), df['磁芯材料'].mean()]
# #
# # # 设置边界：根据数据的实际范围进行设置
# # bounds = [
# #     (df['温度，oC'].min(), df['温度，oC'].max()),
# #     (df['励磁波形'].min(), df['励磁波形'].max()),
# #     (df['磁芯材料'].min(), df['磁芯材料'].max())
# # ]
# #
# # result = minimize(loss_function, initial_guess, bounds=bounds)
# #
# # # 7. 输出最优结果
# # optimal_conditions = result.x
# # min_loss = result.fun
# #
# # print(f"最优温度: {optimal_conditions[0]} °C")
# # print(f"最优励磁波形: {optimal_conditions[1]}")
# # print(f"最优磁芯材料: {optimal_conditions[2]}")
# # print(f"最小磁芯损耗: {min_loss}")
# #
# # # 8. 绘制响应面图（温度 vs 励磁波形，固定磁芯材料为最优值）
# # # temp_range = np.linspace(df['温度，oC'].min(), df['温度，oC'].max(), 50)
# # # waveform_range = np.linspace(df['励磁波形'].min(), df['励磁波形'].max(), 50)
# #
# # # temp_range = np.linspace(df['温度，oC'].min(), df['温度，oC'].max(), 50)
# # # waveform_range = np.linspace(df['磁芯材料'].min(), df['磁芯材料'].max(), 50)
# #
# # temp_range = np.linspace(df['励磁波形'].min(), df['励磁波形'].max(), 50)
# # waveform_range = np.linspace(df['磁芯材料'].min(), df['磁芯材料'].max(), 50)
# # Temp, Waveform = np.meshgrid(temp_range, waveform_range)
# #
# # # 生成网格点对应的损耗值
# # Loss = np.array([loss_function([T, W, optimal_conditions[0]]) for T, W in zip(np.ravel(Temp), np.ravel(Waveform))])
# #
# # Loss = Loss.reshape(Temp.shape)
# #
# # # 9. 绘制 3D 响应面图
# # fig = plt.figure(figsize=(10, 7))
# # ax = fig.add_subplot(111, projection='3d')
# #
# # ax.plot_surface(Temp, Waveform, Loss, cmap='viridis')
# # ax.set_xlabel('励磁波形')
# # ax.set_ylabel('磁芯材料')
# # ax.set_zlabel('磁芯损耗')
# #
# # plt.title('温度和励磁波形对磁芯损耗的影响')
# # plt.show()
# #
# #
# #
# #
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.linear_model import LinearRegression
# # from scipy.optimize import minimize
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # import warnings
# # warnings.filterwarnings(action='ignore')
# #
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
# # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# #
# # # 1. 读取Excel文件
# # file_path = r'C:\Users\rmw\Desktop\附件一（训练集）.xlsx'  # 替换为您的文件路径
# # xls = pd.read_excel(file_path, sheet_name=None)
# #
# # # 将所有子表按行合并
# # df_list = []
# # for sheet_name, df in xls.items():
# #     df_list.append(df)
# #
# # # 合并所有子表为一个大的DataFrame
# # df = pd.concat(df_list, ignore_index=True)
# #
# # # 因子化
# # df['励磁波形'] = pd.factorize(df['励磁波形'])[0]
# # df['磁芯材料'] = pd.factorize(df['磁芯材料'])[0]
# #
# # # 2. 提取自变量（温度、励磁波形、磁芯材料）和因变量（磁芯损耗）
# # X = df[['温度，oC', '励磁波形', '磁芯材料']]
# # y = df['磁芯损耗，w/m3']
# #
# # # 3. 使用 PolynomialFeatures 构建二次回归模型
# # poly = PolynomialFeatures(degree=2, include_bias=False)
# # X_poly = poly.fit_transform(X)
# #
# # # 4. 线性回归拟合二次项特征
# # model = LinearRegression()
# # model.fit(X_poly, y)
# #
# # # 5. 定义损耗函数，用于最小化
# # def loss_function(variables):
# #     variables_poly = poly.transform([variables])
# #     return model.predict(variables_poly)[0]
# #
# # # 6. 优化寻找最优条件，初始猜测为数据的平均值
# # initial_guess = [df['温度，oC'].mean(), df['励磁波形'].mean(), df['磁芯材料'].mean()]
# #
# # # 设置边界：根据数据的实际范围进行设置
# # bounds = [
# #     (df['温度，oC'].min(), df['温度，oC'].max()),
# #     (df['励磁波形'].min(), df['励磁波形'].max()),
# #     (df['磁芯材料'].min(), df['磁芯材料'].max())
# # ]
# #
# # result = minimize(loss_function, initial_guess, bounds=bounds)
# #
# # # 7. 输出最优结果
# # optimal_conditions = result.x
# # min_loss = result.fun
# #
# # print(f"最优温度: {optimal_conditions[0]} °C")
# # print(f"最优励磁波形: {optimal_conditions[1]}")
# # print(f"最优磁芯材料: {optimal_conditions[2]}")
# # print(f"最小磁芯损耗: {min_loss}")
# #
# # # 8. 绘制响应面图（温度 vs 磁芯损耗，固定励磁波形和磁芯材料）
# # temp_range = np.linspace(df['温度，oC'].min(), df['温度，oC'].max(), 50)
# # material_range = np.linspace(df['磁芯材料'].min(), df['磁芯材料'].max(), 50)
# # Temp, Material = np.meshgrid(temp_range, material_range)
# #
# # # 生成网格点对应的损耗值（固定励磁波形为最优值）
# # Loss = np.array([loss_function([T, optimal_conditions[1], M]) for T, M in zip(np.ravel(Temp), np.ravel(Material))])
# # Loss = Loss.reshape(Temp.shape)
# #
# # # 9. 绘制 3D 响应面图（温度 vs 磁芯损耗）
# # fig = plt.figure(figsize=(10, 7))
# # ax = fig.add_subplot(111, projection='3d')
# #
# # ax.plot_surface(Temp, Material, Loss, cmap='viridis')
# # ax.set_xlabel('温度 (°C)')
# # ax.set_ylabel('磁芯材料')
# # ax.set_zlabel('磁芯损耗')
# #
# # plt.title('温度和磁芯材料对磁芯损耗的影响')
# # plt.show()
# #
# #
# #
# # import seaborn as sns
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.linear_model import LinearRegression
# # from scipy.optimize import minimize
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import StandardScaler
# # import warnings
# # warnings.filterwarnings(action='ignore')
# # from mpl_toolkits.mplot3d import Axes3D
# #
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
# # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# #
# # # 1. 读取Excel文件
# # file_path = r'C:\Users\rmw\Desktop\附件一（训练集）.xlsx'  # 替换为您的文件路径
# # xls = pd.read_excel(file_path, sheet_name=None)
# # # df = pd.read_excel(file_path)
# #
# # # 将所有子表按行合并
# # df_list = []
# # for sheet_name, df in xls.items():
# #     df_list.append(df)
# #
# # # 合并所有子表为一个大的DataFrame
# # df = pd.concat(df_list, ignore_index=True)
# #
# # # 因子化
# # df['励磁波形'] = pd.factorize(df['励磁波形'])[0]
# # df['磁芯材料'] = pd.factorize(df['磁芯材料'])[0]
# #
# # # 2. 提取自变量（温度、励磁波形、磁芯材料）和因变量（磁芯损耗）
# # X = df[['温度，oC', '励磁波形', '磁芯材料']]
# # y = df['磁芯损耗，w/m3']
# #
# # # 3. 使用 PolynomialFeatures 构建二次回归模型
# # poly = PolynomialFeatures(degree=2, include_bias=False)
# # X_poly = poly.fit_transform(X)
# #
# # # 4. 线性回归拟合二次项特征
# # model = LinearRegression()
# # model.fit(X_poly, y)
# #
# # # 5. 定义损耗函数，用于最小化
# # def loss_function(variables):
# #     variables_poly = poly.transform([variables])
# #     return model.predict(variables_poly)[0]
# #
# # # 6. 优化寻找最优条件，初始猜测为数据的平均值
# # initial_guess = [df['温度，oC'].mean(), df['励磁波形'].mean(), df['磁芯材料'].mean()]
# #
# # # 设置边界：根据数据的实际范围进行设置
# # bounds = [
# #     (df['温度，oC'].min(), df['温度，oC'].max()),
# #     (df['励磁波形'].min(), df['励磁波形'].max()),
# #     (df['磁芯材料'].min(), df['磁芯材料'].max())
# # ]
# #
# # result = minimize(loss_function, initial_guess, bounds=bounds)
# #
# # # 7. 输出最优结果
# # optimal_conditions = result.x
# # min_loss = result.fun
# #
# # print(f"最优温度: {optimal_conditions[0]} °C")
# # print(f"最优励磁波形: {optimal_conditions[1]}")
# # print(f"最优磁芯材料: {optimal_conditions[2]}")
# # print(f"最小磁芯损耗: {min_loss}")
# #
# # # 8. 绘制响应面图（等高线图：励磁波形 vs 磁芯材料，固定温度为最优值）
# # temp_fixed = optimal_conditions[2]
# # waveform_range = np.linspace(df['温度，oC'].min(), df['温度，oC'].max(), 50)
# # material_range = np.linspace(df['励磁波形'].min(), df['励磁波形'].max(), 50)
# # Waveform, Material = np.meshgrid(waveform_range, material_range)
# #
# # # 生成网格点对应的损耗值
# # Loss = np.array([loss_function([temp_fixed, W, M]) for W, M in zip(np.ravel(Waveform), np.ravel(Material))])
# # Loss = Loss.reshape(Waveform.shape)
# #
# # # 9. 绘制等高线图
# # plt.figure(figsize=(8, 6))
# # contour = plt.contour(Waveform, Material, Loss, levels=20, cmap='viridis')
# # plt.clabel(contour, inline=True, fontsize=8)
# # plt.colorbar(contour)
# #
# # plt.xlabel('温度')
# # plt.ylabel('励磁波形')
# # plt.title(f'磁芯材料形固定为 {temp_fixed:.2f} °C 时，温度与励磁波形对磁芯损耗的等高线图')
# #
# # plt.show()
# #
# # # 10. 可选：继续绘制 3D 响应面图
# # fig = plt.figure(figsize=(10, 7))
# # ax = fig.add_subplot(111, projection='3d')
# #
# # ax.plot_surface(Waveform, Material, Loss, cmap='viridis')
# # ax.set_xlabel('温度')
# # ax.set_ylabel('励磁波形')
# # ax.set_zlabel('磁芯损耗')
# #
# # plt.title('温度与励磁波形对磁芯损耗的影响 (固定波形)')
# # plt.show()
# #
# #
# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# # # # 读取xlsx文件
# # # file_path = 'your_file_path.xlsx'  # 替换为您的文件路径
# # # df = pd.read_excel(file_path)
# #
# # # 绘制温度 vs 磁芯损耗的箱型图
# # plt.figure(figsize=(12, 8))
# #
# # # 使用Seaborn的箱型图，x轴为温度，y轴为磁芯损耗
# # sns.boxplot(x='励磁波形', y='磁芯损耗，w/m3', data=df)
# # # sns.violinplot(x='温度，oC', y='磁芯损耗，w/m3', data=df)
# #
# # # 添加标题和标签
# # plt.title('不同温度下的磁芯损耗箱型图')
# # plt.xlabel('励磁波形')
# # plt.ylabel('磁芯损耗 (w/m3)')
# #
# # # 显示图表
# # # plt.xticks(rotation=45)  # 如果温度种类很多，可以旋转x轴标签以免重叠
# # plt.show()
# #
# #
# #
# #
# #
# # import pandas as pd
# # from scipy.stats import pearsonr, spearmanr
# #
# # # 1. 读取Excel文件
# #
# # # 2. 选择需要分析的列，假设你想分析‘温度’、‘励磁波形’、‘磁芯材料’和‘磁芯损耗’
# # variables = ['温度，oC', '励磁波形', '磁芯材料', '磁芯损耗，w/m3']
# #
# # # 3. 皮尔逊相关性分析
# # print("皮尔逊相关性分析结果：")
# # pearson_results = df[variables].corr(method='pearson')  # 使用pandas的corr函数进行皮尔逊相关性分析
# # print(pearson_results)
# #
# # # 4. 斯皮尔曼相关性分析
# # print("\n斯皮尔曼相关性分析结果：")
# # spearman_results = df[variables].corr(method='spearman')  # 使用pandas的corr函数进行斯皮尔曼相关性分析
# # print(spearman_results)
# #
# # # 5. 逐对计算皮尔逊和斯皮尔曼相关系数 (附带p值)
# # for var1 in variables:
# #     for var2 in variables:
# #         if var1 != var2:
# #             # 计算皮尔逊相关系数和p值
# #             pearson_corr, pearson_pval = pearsonr(df[var1], df[var2])
# #             # 计算斯皮尔曼相关系数和p值
# #             spearman_corr, spearman_pval = spearmanr(df[var1], df[var2])
# #
# #             print(f"\n变量 {var1} 和 {var2} 的相关性分析：")
# #             print(f"皮尔逊相关系数: {pearson_corr:.3f}, p值: {pearson_pval:.3f}")
# #             print(f"斯皮尔曼相关系数: {spearman_corr:.3f}, p值: {spearman_pval:.3f}")
# #
# #
# #
# import matplotlib.pyplot as plt
#
# # 横坐标温度
# temperature = ["电解槽1", "电解槽2", "电解槽3"]
#
# # 纵坐标对应的数值（你可以填入实际的数值）
# values = [0.986,0.982,0.992]  # 示例数值
#
# # 创建柱状图，设置柱宽和颜色
# colors=['#00796B', '#303F9F', '#1976D2']
# plt.bar(temperature, values, width=0.65, color="lightsteelblue")  # width 参数控制柱宽, color 控制颜色
# plt.xticks(temperature)
# # 添加标题和标签
# plt.title('不同电解槽下的R2柱状图')
# plt.xlabel('电解槽')
# plt.ylabel('R2')
#
# for i, value in enumerate(values):
#     plt.text(temperature[i], value, str(value), ha='center', va='bottom')  # +5 让数值在柱子上方
# # 显示图形
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# # 模拟数据
# true_values = np.sin(np.arange(0, 170) / 10) * 50 + 100  # 真实值
# predicted_values = true_values + np.random.normal(0, 10, size=true_values.shape)  # 预测值
# std_dev = np.random.uniform(5, 10, size=true_values.shape)  # 假设的标准差
#
# # 绘制图形
# plt.figure(figsize=(10, 5))
# # plt.plot(true_values, label='True', color='black', linewidth=2)  # 真实值折线
# plt.plot(predicted_values, label='Predicted', color='blue', linestyle='-', linewidth=1)  # 预测值折线
#
# # 绘制置信区间
# plt.fill_between(range(len(predicted_values)),
#                  predicted_values - std_dev,
#                  predicted_values + std_dev,
#                  color='gray', alpha=0.5, label='80% Confidence Interval')
#
# # 添加标签和图例
# plt.title('80%置信度下预测结果')
# # plt.xlabel('测试点')
# plt.ylabel('出铝量')
# plt.legend()
# plt.grid()
#
# # 显示图形
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取CSV文件路径
# file_paths = [
#     r'E:\pychar\RL-classification\return.csv',
#     r'E:\pychar\RL-classification\return2.csv',
#     r'E:\pychar\RL-classification\return6.csv',
#     r'E:\pychar\RL-classification\return4.csv',
# ]
#
# # 用于存储归一化后的数据
# normalized_data = []
#
# # 归一化并取负值
# for file_path in file_paths:
#     data = pd.read_csv(file_path)
#     y_data = data.iloc[:, 0]  # 假设Y轴数据在第一列
#     # Min-Max归一化
#     normalized_y = (y_data - y_data.min()) / (y_data.max() - y_data.min())
#     # 取负值
#     normalized_y = -normalized_y
#     normalized_data.append(normalized_y)
#
# # 生成X轴数据
# x_data = range(len(normalized_data[0]))  # 假设每个文件数据长度相同
#
# colors = ['blue', 'wheat', 'red', 'gray']
# # 绘制折线图
# plt.figure(figsize=(12, 8))
# for i, y in enumerate(normalized_data):
#     if i == 2:  # 假设要加粗第二条曲线
#         plt.plot(x_data, y, label=f'File {i + 1}', linewidth=3.5,color=colors[i])  # 加粗
#     else:
#         plt.plot(x_data, y,label=f'File {i + 1},',linewidth=2)
#
# plt.title('Different reward function corresponding to the reward change')
# plt.xlabel('episode')
# plt.ylabel('Reward')
# # plt.legend()  # 显示图例
# plt.grid()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件路径
file_paths = [
    r'E:\pychar\RL-classification\return.csv',
    r'E:\pychar\RL-classification\return4.csv',
    r'E:\pychar\RL-classification\return6.csv',
    r'E:\pychar\RL-classification\return2.csv',


]

# 用于存储归一化后的数据
normalized_data = []

# 归一化并取负值
for file_path in file_paths:
    data = pd.read_csv(file_path)
    y_data = data.iloc[:, 0]  # 假设Y轴数据在第一列
    # Min-Max归一化
    normalized_y = (y_data - y_data.min()) / (y_data.max() - y_data.min())
    # 取负值
    normalized_y = -normalized_y
    normalized_data.append(normalized_y)

# 生成X轴数据
x_data = range(len(normalized_data[0]))  # 假设每个文件数据长度相同

colors = ['blue', 'green', 'red', 'orange']
linestyles = [ 'dotted', 'dashed','solid', 'dashdot']
# 绘制折线图
plt.figure(figsize=(12, 8))
for i, y in enumerate(normalized_data):
    if i == 2:  # 假设要加粗第二条曲线
        plt.plot(x_data, y, label=f'File {i + 1}', linewidth=3.5, color=colors[i],linestyle=linestyles[i % len(linestyles)])  # 加粗
    else:
        plt.plot(x_data, y, label=f'File {i + 1}', linewidth=2, color=colors[i],linestyle=linestyles[i % len(linestyles)])

plt.title('Different reward function corresponding to the reward change')
plt.xlabel('Episode')
plt.ylabel('Reward')
# plt.legend(prop={'size': 32})  # 显示图例
plt.grid()
plt.show()


