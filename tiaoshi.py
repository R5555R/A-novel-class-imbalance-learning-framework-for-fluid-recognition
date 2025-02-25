# import numpy as np
# from catboost import CatBoostRegressor
# from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
# from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
# from xgboost import XGBRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.svm import SVR
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
#
#
#
# # from makedata import getdata
# #
#
#
#
# # import pandas as pd
# #
# # # 读取 .dat 文件
# # # 假设 .dat 文件中各列由多个空格分隔
# # dat_file = r'C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\TE\d08_te.dat'
# #
# # # 使用 pandas 读取文件，多个空格分隔符用 '\s+' 表示
# # df = pd.read_csv(dat_file, sep='\s+', engine='python')
# # df = df.head(159)
# #
# # # 将 DataFrame 保存为 .xls 文件
# # output_file = r'C:\Users\rmw\Desktop\output8.xlsx'
# # df.to_excel(output_file, index=False)
# #
# # print(f"Data has been successfully converted to {output_file}")
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
#
#
# def getdata():
#     file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\2012-6-25天泰数据\2012-6-25天泰数据\data(noise).xls"
#     # df = pd.read_excel(file_path, sheet_name=2)
#     # file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\TE\output.xlsx"
#     df = pd.read_excel(file_path)
#
#     data=df.iloc[:,3:]
#     # data = df.iloc[:, :]
#     # data.drop('B11', axis=1, inplace=True)
#
#     data.drop('出铝量', axis=1, inplace=True)
#     data.drop('波动(秒)', axis=1, inplace=True)
#     data.drop('AE等待时间', axis=1, inplace=True)
#     data.drop('电流效率', axis=1, inplace=True)
#     data.drop('单槽吨铝直流单耗', axis=1, inplace=True)
#
#     Q1 = data['实际出铝'].quantile(0.1)  # 下四分位数
#     Q3 = data['实际出铝'].quantile(0.9)  # 上四分位数
#
#     print(Q1)
#     print(Q3)
#
#     IQR = Q3 - Q1  # 四分位距
#     #
#     # 定义异常值的范围
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#
#     # 过滤掉异常值
#     data = data[(data['实际出铝'] >= lower_bound) & (data['实际出铝'] <= upper_bound)]
#
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import matplotlib.font_manager as fm
#
#     # 设置字体
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
#     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
#     # 假设 df 是包含特征和标签的数据框，'label' 是标签列的名称
#     # label = 'A19'  # 替换为你的标签列名称
#     label = '实际出铝'
#     correlation_matrix = data.corr()
#
#     # 获取特征与标签的相关性，并计算其绝对值
#     correlation_with_label = correlation_matrix[label].abs()
#
#     # 排除标签自身并排序
#     correlation_with_label = correlation_with_label.drop(label)
#     sorted_correlation = correlation_with_label.sort_values(ascending=False)
#
#     # 选择相关性绝对值最大的前十个特征
#     top_10_features = sorted_correlation.head(10)
#     top_10_features_index=sorted_correlation.head(10).index
#
#     #提取数据
#     data_ = data[top_10_features_index.tolist() + [label]]
#
#     print(data_)
#
#     # 输出结果
#     print("Top 10 features with the highest absolute correlation to the label:")
#     print(top_10_features)
#
#     X = data_.iloc[:, :-1]
#     Y = data_.iloc[:, -1]
#
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     Y_train = Y_train.reset_index(drop=True)
#     Y_test = Y_test.reset_index(drop=True)
#
#     Y_train = Y_train.to_numpy().reshape(-1, 1)
#     Y_test = Y_test.to_numpy().reshape(-1, 1)
#
#     return X_train_scaled, X_test_scaled, Y_train, Y_test
#
#
# x_train, x_test, y_train, y_test = getdata()
# X_train = x_train
# X_test = x_test
# Y_train = y_train
# Y_test = y_test
#
# # # 定义模型列表
# models = []
# estimator0 = RandomForestRegressor(n_estimators=100, random_state=42)  # XGBoost
# estimator1 = SVR(kernel='rbf', C=1.0, epsilon=0.1)          # Support Vector Machine
# estimator2 = LGBMRegressor(random_state=42)  # LightGBM
# estimator3 = CatBoostRegressor(verbose=0, random_state=42)  # CatBoost
#
#
# # 将模型加入列表并训练
# estimator = [estimator0, estimator1, estimator2, estimator3]
# for reg in estimator:
#     reg.fit(X_train, Y_train)
#     models.append(reg)
# print("RF        SVM        LightGBM     CatBoost")
# print("MSE  ",[mean_squared_error(y_test, k.predict(x_test)) for k in models])
#
# print("MAE  ",[mean_absolute_error(y_test, k.predict(x_test)) for k in models])
#
# print("RMSE ",[np.sqrt(mean_squared_error(y_test, k.predict(x_test))) for k in models])
#
# print("r2   ",[r2_score(y_test, k.predict(x_test)) for k in models])
#
# print("MAPE :",[mean_absolute_percentage_error(y_test, k.predict(x_test)) for k in models])
#
#


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 计算评价指标
from xgboost import XGBRegressor


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mae, rmse, r2, mape


# 读取训练和测试文件
train_file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\天泰300KA电解槽\train.xlsx"  # 替换为实际路径
test_file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\天泰300KA电解槽\test.xlsx"  # 替换为实际路径

# 假设我们使用第一个子表，读取数据
train_data = pd.read_excel(train_file_path, sheet_name=0)
test_data = pd.read_excel(test_file_path, sheet_name=0)

train_data=train_data.iloc[:,4:]
test_data=test_data.iloc[:,4:]

columns_to_drop = ['AE次数', "天车指示量","计划出铝量","电流效率","铸造计量","一点测定","Fe超标数量","超标分子比","超标铝水平","超标电解质水平","超标槽温","超标平均电压","黑电耗（按测试黑电压3.261计算）","整流效率","备注","修炉方式"]  # 替换为你需要删除的列名
train_data.drop(columns=columns_to_drop, axis=1, inplace=True)
test_data.drop(columns=columns_to_drop, axis=1, inplace=True)


# 指定标签字段（列名需要替换为实际列名）
label_column = "实际出铝量"  # 替换为实际标签字段名

# 相关性分析
correlation_matrix = train_data.corr()
correlations = correlation_matrix[label_column].abs().sort_values(ascending=False)

# 获取相关性最强的前10个字段（去掉标签字段本身）

top_features = correlations.index[1:11]  # 从1开始去掉标签字段
top_features_value = correlations[1:11]

print("相关性最强的字段及其相关性值：")
for feature, value in top_features_value.items():
    print(f"{feature}: {value:.4f}")

# 数据选择
train_features = train_data[top_features]
train_labels = train_data[label_column]

# 测试数据也选取相同的字段
test_features = test_data[top_features]
test_labels = test_data[label_column]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_features = scaler.transform(test_features)

y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

y_train = y_train.to_numpy().reshape(-1, 1)
y_val = y_val.to_numpy().reshape(-1, 1)


# 模型列表
models = {
    # "Random Forest": RandomForestRegressor(random_state=42),
    "SVM": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=False, random_state=42),
    # "GradientBoost":GradientBoostingRegressor(),
    # "ExtraTree":ExtraTreesRegressor(),
    "XGBoost":XGBRegressor()

}

# 训练和测试
results = {}

for model_name, model in models.items():
    print(f"正在训练模型: {model_name}")
    model.fit(X_train, y_train)

    # 验证集预测
    val_pred = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_pred)

    # 测试集预测
    test_pred = model.predict(test_features)
    test_metrics = calculate_metrics(test_labels, test_pred)

    # 保存结果
    results[model_name] = {
        "Validation Metrics": val_metrics,
        "Test Metrics": test_metrics,
    }

# 打印结果
for model_name, metrics in results.items():
    print(f"\n模型: {model_name}")
    print("验证集指标 (MSE, MAE, RMSE, R2, MAPE):", metrics["Validation Metrics"])
    print("测试集指标 (MSE, MAE, RMSE, R2, MAPE):", metrics["Test Metrics"])

