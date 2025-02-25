import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# 读取 xls 文件
def makedata():
    # 读取训练和测试文件
    train_file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\天泰300KA电解槽\train.xlsx"  # 替换为实际路径
    test_file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\天泰300KA电解槽\test.xlsx"  # 替换为实际路径

    # 假设我们使用第一个子表，读取数据
    train_data = pd.read_excel(train_file_path, sheet_name=0)
    test_data = pd.read_excel(test_file_path, sheet_name=0)

    train_data = train_data.iloc[:, 4:]
    test_data = test_data.iloc[:, 4:]

    columns_to_drop = ['AE次数', "天车指示量", "计划出铝量", "电流效率", "铸造计量", "一点测定", "Fe超标数量", "超标分子比", "超标铝水平", "超标电解质水平", "超标槽温",
                       "超标平均电压", "黑电耗（按测试黑电压3.261计算）", "整流效率", "备注", "修炉方式"]  # 替换为你需要删除的列名
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
    X_train, X_test, Y_train, Y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    test_features = scaler.fit_transform(test_features)

    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    test_labels = test_labels.reset_index(drop=True)

    Y_train = Y_train.to_numpy().reshape(-1, 1)
    Y_test = Y_test.to_numpy().reshape(-1, 1)
    test_labels = test_labels.to_numpy().reshape(-1, 1)

    return X_train_scaled, X_test_scaled, Y_train, Y_test


def getdata():
    # file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\2012-6-25天泰数据\2012-6-25天泰数据\2011年1-8月报表数据.xls"
    # df = pd.read_excel(file_path, sheet_name=2)
    file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\2012-6-25天泰数据\2012-6-25天泰数据\data(noise).xls"
    # file_path = r"C:\Users\rmw\Desktop\新建文件夹\destination\铝电解\数据集\TE\output.xlsx"
    df = pd.read_excel(file_path)

    data=df.iloc[:,3:]
    # # data = df.iloc[:, :]
    data.drop('出铝量', axis=1, inplace=True)
    data.drop('波动(秒)', axis=1, inplace=True)
    data.drop('AE等待时间', axis=1, inplace=True)
    data.drop('电流效率', axis=1, inplace=True)
    data.drop('单槽吨铝直流单耗', axis=1, inplace=True)

    #使用箱线图的上下四分位数（IQR）方法来检测和删除异常值
    Q1 = data['实际出铝'].quantile(0.1)  # 下四分位数
    Q3 = data['实际出铝'].quantile(0.9)  # 上四分位数

    print(Q1)
    print(Q3)

    IQR = Q3 - Q1  # 四分位距
    #
    # 定义异常值的范围
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 过滤掉异常值
    data = data[(data['实际出铝'] >= lower_bound) & (data['实际出铝'] <= upper_bound)]


    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 假设 df 是包含特征和标签的数据框，'label' 是标签列的名称
    label = '实际出铝'  # 替换为你的标签列名称
    # label = 'A19'  # 替换为你的标签列名称
    correlation_matrix = data.corr()

    # 获取特征与标签的相关性，并计算其绝对值
    correlation_with_label = correlation_matrix[label].abs()

    # 排除标签自身并排序
    correlation_with_label = correlation_with_label.drop(label)
    sorted_correlation = correlation_with_label.sort_values(ascending=False)

    # 选择相关性绝对值最大的前十个特征
    top_10_features = sorted_correlation.head(10)
    top_10_features_index=sorted_correlation.head(10).index

    # top_10_features = sorted_correlation.head(15)
    # top_10_features_index = sorted_correlation.head(15).index

    #提取数据
    data_ = data[top_10_features_index.tolist() + [label]]

    # print(data_)
    #
    # # 输出结果
    print("Top 10 features with the highest absolute correlation to the label:")
    print(top_10_features)
    #
    #
    # # 热力图可视化相关性
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
    #             xticklabels=correlation_matrix.columns,
    #             yticklabels=correlation_matrix.columns)
    # plt.title("特征相关性热力图")  # 中文标题
    # plt.show()
    #
    #
    # # 可视化前十个特征与标签的相关性
    # plt.figure(figsize=(10, 6))
    # top_10_features.plot(kind='barh')
    # plt.title("Top 10 Features with Highest Absolute Correlation to Label")
    # plt.xlabel("Absolute Correlation")
    # plt.ylabel("Feature")
    # plt.show()


    X = data_.iloc[:, :-1]
    Y = data_.iloc[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    Y_train = Y_train.to_numpy().reshape(-1, 1)
    Y_test = Y_test.to_numpy().reshape(-1, 1)

    return X_train_scaled, X_test_scaled, Y_train, Y_test

# X_train_scaled, X_test_scaled, Y_train, Y_test=getdata()
print(0)

