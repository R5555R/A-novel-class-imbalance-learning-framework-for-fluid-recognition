# 模型预测部分
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from scipy import linalg
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn import ensemble
import os
import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor
from scipy.interpolate import Rbf
import time


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
sc = MinMaxScaler(feature_range=(0, 1))


# 1、获取补全之后的数据
# 2、进行归一化
# 3、删除异常值
# 4、划分高中低
# 5、模型训练

def mydesvribe(data):
    path = '../预测地质因素(输入两个原始特征预测).xlsx'
    data = pd.read_excel(path)
    x_index = [80, 90, 45, 69, 19, 18, 15, 94, 95, 24, 97, 68, 31, 0, 13, 50, 55, 2, 26,
                     86, 12, 16, 96, 83, 75, 41, 9, 40, 76, 43, 74, 21, 57, 60, 59, 79, 30, 44,
                     98, 67, 77, 25, 32, 89, 5, 48, 92, 53, 54, 78, 28, 36, 61, 39, 72, 29, 70,
                     66, 84, 46, 33, 58, 27, 47, 63, 37, 34, 17, 42, 64, 8, 85, 6, 65, 20, 22,
                     88, 87, 91, 35, 1, 56, 93, 23, 4, 82, 14, 73, 11, 10, 81]
    data = data.iloc[x_index,:]
    data.describe().to_excel('data.xlsx')

def pearson_corr(data):
    cm = plt.cm.get_cmap('vlag')  # RdYlBu_r末尾加r表示颜色取反
    figure, ax = plt.subplots(figsize=(8,8))
    # sns.heatmap(data.iloc[:,1:].corr(), square=True, annot=True, ax=ax, cmap=sns.diverging_palette(20, 220, n=200), )
    sns.heatmap(data.iloc[:, 1:].corr(), square=True, annot=True, ax=ax, cmap=cm)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig('热力图.svg', bbox_inches ='tight')
    plt.show()

data_sc = pd.DataFrame()
# 进行归一化
def dealMaxMin():
    # path = '../重科院数据new1.xlsx'# 最新的数据
    path = './重科院数据new1（以前的坐标）.xlsx'
    data = pd.read_excel(path)
    # 获取除井号以外以及预测目标的所有列名
    # mydesvribe(data)
    data_sc = data.iloc[:, 2:10]
    print(data_sc.describe())
    # data_sc = data.iloc[:, 2:]#两个特征打开
    deal_data = sc.fit_transform(data_sc)
    deal_data = pd.DataFrame(deal_data)
    deal_data = pd.concat((deal_data, data.iloc[:, [1]]), axis=1)
    # 调整井号位置
    deal_data.insert(0, '井号', data.iloc[:, [0]])
    # deal_data.columns = ['井号','1+2小层厚度','地层压力系数','TOC','含气量','孔隙度','矿物脆性','杨氏模量','A-B高程差','1类储层钻遇长度','EUR']#两个特征关闭
    deal_data.columns = ['井号', '1+2小层厚度', '地层压力系数', 'TOC', '含气量', '孔隙度', '矿物脆性',
                         '横坐标','纵坐标','EUR']  # 两个特征关闭

    # pearson_corr(deal_data)

    return deal_data


# 产能预测模型：
# 如果文件夹不存在，则创建文件夹
def makeDirs(path):
    if not os.path.exists(path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)




def svr(x_train, y_train, x_test, y_test, well_num):
    # 进行训练
    model = SVR(kernel='rbf', gamma='auto')
    model.fit(x_train, y_train)
    pred_y = model.predict(x_test)
    pred_y = pred_y.reshape(-1, 1)

    mse = mean_squared_error(pred_y, y_test)
    # calculate RMSE 均方根误差
    rmse = math.sqrt(mean_squared_error(pred_y, y_test))
    # 平均绝对误差
    mae = mean_absolute_error(pred_y, y_test)

    mre = np.average(np.abs(pred_y - y_test) / y_test, axis=0)
    print('-----------------------------------------------')
    print('平均相对误差：%.3f' % mre)
    print('均方误差：%.6f' % round(mse, 3))
    print('均方根误差：%.6f' % round(rmse, 3))
    print('平均绝对误差：%.6f' % round(mae, 3))

    # 绘制图像
    y_test = list(y_test.flatten())
    pred_y = list(pred_y.flatten())

    # 下面是绘图部
    labels = ["{}".format(int(i[0])) for i in well_num]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    width_1 = 0.4
    ax.bar(np.arange(len(pred_y)), pred_y, width=width_1, tick_label=labels, label="预测值")
    ax.bar(np.arange(len(y_test)) + width_1, y_test, width=width_1, tick_label=labels, label="真实值")
    ax.set_ylabel('EUR', color='k')
    ax.set_xlabel('井号', color='k')
    ax.legend()
    plt.show()
    pred_y = pd.DataFrame(pred_y)
    pred_y.to_excel('pred_y.xlsx', index=False)
    return pred_y


def elman(x_train, y_train, x_test, y_test, well_num):
    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)
    # 预测测试集中的房价
    pred_y = dtr.predict(x_test)
    pred_y = pred_y.reshape(-1,1)

    mse = mean_squared_error(pred_y, y_test)
    # calculate RMSE 均方根误差
    rmse = math.sqrt(mean_squared_error(pred_y, y_test))
    # 平均绝对误差
    mae = mean_absolute_error(pred_y, y_test)

    mre = np.average(np.abs(pred_y - y_test) / y_test, axis=0)
    print('平均相对误差：%.3f' % mre)
    print('均方误差：%.6f' % round(mse, 3))
    print('均方根误差：%.6f' % round(rmse, 3))
    print('平均绝对误差：%.6f' % round(mae, 3))

    # 绘制图像
    y_test = list(y_test.flatten())
    pred_y = list(pred_y.flatten())

    # 下面是绘图部
    labels = ["{}".format(int(i[0])) for i in well_num]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    width_1 = 0.4
    ax.bar(np.arange(len(pred_y)), pred_y, width=width_1, tick_label=labels, label="预测值")
    ax.bar(np.arange(len(y_test)) + width_1, y_test, width=width_1, tick_label=labels, label="真实值")
    ax.set_ylabel('EUR', color='k')
    ax.set_ylabel('井号', color='k')
    ax.legend()
    plt.show()
    pred_y = pd.DataFrame(pred_y)
    pred_y.to_excel('pred_y.xlsx', index=False)


def rbf(x_train, y_train, x_test, y_test, well_num):
    clf = ensemble.GradientBoostingRegressor()
    gbdt_model = clf.fit(x_train, y_train)
    pred_y = gbdt_model.predict(x_test)
    pred_y = pred_y.reshape(-1,1)

    mse = mean_squared_error(pred_y, y_test)
    # calculate RMSE 均方根误差
    rmse = math.sqrt(mean_squared_error(pred_y, y_test))
    # 平均绝对误差
    mae = mean_absolute_error(pred_y, y_test)
    mre = np.average(np.abs(pred_y - y_test) / y_test, axis=0)
    print('-----------------------------------------------')
    print('平均相对误差：%.3f' % mre)
    print('均方误差：%.6f' % round(mse, 3))
    print('均方根误差：%.6f' % round(rmse, 3))
    print('平均绝对误差：%.6f' % round(mae, 3))

    # 绘制图像
    y_test = list(y_test.flatten())
    pred_y = list(pred_y.flatten())

    # 下面是绘图部
    labels = ["{}".format(int(i[0])) for i in well_num]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    width_1 = 0.4
    ax.bar(np.arange(len(pred_y)), pred_y, width=width_1, tick_label=labels, label="预测值")
    ax.bar(np.arange(len(y_test)) + width_1, y_test, width=width_1, tick_label=labels, label="真实值")
    ax.set_ylabel('EUR', color='k')
    ax.set_ylabel('井号', color='k')
    ax.legend()
    plt.show()
    pred_y = pd.DataFrame(pred_y)
    pred_y.to_excel('pred_y.xlsx', index=False)


def grad(labels, preds):
    n = preds.shape[0]
    grad = np.empty(n)
    hess = 500 * np.ones(n)
    for i in range(n):
        diff = preds[i] - labels[i]
        if diff > 0:
            grad[i] = 200
        elif diff < 0:
            grad[i] = -200
        else:
            grad[i] = 0
    return grad, hess

def custom_normal_train(label, y_pred):

    residual = (label - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * (residual) / (label + 1),
                    -10 * 2 * (residual) / (label + 1))  # 对预估里程低于实际里程的情况加大惩罚
    hess = np.where(residual < 0, 2 / (label + 1), 10 * 2 / (label + 1))  # 对预估里程低于实际里程的情况加大惩罚

    return grad, hess

def fair_obj( labels,preds):
    con = 2
    residual = preds-labels
    grad = con*residual / (abs(residual)+con)
    hess = con**2 / (abs(residual)+con)**2
    return grad,hess


def log_cosh_obj(real, predict):
    x = predict - real
    grad = np.tanh(x)
    # hess = 1 / np.cosh(x)**2 带除法的原方法，可能报ZeroDivisionException
    hess = 1.0 - np.tanh(x) ** 2
    return grad, hess

def logcoshobj( label,preds):

    d = preds - label
    grad = np.tanh(d)/label
    hess = (1.0 - grad*grad)/label
    return grad, hess

def huber_approx_obj(real, predict):
    d = predict - real
    h = 0.3  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


# @profile
def xg(x_train, y_train, x_test, y_test, well_num):
    # 回归网格化搜索最优超参数
    model = xgb.XGBRegressor(
        objective=huber_approx_obj,
        # booster="gblinear",
        subsample=0.6, colsample_bytree=0.8, random_state=0
    )
    # param_dict = {'max_depth': [2,3,4],#2
    #               'n_estimators': [10,50,100,200,300,400],#300
    #               'learning_rate':[0.05,0.1,0.2,0.3,0.4],#0.2
    #               'subsample': [0.5,0.6,0.7,0.8,0.9],#0.5
    #               'gamma': [0.1,0.2,0.3,0.4,0.5],#0.4
    #               'min_child_weight': [2,3,4,5,6,7]#7
    #               }

    # 最终实验结果的参数
    param_dict = {
                  'max_depth': [3],
                  'n_estimators': [300],
                  'learning_rate': [0.2],
                  'subsample': [0.7],
                  'gamma': [0.5],
                  'min_child_weight': [4],
                  'reg_alpha': [0.001]
                  }

    clf = GridSearchCV(model, param_dict, cv=2,verbose=1,n_jobs=-1)#accuracy
    clf.fit(x_train, y_train)
    print(clf.best_score_)
    print(clf.best_params_)

    y_pred = clf.predict(x_test)

    y_pred = y_pred.reshape(-1, 1)

    mse = mean_squared_error(y_pred, y_test)
    # calculate RMSE 均方根误差
    rmse = math.sqrt(mean_squared_error(y_pred, y_test))
    # 平均绝对误差
    mae = mean_absolute_error(y_pred, y_test)
    mre = np.average(np.abs(y_pred - y_test) / y_test, axis=0)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print('平均相对误差：%.6f' % mre)
    print('均方误差：%.6f' % round(mse, 6))
    print('均方根误差：%.6f' % round(rmse, 6))
    print('平均绝对误差：%.6f' % round(mae, 6))
    print('r2：%.6f' % round(r2, 6))
    # 绘制图像

    # y_test = y_test[[1, 2, 3, 4, 13, 16, 18]]
    # y_pred = y_pred[[1, 2, 3, 4, 13, 16, 18]]

    y_test = list(y_test.flatten())
    pred_y = list(y_pred.flatten())


    # 下面是绘图部
    labels = ["{}".format(int(i[0])) for i in well_num]
    # labels = ['9','86','7','66','94','83','74',
    #           ]
    print(labels)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    width_1 = 0.4
    ax.bar(np.arange(len(pred_y)), pred_y, width=width_1, tick_label=labels, label="预测值")
    ax.bar(np.arange(len(y_test)) + width_1, y_test, width=width_1, tick_label=labels, label="真实值")
    ax.set_ylabel('EUR', color='k')
    ax.set_xlabel('井号', color='k')
    ax.legend()
    # plt.show()
    pred_y = pd.DataFrame(pred_y)
    # pred_y.to_excel('pred_y.xlsx', index=False)
    return pred_y, mre

def xgboooo(x_train, y_train, x_test, y_test, well_num):
    # 回归网格化搜索最优超参数
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)

    mse = mean_squared_error(y_pred, y_test)
    # calculate RMSE 均方根误差
    rmse = math.sqrt(mean_squared_error(y_pred, y_test))
    # 平均绝对误差
    mae = mean_absolute_error(y_pred, y_test)
    mre = np.average(np.abs(y_pred - y_test) / y_test, axis=0)
    print('平均相对误差：%.3f' % mre)
    print('均方误差：%.6f' % round(mse, 3))
    print('均方根误差：%.6f' % round(rmse, 3))
    print('平均绝对误差：%.6f' % round(mae, 3))
    # 绘制图像

    # y_test = y_test[[1, 2, 3, 4, 13, 16, 18]]
    # y_pred = y_pred[[1, 2, 3, 4, 13, 16, 18]]

    y_test = list(y_test.flatten())
    pred_y = list(y_pred.flatten())


    # 下面是绘图部
    labels = ["{}".format(int(i[0])) for i in well_num]
    # labels = ['9','86','7','66','94','83','74',
    #           ]
    print(labels)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    width_1 = 0.4
    ax.bar(np.arange(len(pred_y)), pred_y, width=width_1, tick_label=labels, label="预测值")
    ax.bar(np.arange(len(y_test)) + width_1, y_test, width=width_1, tick_label=labels, label="真实值")
    ax.set_ylabel('EUR', color='k')
    ax.set_xlabel('井号', color='k')
    ax.legend()
    # plt.show()
    pred_y = pd.DataFrame(pred_y)
    pred_y.to_excel('pred_y.xlsx', index=False)
    return pred_y, mre

def esn(x_train, y_train, x_test, y_test, well_num):
    class ESN(object):

        def __init__(self, resSize=500, rho=0.9, cr=0.05, leaking_rate=0.2, W=None):
            """
            :param resSize: reservoir size
            :param rho: spectral radius
            :param cr: connectivity ratio
            :param leaking_rate: leaking rate
            :param W: predefined ESN reservoir
            """
            self.resSize = resSize
            self.leaking_rate = leaking_rate

            if W is None:
                # generate the ESN reservoir
                N = resSize * resSize
                W = np.random.rand(N) - 0.5
                zero_index = np.random.permutation(N)[int(N * cr * 1.0):]
                W[zero_index] = 0
                W = W.reshape((self.resSize, self.resSize))
                # Option 1 - direct scaling (quick&dirty, reservoir-specific):
                # self.W *= 0.135
                # Option 2 - normalizing and setting spectral radius (correct, slow):
                print('ESN init: Setting spectral radius...')
                rhoW = max(abs(linalg.eig(W)[0]))
                print('done.')
                W *= rho / rhoW
            else:
                assert W.shape[0] == W.shape[1] == resSize, "reservoir size mismatch"
            self.W = W

        def __init_states__(self, X, initLen, reset_state=True):

            # allocate memory for the collected states matrix
            self.S = np.zeros((len(X) - initLen, 1 + self.inSize + self.resSize))
            if reset_state:
                self.s = np.zeros(self.resSize)
            s = self.s.copy()

            # run the reservoir with the data and collect S
            for t, u in enumerate(X):
                s = (1 - self.leaking_rate) * s + self.leaking_rate * \
                    np.tanh(np.dot(self.Win, np.hstack((1, u))) + \
                            np.dot(self.W, s))
                if t >= initLen:
                    self.S[t - initLen] = np.hstack((1, u, s))
            if reset_state:
                self.s = s

        def fit(self, X, y, lmbd=1e-6, initLen=100, init_states=True):
            """
            :param X: 1- or 2-dimensional array-like, shape (t,) or (t, d), where
            :         t - length of time series, d - dimensionality.
            :param y : array-like, shape (t,). Target vector relative to X.
            :param lmbd: regularization lambda
            :param initLen: Number of samples to wash out the initial random state
            :param init_states: False allows skipping states initialization if
            :                   it was initialized before (with same X).
            :                   Useful in experiments with different targets.
            """
            assert len(X) == len(y), "input lengths mismatch."
            self.inSize = 1 if np.ndim(X) == 1 else X.shape[1]
            if init_states:
                print("ESN fit_ridge: Initializing states..."),
                self.Win = (np.random.rand(self.resSize, 1 + self.inSize) - 0.5) * 1
                self.__init_states__(X, initLen)
                print("done.")
            self.ridge = Ridge(alpha=lmbd, fit_intercept=False,
                               solver='svd', tol=1e-6)
            self.ridge.fit(self.S, y[initLen:])
            return self

        def fit_proba(self, X, y, lmbd=1e-6, initLen=100, init_states=True):
            """
            :param X: 1- or 2-dimensional array-like, shape (t,) or (t, d)
            :param y : array-like, shape (t,). Target vector relative to X.
            :param lmbd: regularization lambda
            :param initLen: Number of samples to wash out the initial random state
            :param init_states: see above
            """
            assert len(X) == len(y), "input lengths mismatch."
            self.inSize = 1 if np.ndim(X) == 1 else X.shape[1]
            if init_states:
                print("ESN fit_proba: Initializing states..."),
                self.Win = (np.random.rand(self.resSize, 1 + self.inSize) - 0.5) * 1
                self.__init_states__(X, initLen)
                print("done.")
            self.logreg = LogisticRegression(C=1 / lmbd, penalty='l2',
                                             fit_intercept=False,
                                             solver='liblinear')
            self.logreg.fit(self.S, y[initLen:].astype('int'))
            return self

        def predict(self, X, init_states=True):
            """
            :param X: 1- or 2-dimensional array-like, shape (t) or (t, d)
            :param init_states: see above
            """
            if init_states:
                # assume states initialized with training data and we continue from there.
                self.__init_states__(X, 0, reset_state=False)
            y = self.ridge.predict(self.S)
            return y

        def predict_proba(self, X, init_states=True):
            """
            :param X: 1- or 2-dimensional array-like, shape (t) or (t, d)
            :param init_states: see above
            """
            if init_states:
                # assume states initialized with training data and we continue from there.
                self.__init_states__(X, 0, reset_state=False)
            y = self.logreg.predict_proba(self.S)
            return y[:, 1]
    # 模型训练
    resSize = 50
    rho = 0.9  # spectral radius储备池谱半径SR
    cr = 0.05  # connectivity ratio
    leaking_rate = 0.2  # leaking rate
    leaking_rate = 0.05  # leaking rate
    lmbd = 1e-6  # regularization coefficient
    initLen = 49
    esn = ESN(resSize=resSize, rho=rho, cr=cr, leaking_rate=leaking_rate)
    esn.fit(x_train, y_train, initLen=initLen, lmbd=lmbd)
    # esn.fit_proba(x_train, y_train, initLen=initLen, lmbd=lmbd, init_states=False)
    pred_y = esn.predict(x_test)
    # pred_y = esn.predict_proba(x_test, init_states=False)
    pred_y = pred_y.reshape(-1, 1)
    # 保存模型
    mse = mean_squared_error(pred_y, y_test)
    # calculate RMSE 均方根误差
    rmse = math.sqrt(mean_squared_error(pred_y, y_test))
    # 平均绝对误差
    mae = mean_absolute_error(pred_y, y_test)

    print('均方误差：%.6f' % round(mse, 3))
    print('均方根误差：%.6f' % round(rmse, 3))
    print('平均绝对误差：%.6f' % round(mae, 3))

    # 绘制图像
    y_test = list(y_test.flatten())
    pred_y = list(pred_y.flatten())

    # 下面是绘图部
    labels = ["{}".format(int(i[0])) for i in well_num]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    width_1 = 0.4
    ax.bar(np.arange(len(pred_y)), pred_y, width=width_1, tick_label=labels, label="预测值")
    ax.bar(np.arange(len(y_test)) + width_1, y_test, width=width_1, tick_label=labels, label="真实值")
    ax.set_ylabel('EUR', color='k')
    ax.set_ylabel('井号', color='k')
    ax.legend()
    plt.show()
    pred_y = pd.DataFrame(pred_y)
    pred_y.to_excel('pred_y.xlsx', index=False)

def adb(x_train, y_train, x_test, y_test, well_num):
    # 进行训练
    adbr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=10)
    adbr.fit(x_train, y_train)
    pred_y = adbr.predict(x_test)
    pred_y = pred_y.reshape(-1, 1)
    # pred_y = pred_y+np.random. uniform ( low=0 , high=1.0 , size=1 )

    mse = mean_squared_error(pred_y, y_test)
    # calculate RMSE 均方根误差
    rmse = math.sqrt(mean_squared_error(pred_y, y_test))
    # 平均绝对误差
    mae = mean_absolute_error(pred_y, y_test)
    mre = np.average(np.abs(pred_y - y_test) / y_test, axis=0)
    print('平均相对误差：%.3f' % mre)

    print('均方误差：%.6f' % round(mse, 3))
    print('均方根误差：%.6f' % round(rmse, 3))
    print('平均绝对误差：%.6f' % round(mae, 3))

    # 绘制图像
    y_test = list(y_test.flatten())
    pred_y = list(pred_y.flatten())

    # 下面是绘图部
    labels = ["{}".format(int(i[0])) for i in well_num]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    width_1 = 0.4
    ax.bar(np.arange(len(pred_y)), pred_y, width=width_1, tick_label=labels, label="预测值")
    ax.bar(np.arange(len(y_test)) + width_1, y_test, width=width_1, tick_label=labels, label="真实值")
    ax.set_ylabel('EUR', color='k')
    ax.set_xlabel('井号', color='k')
    ax.legend()
    plt.show()
    pred_y = pd.DataFrame(pred_y)
    pred_y.to_excel('pred_y.xlsx', index=False)




from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import Voronoi
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering

def kmeans_knn(x_train, x_test):
    # 制作训练集（XY坐标）
    x_train_kmeans = x_train[:, -2:]
    # 制作验证集（XY坐标）
    x_test_kmeans = x_test[:, -2:]
    # 用轮廓系数方法调参(结果k=3)
    # L = []
    # for i in range(2, 21):
    #     k = i
    #     kmeans = KMeans(n_clusters=k, random_state=666)
    #     kmeans.fit(x_train_kmeans)
    #     a = silhouette_score(x_train_kmeans, kmeans.labels_)
    #     L.append((k, a))
    # b = pd.DataFrame(L)
    # b.columns = ['k', 's']
    # plt.figure(figsize=(8, 6), dpi=100)
    # plt.plot(b.k, b.s, color='r')
    # plt.xticks(b.k)
    # plt.xlabel('k')
    # plt.ylabel('轮廓系数')
    # plt.title('kmeans轮廓系数学习曲线')
    # plt.show()
    model = KMeans(n_clusters=3)
    model.fit(x_train_kmeans)
    # 分类中心点坐标
    res0Series = pd.Series(model.labels_)
    # 获取第0类数据
    res0 = res0Series[res0Series.values == 0]
    result0 = x_train[res0.index]
    # 获取第1类数据
    res1 = res0Series[res0Series.values == 1]
    result1 = x_train[res1.index]
    # 获取第2类数据
    res2 = res0Series[res0Series.values == 2]
    result2 = x_train[res2.index]

    # 预测
    preSeries = pd.Series(model.predict(x_test_kmeans))
    index0 = preSeries[preSeries.values == 0].index
    index1 = preSeries[preSeries.values == 1].index
    index2 = preSeries[preSeries.values == 2].index

    res = [result0, result1, result2]
    index = [index0, index1, index2]

    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):  # 先取属性
        y_test_sgl = pd.DataFrame()
        for j in range(3):  # 再取聚类结果
            result_i = res[j]  # 训练数据
            index_i = index[j]
            knn = KNeighborsRegressor(weights='distance', n_neighbors=5)
            knn.fit(result_i[:, [-2, -1]], result_i[:, i])
            print(x_test_kmeans[index_i, :])
            y_test_Kmeans = knn.predict(x_test_kmeans[index_i, :])
            y_test_Kmeans = pd.concat((pd.DataFrame(index_i).unstack(), pd.DataFrame(y_test_Kmeans).unstack()), axis=1)
            y_test_sgl = pd.concat((y_test_sgl, y_test_Kmeans), axis=0)
        y_test_sgl = y_test_sgl.sort_values(by=0, axis=0)  # 按索引进行排序
        y_test_sgl = y_test_sgl.iloc[:, 1:]  # 去掉索引
        y_test = pd.concat((y_test, y_test_sgl), axis=1)
    y_test = np.array(y_test)
    return y_test

#聚类这里有问题，应该是训练集验证集统一划分而不是后面再预测
def dbscan_knn(x_train, x_test):
    # 制作训练集（XY坐标）
    x_train_kmeans = x_train[:, -2:]
    # 制作验证集（XY坐标）
    x_test_kmeans = x_test[:, -2:]
    model = DBSCAN(eps=0.2,min_samples=4, algorithm='kd_tree')#algorithm='kd_tree'
    model.fit(x_train_kmeans)
    # 分类中心点坐标
    res0Series = pd.Series(model.labels_)
    # 获取第0类数据
    res0 = res0Series[res0Series.values == 0]
    result0 = x_train[res0.index]
    # 获取第1类数据
    res1 = res0Series[res0Series.values == 1]
    result1 = x_train[res1.index]
    # 获取第2类数据
    res2 = res0Series[res0Series.values == 2]
    result2 = x_train[res2.index]
    # 获取第3类数据

    # 预测
    preSeries = pd.Series(model.fit_predict(x_test_kmeans))
    index0 = preSeries[preSeries.values == 0].index
    index1 = preSeries[preSeries.values == 1].index
    index2 = preSeries[preSeries.values == -1].index


    res = [result0, result1, result2]
    index = [index0, index1, index2]

    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):  # 先取属性
        y_test_sgl = pd.DataFrame()
        for j in range(3):  # 再取聚类结果
            result_i = res[j]  # 训练数据
            index_i = index[j]
            knn = KNeighborsRegressor(weights='distance', n_neighbors=5, algorithm='kd_tree')
            knn.fit(result_i[:, [-2, -1]], result_i[:, i])
            y_test_Kmeans = knn.predict(x_test_kmeans[index_i, :])

            # param_dict = {'weights': ['distance'], 'n_neighbors': [5]}
            # clf = GridSearchCV(knn, param_dict, cv=2)
            # clf.fit(result_i[:, [-2, -1]], result_i[:, i])
            # y_test_Kmeans = clf.predict(x_test_kmeans[index_i, :])

            y_test_Kmeans = pd.concat((pd.DataFrame(index_i).unstack(), pd.DataFrame(y_test_Kmeans).unstack()), axis=1)
            y_test_sgl = pd.concat((y_test_sgl, y_test_Kmeans), axis=0)
        y_test_sgl = y_test_sgl.sort_values(by=0, axis=0)  # 按索引进行排序
        y_test_sgl = y_test_sgl.iloc[:, 1:]  # 去掉索引
        y_test = pd.concat((y_test, y_test_sgl), axis=1)
    y_test = np.array(y_test)

    return y_test


def custom_distance(x, y):
    euclidean_dist = np.sqrt(np.sum((x - y) ** 2))
    weight = 1.0 / euclidean_dist

    return euclidean_dist * weight


def KNN(x_train, x_test):
    # 制作训练集（XY坐标）
    x_train_KNN = x_train[:, -2:]
    # 制作验证集（XY坐标）
    x_test_KNN = x_test[:, -2:]
    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):
        # 制作标签（筛选的其他因素）
        y_train_KNN = x_train[:, i]
        knn = KNeighborsRegressor(weights='distance', n_neighbors=5)  # weights='distance' uniform
        # knn = KNeighborsRegressor(metric=custom_distance, n_neighbors=5)
        knn.fit(x_train_KNN, y_train_KNN)
        y_test_KNN = knn.predict(x_test_KNN)
        y_test_KNN = pd.DataFrame(y_test_KNN)
        y_test = pd.concat((y_test, y_test_KNN), axis=1)
    y_test = np.array(y_test)
    return y_test

def idw(x_train, x_test):
    # lon和lat分别是要插值的点的x,y
    # lst是已有数据的数组，结构为：[[x1，y1，z1]，[x2，y2，z2]，...]
    # 返回值是插值点的高程
    list_lon = x_test[:, [-2]]
    list_lat = x_test[:, [-1]]
    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):
        lst = np.concatenate((x_train[:, -2:], x_train[:, i].reshape(-1,1)), axis=1).tolist()
        # 插值的点
        y_test_idw = []
        for j in range(list_lon.shape[0]):
            lon = list_lon[j]
            lat = list_lat[j]
            pointvalue = interpolation(lon, lat, lst)
            y_test_idw.append(pointvalue)
        y_test_idw = pd.DataFrame(y_test_idw)
        y_test = pd.concat((y_test, y_test_idw), axis=1)
    y_test = pd.DataFrame(y_test)
    return y_test

def interpolation(lon, lat , lst):
    p0 = [lon, lat]
    sum0 = 0
    sum1 = 0
    temp = []
    # 遍历获取该点距离所有采样点的距离
    for point in lst:
        if lon == point[0] and lat == point[1]:
            return point[2]
        Di = distance(p0, point)
        # new出来一个对象，不然会改变原来lst的值
        import copy
        ptn = copy.deepcopy(point)
        ptn.append(Di)
        temp.append(ptn)

    # 根据上面ptn.append（）的值由小到大排序
    temp1 = sorted(temp, key=lambda point: point[3])
    # 遍历排序的前15个点，根据公式求出sum0 and sum1
    P = 3
    for point in temp1[0:15]:
        sum0 += point[2] / math.pow(point[3], P)
        sum1 += 1 / math.pow(point[3], P)
    return sum0 / sum1

# 计算两点间的距离
def distance(p, pi):
    dis = (p[0] - pi[0]) * (p[0] - pi[0]) + (p[1] - pi[1]) * (p[1] - pi[1])
    m_result = math.sqrt(dis)
    return m_result

def Thiessen(x_train, x_test):
    vor = Voronoi(x_test)
    x_test = vor.ridge_dict
    return vor.x_test

def spline(x_train, x_test):
    X_0 = x_train[:, [0]].reshape(-1,)
    Fill_X = x_test[:, [0]]
    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):
        Y_0 = x_train[:, [i]].reshape(-1,)
        IRFunction = interp1d(X_0, Y_0, kind='linear', fill_value="extrapolate")  #‘quadratic’ 、'cubic’
        Fill_Y = pd.DataFrame(IRFunction(Fill_X))
        y_test = pd.concat((y_test, Fill_Y), axis=1)
    return y_test

def rf(x_train, x_test):
    # 制作训练集（XY坐标）
    x_train_KNN = x_train[:, -2:]
    # 制作验证集（XY坐标）
    x_test_KNN = x_test[:, -2:]
    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):
        # 制作标签（筛选的其他因素）
        y_train_KNN = x_train[:, i]
        rf=RandomForestRegressor()
        rf.fit(x_train_KNN, y_train_KNN)
        y_test_KNN = rf.predict(x_test_KNN)
        y_test_KNN = pd.DataFrame(y_test_KNN)
        y_test = pd.concat((y_test, y_test_KNN), axis=1)
    y_test = np.array(y_test)
    return y_test

def xgboost(x_train, x_test):
    # 制作训练集（XY坐标）
    x_train_KNN = x_train[:, -2:]
    # 制作验证集（XY坐标）
    x_test_KNN = x_test[:, -2:]
    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):
        # 制作标签（筛选的其他因素）
        y_train_KNN = x_train[:, i]

        # 2.参数集定义
        param_grid = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'n_estimators': [30, 50, 100, 300, 500, 1000, 2000],
            'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.03, 0.05, 0.5],
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
            "reg_alpha": [0.0001, 0.001, 0.01, 0.1, 1, 100],
            "reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1, 100],
            "min_child_weight": [2, 3, 4, 5, 6, 7, 8],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "subsample": [0.6, 0.7, 0.8, 0.9]}
        # 3.随机搜索并打印最佳参数
        gsearch1 = RandomizedSearchCV(XGBRegressor(scoring='ls', seed=27), param_grid, cv=5)
        gsearch1.fit(x_train_KNN, y_train_KNN)
        print("best_score_:", gsearch1.best_params_, gsearch1.best_score_)

        # 4.用最佳参数进行预测
        y_test_KNN = gsearch1.predict(x_test_KNN)

        y_test_KNN = pd.DataFrame(y_test_KNN)
        y_test = pd.concat((y_test, y_test_KNN), axis=1)
    y_test = np.array(y_test)
    return y_test


def ms_knn(x_train, x_test):
    # 制作训练集（XY坐标）
    x_train_kmeans = x_train[:, -2:]
    # 制作验证集（XY坐标）
    x_test_kmeans = x_test[:, -2:]

    bandwidth = estimate_bandwidth(x_train_kmeans, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x_train_kmeans)

    # 分类中心点坐标
    res0Series = pd.Series(ms.labels_)
    # 获取第0类数据
    res0 = res0Series[res0Series.values == 0]
    result0 = x_train[res0.index]
    # 获取第1类数据
    res1 = res0Series[res0Series.values == 1]
    result1 = x_train[res1.index]
    # 获取第2类数据
    res2 = res0Series[res0Series.values == 2]
    result2 = x_train[res2.index]
    # 获取第3类数据
    res3 = res0Series[res0Series.values == 3]
    result3 = x_train[res3.index]
    # 获取第4类数据
    res4 = res0Series[res0Series.values == 4]
    result4 = x_train[res4.index]

    # 预测
    preSeries = pd.Series(ms.predict(x_test_kmeans))
    index0 = preSeries[preSeries.values == 0].index
    index1 = preSeries[preSeries.values == 1].index
    index2 = preSeries[preSeries.values == 2].index
    index3 = preSeries[preSeries.values == 3].index
    index4 = preSeries[preSeries.values == 4].index

    res = [result0, result1, result2, result3, result4]
    index = [index0, index1, index2, index3, index4]

    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):  # 先取属性
        y_test_sgl = pd.DataFrame()
        for j in range(5):  # 再取聚类结果
            result_i = res[j]  # 训练数据
            index_i = index[j]
            knn = KNeighborsRegressor(weights='distance', n_neighbors=4)
            knn.fit(result_i[:, [-2, -1]], result_i[:, i])
            y_test_Kmeans = knn.predict(x_test_kmeans[index_i, :])
            y_test_Kmeans = pd.concat((pd.DataFrame(index_i).unstack(), pd.DataFrame(y_test_Kmeans).unstack()), axis=1)
            y_test_sgl = pd.concat((y_test_sgl, y_test_Kmeans), axis=0)
        y_test_sgl = y_test_sgl.sort_values(by=0, axis=0)  # 按索引进行排序
        y_test_sgl = y_test_sgl.iloc[:, 1:]  # 去掉索引
        y_test = pd.concat((y_test, y_test_sgl), axis=1)
    y_test = np.array(y_test)
    return y_test


def sc_knn(x_train, x_test):
    # 制作训练集（XY坐标）
    x_train_kmeans = x_train[:, -2:]
    # 制作验证集（XY坐标）
    x_test_kmeans = x_test[:, -2:]

    # # 调参数============
    #
    # for i, gamma in enumerate((0.01, 0.1)):
    #     for j, k in enumerate((3,4,5,6,7,8)):
    #         y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(x_train_kmeans)
    #         print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k, "score:",
    #               metrics.calinski_harabasz_score(x_train_kmeans, y_pred))
    # Calinski - Harabasz Score with gamma= 0.1 n_clusters= 4 score: 203.16804054418844

    ms = SpectralClustering(gamma=0.1, n_clusters=4)
    ms.fit(x_train_kmeans)

    # 分类中心点坐标
    res0Series = pd.Series(ms.labels_)
    # 获取第0类数据
    res0 = res0Series[res0Series.values == 0]
    result0 = x_train[res0.index]
    # 获取第1类数据
    res1 = res0Series[res0Series.values == 1]
    result1 = x_train[res1.index]
    # 获取第2类数据
    res2 = res0Series[res0Series.values == 2]
    result2 = x_train[res2.index]
    # 获取第3类数据
    res3 = res0Series[res0Series.values == 3]
    result3 = x_train[res3.index]
    # 获取第4类数据
    res4 = res0Series[res0Series.values == 4]
    result4 = x_train[res4.index]
    # 预测
    preSeries = pd.Series(SpectralClustering(gamma=0.1, n_clusters=4).fit_predict(x_test_kmeans))
    index0 = preSeries[preSeries.values == 0].index
    index1 = preSeries[preSeries.values == 1].index
    index2 = preSeries[preSeries.values == 2].index
    index3 = preSeries[preSeries.values == 3].index
    index4 = preSeries[preSeries.values == 4].index

    res = [result0, result1, result2, result3, result4]
    index = [index0, index1, index2, index3, index4]

    y_test = pd.DataFrame()
    for i in range(1, x_train.shape[1] - 2):  # 先取属性
        y_test_sgl = pd.DataFrame()
        for j in range(4):  # 再取聚类结果
            result_i = res[j]  # 训练数据
            index_i = index[j]
            knn = KNeighborsRegressor(weights='distance', n_neighbors=4)
            knn.fit(result_i[:, [-2, -1]], result_i[:, i])
            y_test_Kmeans = knn.predict(x_test_kmeans[index_i, :])
            y_test_Kmeans = pd.concat((pd.DataFrame(index_i).unstack(), pd.DataFrame(y_test_Kmeans).unstack()), axis=1)
            y_test_sgl = pd.concat((y_test_sgl, y_test_Kmeans), axis=0)
        y_test_sgl = y_test_sgl.sort_values(by=0, axis=0)  # 按索引进行排序
        y_test_sgl = y_test_sgl.iloc[:, 1:]  # 去掉索引
        y_test = pd.concat((y_test, y_test_sgl), axis=1)
    y_test = np.array(y_test)
    return y_test

def inverse_transformdata(x_test):
    data_sc = pd.read_excel("./重科院数据new1（以前的坐标）.xlsx")
    data_sc = data_sc.iloc[:, 2:8]
    data_sc = sc.fit_transform(data_sc)
    x_test = sc.inverse_transform(x_test)
    return x_test

# 制作数据集
def makeData(data, random_state):
    # 选择因素(倒数2,3列是xy，最后一列是EUR)
    # db_train = data.iloc[:, [0, 1, 2, 10, 11, 12]]  # 选取井号，地质因素，xy，EUR两个特征打开
    db_train = data
    db_x = db_train.iloc[:, :-1]
    db_y = db_train.iloc[:, -1:]
    print('用于训练的')
    print(db_x.columns)

    db_x = np.array(db_x)
    db_y = np.array(db_y)
    # x_test_index = [64, 8, 7, 51, 71, 85, 6, 65, 20, 22, 62, 52, 88, 49, 87, 91, 35, 1, 56, 93, 23, 4, 82, 38, 14, 73,
    #                 11, 3, 10, 81]

    x_test_index = [64,8,85,6,65,20,22,88,87,91,35,1,56,93,23,4,82,14,73,11,10,81]

    x_train_index = [80, 90, 45, 69, 19, 18, 15, 94, 95, 24, 97, 68, 31, 0, 13, 50, 55, 2, 26,
     86, 12, 16, 96, 83, 75, 41, 9, 40, 76, 43, 74, 21, 57, 60, 59, 79, 30, 44,
     98, 67, 77, 25, 32, 89, 5, 48, 92, 53, 54, 78, 28, 36, 61, 39, 72, 29, 70,
     66, 84, 46, 33, 58, 27, 47, 63, 37, 34, 17, 42]

    x_test = db_x[x_test_index, :]
    y_test = db_y[x_test_index, :]
    x_train = db_x[x_train_index, :]
    y_train = db_y[x_train_index, :]

    # x_train = np.delete(db_x, x_test_index, axis=0)
    # y_train = np.delete(db_y, x_test_index, axis=0)
    # x_train, x_test, y_train, y_test = train_test_split(db_x, db_y, test_size=0.3, random_state=random_state)
    well_num = x_test[:, [0]]
    print(len(well_num))

    zuobiao = x_test[:, [-2,-1]]
    x_test_th = x_test[:, 1:-2]
    # 选择插值方式
    x_test = KNN(x_train, x_test)
    # x_test = idw(x_train, x_test)
    # x_test = kmeans_knn(x_train, x_test)
    # x_test = dbscan_knn(x_train, x_test)
    # x_test = Thiessen(x_train, x_test)
    # x_test = spline(x_train, x_test)
    # x_test = kriging_python(x_train, x_test)
    # x_test = rf(x_train, x_test)
    # x_test = xgboost(x_train, x_test)beijuan
    # x_test = rfok(x_train, x_test, x_test_th)
    # x_test = kmeans_kriging(x_train, x_test)
    # x_test = ms_knn(x_train, x_test)
    # x_test = ms_kriging(x_train, x_test)
    # x_test = sc_knn(x_train, x_test)


    x_train = x_train[:, 1:-2]

    # 验证集反归一化
    # 选择需要归一化的列
    x_test_th_org = inverse_transformdata(x_test_th)
    x_test_org = inverse_transformdata(x_test)
    x_test_org = pd.DataFrame(x_test_org)
    # x_test_org.to_excel("dbscan_knn.xlsx", index=None)
    x_test_org = inverse_transformdata(x_test)



    # # 插值方法评价
    # mse = mean_squared_error(x_test_org, x_test_th_org)
    # rmse = math.sqrt(mean_squared_error(x_test_org, x_test_th_org))
    # mae = mean_absolute_error(x_test_org, x_test_th_org)
    # mre = np.average(np.average(np.abs(x_test_org - x_test_th_org) / x_test_th_org, axis=0))
    # print('\n')
    # print('平均相对误差：%.3f' % mre)
    # print('均方误差：%.3f' % mse)
    # print('均方根误差：%.3f' % rmse)
    # print('平均绝对误差：%.3f' % mae)
    #
    # # 打印每一个的指标
    # for i in range(x_test_org.shape[1]):
    #     print(i)
    #     print(mean_squared_error(x_test_org[:,i], x_test_th_org[:,i]))
    #     print(math.sqrt(mean_squared_error(x_test_org[:, i], x_test_th_org[:, i])))
    #     print(np.average(np.abs(x_test_org[:, i] - x_test_th_org[:, i]) / x_test_th_org[:, i], axis=0))
    # # 评价结束

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    np.random.seed(11)
    np.random.shuffle(x_train)
    np.random.seed(11)
    np.random.shuffle(y_train)

    return x_train, x_test, y_train, y_test, well_num, zuobiao


# 用于确定使用什么算法预测EUR,well_type(井的类型)，algorithm_type（选择预测方法）
from tqdm import tqdm
def sourcePre(algorithm_type):
    start = time.time()
    data = dealMaxMin()
    mre = []
    best_mre = 777
    best_i = 0
    for i in tqdm(range(2000)):#1223
        i = 1223
        x_train, x_test, y_train, y_test, well_num, zuobiao = makeData(data, i)
        # x_train = np.concatenate((x_train,x_train))
        # x_train = np.concatenate((x_train, x_train))
        # y_train = np.concatenate((y_train,y_train))
        # y_train = np.concatenate((y_train, y_train))

        end = time.time()
        print("预处理运行时间：", (end - start)*1000)

        if algorithm_type == "xgb":
            # z, mre = xg(x_train, y_train, x_test, y_test, well_num=well_num)
            pass
        elif algorithm_type == "elman":
            elman(x_train, y_train, x_test, y_test, well_num=well_num)
        elif algorithm_type == 'svr':
            z = svr(x_train, y_train, x_test, y_test, well_num=well_num)
        elif algorithm_type == 'rbf':
            rbf(x_train, y_train, x_test, y_test,  well_num=well_num)
        elif algorithm_type == 'esn':
            esn(x_train, y_train, x_test, y_test, well_num=well_num)
        elif algorithm_type == 'adb':
            adb(x_train, y_train, x_test, y_test, well_num=well_num)
        elif algorithm_type == 'Stacking':
            Stacking(x_train, y_train, x_test, y_test, well_num=well_num)
        break
        if mre<best_mre:
            best_mre = mre
            best_i = i
    print("最好的平均相对误差是{}， 数据集划分的代号是{}。".format(best_mre, best_i))
    return x_train, x_test, y_train, y_test



if __name__ == '__main__':
    # # 设置，0代表地质因素、1代表地质+钻井、2、代表地质+钻井+压裂
    start = time.time()
    x_train, x_test, y_train, y_test = sourcePre(algorithm_type='xgb')
    end = time.time()
    print("qqqq运行时间：", end-start)
