import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier  # GDBT梯度提升树
from sklearn.ensemble import ExtraTreesClassifier  # ET极限森林
from catboost import CatBoostClassifier  # CatBoost
from xgboost import XGBClassifier  # XGBoost
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import deque
from scipy.stats import mode
from sklearn.metrics import log_loss
from tqdm import tqdm


# load data
train_data = pd.read_csv('..//data//精度1//E-train-1.csv')
train_data1 = pd.read_csv('..//data//精度1//无标签126层数据（8口井）.csv')
test_data = pd.read_csv('..//data//精度1//E-test.csv')
# 合并所有数据
data = pd.concat((train_data, train_data1, test_data), axis=0)
features = ['SP', 'PE', 'GR', 'U', 'TH', 'K', 'AC', 'CNL', 'DEN', 'RLLS', 'RLLD']
train_x = data[features]
le = LabelEncoder()
LABEL = le.fit_transform(data['LABEL'])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data[features].values, LABEL, test_size=0.2)

# 训练三个决策树基分类器
estimator0 = GradientBoostingClassifier()
estimator1 = ExtraTreesClassifier()
estimator2 = CatBoostClassifier()
estimator3 = XGBClassifier()

estimator = [estimator0, estimator1, estimator2, estimator3]
base_classifiers = []

for i in estimator:
    clf = i
    clf.fit(X_train, y_train)
    base_classifiers.append(clf)
print([accuracy_score(y_test, k.predict(X_test)) for k in base_classifiers])
one_hot = OneHotEncoder(sparse=False)
one_hot.fit(y_train.reshape(-1, 1))

# 定义 Q-learning 算法
num_classes = 6
n_actions = 2**6  # 6 个模型的组合数
Q = np.zeros((num_classes, n_actions))  # Q 表
epsilon = 0.1  # ε-greedy 策略的 ε 值
alpha = 0.01  # 学习率
gamma = 0.9  # 折扣因子

# ε-greedy 策略选择动作
def epsilon_greedy(Q, state):
    if np.random.uniform() < epsilon:
        num_models = np.random.randint(1, 7)  # 随机选择要选择的模型数量
        models = np.random.choice([0, 1, 2, 3, 4, 5], size=num_models, replace=False)  # 随机选择模型
        return list(models)  # 返回一个整数列表，代表选择的模型
    else:
        # 根据 Q 值选择最佳的模型
        best_models = np.argwhere(Q[state] == np.amax(Q[state]))
        return best_models.flatten().tolist()  # 返回一个整数列表，代表选择的模型

# 定义环境函数
def get_state(state, features, probs, loss):
    # 将新的特征、预测结果、标准差和性能打包成元组
    data = (features, probs, loss)
    # 将数据添加到队列的末尾
    state.append(data)
    return state

# 定义神经网络Q表
class Q_Net(nn.Module):
    def __init__(self):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(num_classes, 32)
        self.fc2 = nn.Linear(32, n_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# 训练 Q-learning 模型
for episode in tqdm(range(100)):
    # 初始化一个长度为10的队列
    state = deque([0 for _ in range(10)], maxlen=10)  # 初始状态
    probs = []
    teacher_loss = []
    for i in range(len(X_train)):
        action = epsilon_greedy(Q, state)  # 选择动作
        print(action)

        # 执行动作并观察新状态和奖励
        # 选模型有问题，解决方法是把actor进行编码，有多少个动作就编多少码
        for j, clf in enumerate(action):
            probs.append(base_classifiers[clf].predict_proba([X_train[i]])[0])

        # 计算基分类器的预测结果和交叉熵损失
        y_true = one_hot.transform(y_train[i].reshape(-1, 1)).reshape(-1, )
        loss_clf = [log_loss(y_true, pred) for pred in probs]
        teacher_loss.append(sum(loss_clf))  # 该结果要放入state

    # 获取了预测结果后融合预测结果
    soft_targets = np.mean(probs, axis=0)
    # 定义学生模型
    model_student = XGBClassifier()
    # 使用教师模型中的软目标训练学生模型
    model_student.fit(X_train, soft_targets)

    # 计算强化学习模型的预测结果和交叉熵损失
    y_pred_student = model_student.predict(X_test)
    student_loss = log_loss(one_hot.transform(y_test.reshape(-1, 1)).reshape(-1, ), y_pred_student)
    reward = -teacher_loss - student_loss

    features = X_train[i]
    next_state = get_state(state, features, soft_targets, reward)  # 计算下一个状态
    Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action])  # 更新 Q 表

    state = next_state  # 跳转到下一个状态
    print(state)

# 测试 Q-learning 模型
print(Q)
correct = 0
state = 0
pred = []
true = []
for i in range(len(X_test)):
    action = np.argmax(Q[state, :])  # 选择动作
    # print(action)

    # 执行动作并观察新状态和奖励
    if action == 0:
        model = base_classifiers[0]
    elif action == 1:
        model = base_classifiers[1]
    elif action == 2:
        model = base_classifiers[2]
    elif action == 3:
        model = base_classifiers[3]

    result = model.predict([X_test[i]])[0]
    pred.append(result)
    if result == y_test[i]:
        correct += 1
import sklearn.metrics as metrics

print("评价指标-1：")
print(metrics.classification_report(y_test, pred))
print("Accuracy:", correct / len(X_test))
