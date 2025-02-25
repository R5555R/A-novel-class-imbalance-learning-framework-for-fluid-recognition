import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 构造训练数据和软标签
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 5)  # 假设有5个类别
print(y_train)

# 构造数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)  # 输出5个类别的概率值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


net = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 开始训练
for epoch in range(100):
    for X, y_soft in train_loader:
        optimizer.zero_grad()
        y_pred = net(X)
        loss = criterion(y_pred, torch.argmax(y_soft, dim=1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))
X_test = torch.randn(10, 10)
net.eval() # 将模型设为评估模式
with torch.no_grad(): # 不需要计算梯度，加快运算速度
    y_prob = net(X_test)

    print(y_prob) # 输出预测的概率值