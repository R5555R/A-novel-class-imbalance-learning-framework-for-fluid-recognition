
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


import warnings
warnings.filterwarnings(action='ignore')

# 制作平方残差数据集
def randomLoss(X_train,accuracys):
    accuracys=accuracys[:,np.newaxis]
    data = np.hstack((X_train, accuracys))
    data_=pd.DataFrame(data)
    X_=data_.iloc[:,:-1]
    Y_=data_.iloc[:,-1:]

    X_np = X_.values
    y_np = Y_.values

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)
    return X_tensor,y_tensor

class BPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 输入到隐藏层1
        self.fc2 = nn.Linear(128, 64)         # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, output_dim)  # 隐藏层2到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# def train_():
#     x_train_,Y_train_=randomLoss(X_train,accuracys)
#
#     model = BPModel(input_dim=7, output_dim=1)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     num_epochs = 200
#     #训练
#     for epoch in range(num_epochs):
#         # 前向传播
#         outputs = model(x_train_)
#         loss = criterion(outputs, Y_train_)
#
#         # 反向传播和优化
#         optimizer.zero_grad()  # 清除上一步的梯度
#         loss.backward()        # 计算梯度
#         optimizer.step()       # 更新参数
#
#         # 每隔100次输出损失
#         if (epoch + 1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#     # 训练完成后保存模型权重
#     torch.save(model.state_dict(), 'bp_model_weights.pth')


# # 预测
# model.load_state_dict(torch.load('bp_model_weights.pth'))
# model.eval()  # 切换到评估模式
# with torch.no_grad():
#     predictions = model(x_train_)
# # 打印预测结果
# print(predictions)
