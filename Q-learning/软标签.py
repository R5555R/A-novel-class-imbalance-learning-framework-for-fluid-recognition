import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 定义软标签训练数据集
class SoftLabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, running_loss / len(train_loader)))


# 加载数据和标签
data = torch.randn(6000, 11)
labels = torch.randn(6000, 6)

# 定义软标签训练数据集
train_dataset = SoftLabelDataset(data, labels)

# 定义训练数据集的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义神经网络模型、损失函数和优化器
model = Net()
criterion = nn.KLDivLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练神经网络
train(model, train_loader, criterion, optimizer, num_epochs=50)
