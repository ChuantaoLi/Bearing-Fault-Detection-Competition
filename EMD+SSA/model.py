import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    """
    一个简单的一维卷积神经网络，用于序列分类。
    - num_classes: 输出类别的数量。
    """
    def __init__(self, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x