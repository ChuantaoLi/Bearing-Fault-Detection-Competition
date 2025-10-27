# ===================================================================================
# 文件名: model.py
# 描述: CNN 模型结构 (添加特征返回)
# ===================================================================================
import torch.nn as nn
import torch.nn.functional as F


class Simple2DCNN(nn.Module):
    """
    一个简单的二维卷积神经网络，用于 64x32 图像分类。
    - num_classes: 输出类别的数量。
    """

    def __init__(self, num_classes, dropout=0.5):
        super(Simple2DCNN, self).__init__()
        # Input: 1 x 64 x 32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 16 x 32 x 16

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 32 x 16 x 8

        # 全连接层输入维度: 32 * 16 * 8 = 4096
        self.feature_dim = 32 * 16 * 8
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

        # CenterLoss 需要的特征维度
        self.center_feature_dim = 128

    def forward(self, x, return_features=False):
        # 卷积层
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # 展平特征
        x = x.view(x.size(0), -1)

        # 提取 CenterLoss 所需的特征 (fc1 输出)
        center_features = self.dropout(F.relu(self.fc1(x)))

        # 最终分类 Logits
        logits = self.fc2(center_features)

        if return_features:
            # 返回 logits 和 center_features
            return logits, center_features
        return logits