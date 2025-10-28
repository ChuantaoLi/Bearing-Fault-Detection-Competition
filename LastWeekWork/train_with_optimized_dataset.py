#!/usr/bin/env python3
"""
*** 修改版 ***
加载 'augmented_dataset_5fold.pkl'。
Optuna 调优：使用 pkl 文件中预先划分的 5 折数据进行交叉验证。
最终训练：使用第 1 折的数据进行训练/验证划分，训练最终模型。
最终预测：使用最终模型对 pkl 文件中的全局测试集进行预测。
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import optuna
from collections import Counter
import shutil

# ==================== 配置参数 ====================
# *** 修改点: 指向您新生成的、包含5折数据的文件 ***
OPTIMIZED_DATASET_PATH = r"augmented_dataset_5fold.pkl"

"""
每个频谱样本 = 7条时间序列
    ↓
序列1: 低频能量随时间变化 [值1, 值2, ..., 值48]
序列2: 中低频能量随时间变化 [值1, 值2, ..., 值48]
...
序列7: 高频能量随时间变化 [值1, 值2, ..., 值48]

每个包络样本 = 5条时间序列  
    ↓
序列1: 包络特征1随时间变化 [值1, 值2, ..., 值32]
序列2: 包络特征2随时间变化 [值1, 值2, ..., 值32]
...
序列5: 包络特征5随时间变化 [值1, 值2, ..., 值32]
"""

# 超参数优化配置
ENABLE_HYPERPARAMETER_TUNING = True  # 是否启用超参数优化
N_TRIALS = 100  # Optuna试验次数
OPTIMIZATION_TIMEOUT = 7200  # 优化超时时间（秒）

# 训练参数（优化时会被覆盖）
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 40
DROPOUT_RATE = 0.4

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ==================== 辅助函数 ====================
def calculate_class_weights(y_train):
    """计算类别权重用于处理类别不平衡"""
    class_counts = Counter(y_train)
    num_classes = len(class_counts)

    # 确保我们知道总类别数, 即使某个类别在y_train中不存在
    max_class_id = 0
    if class_counts:  # 确保
        max_class_id = max(class_counts.keys())
    total_num_classes = max(num_classes, max_class_id + 1)

    total_samples = len(y_train)

    # 转换为tensor
    weights = torch.zeros(total_num_classes)
    for class_id, count in class_counts.items():
        if count > 0:
            weights[class_id] = total_samples / (total_num_classes * count)
        else:
            weights[class_id] = 1.0  # 理论上不应发生

    # 处理 y_train 中可能缺失的类别
    for i in range(total_num_classes):
        if i not in class_counts:
            weights[i] = 1.0  # 如果某个类别完全缺失，给一个中性权重

    return weights


# ==================== 损失函数定义 ====================
class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡问题

    用一个例子演示这个计算过程：

    1. 假设批次为3，有如下3个样本：
    样本1: [3.0, 1.0, 0.5, -0.5, 1.5, 0.0]   → 真实类别: 0
    样本2: [0.5, 2.5, 1.0, -1.0, 0.0, 0.5]   → 真实类别: 1
    样本3: [1.0, 0.5, 2.0, -0.5, 1.0, -1.0]  → 真实类别: 2

    2. 计算Softmax概率：
    样本1: softmax([3.0, 1.0, 0.5, -0.5, 1.5, 0.0])
      = [0.830, 0.112, 0.037, 0.005, 0.015, 0.001]
      → 真实类别0的概率 p₀ = 0.830
      → ce_loss = -log(0.830) = 0.186
    其余俩样本同理，最终得到：ce_loss = [0.186, 0.408, 0.245]

    3. 计算概率项 (pt)
    pt 并不是"真实类别的概率"，而是"模型预测正确的概率"的数学表达。
    pt = torch.exp(-ce_loss)
    样本1: pt = exp(-0.186) = 0.830  (就是真实类别的概率p₀)
    样本2: pt = exp(-0.408) = 0.665  (就是真实类别的概率p₁)
    样本3: pt = exp(-0.245) = 0.783  (就是真实类别的概率p₂)

    4. 计算调节因子 (1-pt)^γ
    样本1: (1 - 0.830)² = (0.170)² = 0.0289
    样本2: (1 - 0.665)² = (0.335)² = 0.1122
    样本3: (1 - 0.783)² = (0.217)² = 0.0471

    5. 应用alpha平衡因子
    focal_loss = alpha × (1-pt)^γ × ce_loss
    样本1: 0.25 × 0.0289 × 0.186 = 0.00134
    样本2: 0.25 × 0.1122 × 0.408 = 0.01144
    样本3: 0.25 × 0.0471 × 0.245 = 0.00288

    6. 最终损失
    focal_loss = [0.00134, 0.01144, 0.00288]
    均值: (0.00134 + 0.01144 + 0.00288) / 3 = 0.00522
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

        '''
        alpha (默认0.25): 平衡因子，用于调节正负样本的权重
        gamma (默认2.0): 调节因子，控制难易样本的权重
        reduction (默认'mean'): 损失 reduction 方式，可选 'mean'、'sum'、'none'
        class_weights: 额外的类别权重，用于进一步处理类别不平衡
        '''

    def forward(self, inputs, targets):
        """计算交叉熵损失"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # 应用类别权重
        if self.class_weights is not None:
            # 确保 class_weights 在同一设备上
            if self.class_weights.device != targets.device:
                self.class_weights = self.class_weights.to(targets.device)
            weights = self.class_weights[targets]
            ce_loss = ce_loss * weights

        """
        损失函数：FL(pt) = α × (1 - pt)^γ × CE(pt)
        (1 - pt)^γ是调节因子，对于易分类样本 (pt → 1)，此项趋近于0，权重降低，gamma越大，对易分类样本的抑制越强
        α是平衡因子，用来给正负类平衡，多分类可以看作是目标类和非目标类，所以alpha用标量也是合理的
        CE(pt)是类别权重，在前面根据类别不平衡率进行计算
        """

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        """对一个批次内的样本的focal loss进行缩减"""
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失

    用一个例子演示计算过程：

    1. 假设输出的数据 logits (x) 如下：
        tensor([[3.0000, 1.0000, 0.5000, 0.2000],
            [0.5000, 2.5000, 1.0000, 0.1000],
            [1.0000, 0.5000, 2.5000, 0.3000]], requires_grad=True)
        其真实标签为: tensor([0, 1, 2])

    2. 设定平滑参数 smoothing: 0.1，置信度 confidence: 0.9

    3. 计算Softmax概率分布:
        样本0: [0.8307 0.1125 0.0373 0.0195] (总和: 1.0000)
        样本1: [0.0905 0.6652 0.2227 0.0216] (总和: 1.0000)
        样本2: [0.1185 0.064 0.7829 0.0346] (总和: 1.0000)

    4. 计算对数概率 logprobs:
        样本0: [-0.1855 -2.1846 -2.9875 -3.6376]
        样本1: [-2.4024 -0.4076 -1.4979 -3.835  ]
        样本2: [-2.1328 -2.7489 -0.2448 -3.3629]

    5. 提取的真实类别对数概率
        目标标签调整形状: target.unsqueeze(1): [[0], [1], [2]]
        形状: torch.Size([3, 1])
        gathered_logprobs: [[-0.18546521663665771], [-0.4076051712036133], [-0.24478435516357422]]

        负对数似然损失 (before squeeze): [[0.18546521663665771], [0.4076051712036133], [0.24478435516357422]]
        负对数似然损失 (after squeeze): [0.18546521663665771, 0.4076051712036133, 0.24478435516357422]
        形状: torch.Size([3])

    6. 计算平滑损失
        平滑损失: [2.248800039291382, 2.0357494354248047, 2.1220998764038086]
        形状: torch.Size([3])

    7. 组合损失
        损失组合公式: loss = confidence × nll_loss + smoothing × smooth_loss

        样本0: 0.9×0.1855 + 0.1×2.2488 = 0.3928
        样本1: 0.9×0.4076 + 0.1×2.0357 = 0.5703
        样本2: 0.9×0.2448 + 0.1×2.1221 = 0.4325

        逐样本损失: [0.392765074968338, 0.5703092217445374, 0.43250468373298645]

    8. 最终损失
        批量平均损失: 0.4652
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        confidence = 1. - self.smoothing  # 标签的置信度等于1减去平滑度，平滑度默认为1
        logprobs = F.log_softmax(logits, dim=-1)  # 对模型最后一层的输出进行softmax归一化

        """
        对于logits = torch.tensor([[2.0, 1.0, 0.1]])，将logits转换为概率分布
        结果会是 tensor([[0.6590, 0.2424, 0.0986]])，满足: 0.6590 + 0.2424 + 0.0986 ≈ 1.0
        对概率取自然对数：logprobs = torch.log(probs)
        结果会是 tensor([[-0.4170, -1.4170, -2.2170]])
        """

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        """
        上面这两行代码的作用是：从每个样本的概率分布中，只提取真实类别对应的损失值
        target.unsqueeze(1): 将目标标签从形状 [batch_size] 变为 [batch_size, 1]
        logprobs.gather(): 从logprobs中提取对应目标标签位置的对数概率
        -logprobs.gather(): 得到负对数似然损失
        squeeze(1): 将形状从 [batch_size, 1] 变回 [batch_size]
        gather后会得到差不多这样的形式：
        tensor([[-0.4170],   # 样本1的类别0的对数概率
         [-1.6000],   # 样本2的类别2的对数概率  
         [-1.7000]])  # 样本3的类别1的对数概率
        """

        smooth_loss = -logprobs.mean(dim=-1)  # 计算样本所有类别的平均损失，作为正则化项
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        """
        上面这两行代码是标签平滑的核心
        smooth_loos会差不多是下面这个形式：
        样本1: -(-0.4170 + -1.4170 + -2.3170 + -1.9170)/4 = -(-6.068)/4 = 1.517
        样本2: -(-2.1000 + -0.1000 + -1.6000 + -2.4000)/4 = -(-6.200)/4 = 1.550
        样本3: -(-1.2000 + -1.7000 + -0.2000 + -1.9000)/4 = -(-5.000)/4 = 1.250
        """

        return loss.mean()


class CenterLoss(nn.Module):
    """Center Loss用于增强特征的类内紧密度"""

    def __init__(self, num_classes, feature_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu

        """
        中心向量是Center Loss中为每个类别学习的一个代表性特征向量，可以理解为该类别的"平均特征"或"理想特征"
        中心向量是模型的可学习参数，不是计算得到的，而是在训练过程中通过梯度下降自动学习的
        """

        # *** 修改点: 增加CUDA可用性检查 ***
        if self.use_gpu and torch.cuda.is_available():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))
            if self.use_gpu:
                print("警告: CenterLoss 请求 GPU 但 CUDA 不可用, 自动降级到 CPU。")
                self.use_gpu = False

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        """
        x_norm = torch.pow(x, 2).sum(dim=1, keepdim=True)：对于批次中每个样本i，计算其特征向量的平方和，结果形状为 [batch_size, 1]
        x_norm_expanded = x_norm.expand(batch_size, self.num_classes)：将每个样本的平方和扩展为与类别数相同的列数，结果形状为 [batch_size, num_classes]

        假设x_norm为：
            tensor([[5.],   # 1² + 2² = 1 + 4 = 5
            [10.],  # 3² + 1² = 9 + 1 = 10
            [6.5]]) # 0.5² + 2.5² = 0.25 + 6.25 = 6.5
        那么拓展后的x_norm_expanded就为：
            tensor([[5., 5., 5., 5.],
            [10., 10., 10., 10.],
            [6.5, 6.5, 6.5, 6.5]])
        expand() 函数将 [3, 1] 的张量沿着第1维度复制4次，这样每行都变成了相同的4个值。

        center_norm = torch.pow(self.centers, 2).sum(dim=1, keepdim=True)：对于每个类别j，计算其中心向量的平方和，结果形状为 [num_classes, 1]
        center_norm_expanded = center_norm.expand(self.num_classes, batch_size).t()：将每个类别的平方和扩展为与批次大小相同的行数，并转置，结果形状为 [batch_size, num_classes]
        dot_product = torch.mm(x, self.centers.t())：计算批次中每个样本与每个类别中心的点积，结果形状为 [batch_size, num_classes]
        """

        classes = torch.arange(self.num_classes).long()  # 创建类别索引 [0, 1, 2, ..., num_classes-1]
        # *** 修改点: 确保设备一致性 ***
        if self.use_gpu and torch.cuda.is_available():
            classes = classes.cuda()

        # 确保 labels 和 classes 在同一设备上
        if labels.device != classes.device:
            classes = classes.to(labels.device)

        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)  # 拓展标签形状

        """
        假设 labels = [0, 1, 2], 扩展后:
        [[0, 0, 0],
        [1, 1, 1], 
        [2, 2, 2]]
        这里的[0, 1, 2]表示第一个样本的真实标签是0，第二个样本的真实标签是1，第三个样本的真实标签是2
        """

        mask = labels_expanded.eq(classes.expand(batch_size, self.num_classes))  # 创建掩码

        """
        [[True, False, False],
        [False, True, False],
        [False, False, True]]
        """

        dist = distmat * mask.float()  # dist 的 形状是 [batch_size, num_classes]
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size  # loss 是标量

        """
        掩码用于只选择每个样本到其真实类别中心的距离
        计算得到的dist形式大概是这样的：
            tensor([[1.0, 0.0, 0.0, 0.0],   # 只保留样本0到类别0的距离
            [0.0, 1.5, 0.0, 0.0],   # 只保留样本1到类别1的距离  
            [0.0, 0.0, 1.0, 0.0]])  # 只保留样本2到类别2的距离
        loss是首先对dist的所有元素求和，然后除以batch_size
        clamp是用来限制dist数值的，防止极值的影响
        """

        return loss


class CombinedLoss(nn.Module):
    """组合损失函数：结合多种损失函数"""

    def __init__(self, num_classes, feature_dim, use_focal=True, use_label_smoothing=True,
                 use_center_loss=True, focal_alpha=0.25, focal_gamma=2.0,
                 label_smoothing_factor=0.1, center_loss_weight=0.003,
                 class_weights=None):
        super(CombinedLoss, self).__init__()

        self.use_focal = use_focal
        self.use_label_smoothing = use_label_smoothing
        self.use_center_loss = use_center_loss

        # 初始化各种损失函数
        if use_focal:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights)
        else:
            if class_weights is not None:
                self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.ce_loss = nn.CrossEntropyLoss()

        if use_label_smoothing:
            self.label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing_factor)

        if use_center_loss:
            self.center_loss = CenterLoss(num_classes, feature_dim)

        self.center_loss_weight = center_loss_weight

    def forward(self, logits, features, targets):
        total_loss = 0
        loss_dict = {}

        # 主要分类损失
        if self.use_focal:
            if self.use_label_smoothing:
                main_loss = self.label_smoothing_loss(logits, targets)
            else:
                main_loss = self.focal_loss(logits, targets)
        else:
            if self.use_label_smoothing:
                main_loss = self.label_smoothing_loss(logits, targets)
            else:
                main_loss = self.ce_loss(logits, targets)

        total_loss += main_loss
        loss_dict['main_loss'] = main_loss.item()

        # Center Loss
        if self.use_center_loss:
            center_loss = self.center_loss(features, targets)
            total_loss += self.center_loss_weight * center_loss
            loss_dict['center_loss'] = center_loss.item()

        return total_loss, loss_dict


# ==================== P模型定义 ====================
class FullPModel(nn.Module):
    """
    这是一个双分支的神经网络模型，专门设计用于处理频谱和包络两种特征，适用于音频信号分类任务
    频谱特征（spec）：(B, 7, 48)  包络特征（env）：(B, 5, 32)
    """

    def __init__(self, n_spec_bands, n_env_centers, signal_length, num_classes=6, dropout=0.4):
        # signal_length 在此模型中未使用，但保留签名以兼容
        super().__init__()

        # 频谱特征分支
        self.spec_branch = nn.Sequential(
            nn.Conv1d(n_spec_bands, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(dropout)
        )
        """
        输入: [batch, n_spec_bands (7), spec_length (48)]
            → Conv1d → [batch, 16, 48] 
            → Conv1d → [batch, 32, 48]
            → AdaptiveAvgPool1d → [batch, 32, 8]
            → view → [batch, 32×8 = 256]
        """

        # 包络特征分支
        self.env_branch = nn.Sequential(
            nn.Conv1d(n_env_centers, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(dropout)
        )
        """
        输入: [batch, n_env_centers (5), env_length (32)]
            → Conv1d → [batch, 16, 32]
            → Conv1d → [batch, 32, 32]  
            → AdaptiveAvgPool1d → [batch, 32, 8]
            → view → [batch, 32×8 = 256]
        """

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        """
        频谱特征: [batch, 256]
        包络特征: [batch, 256]
        拼接: [batch, 512]
            → Linear → [batch, 128]
            → Linear → [batch, 64]  # 最终特征向量
        """

        # 分类器
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, spec, env, return_features=False):
        """
        spec: 频谱数据，形状 [B, 7, 48]
        env: 包络数据，形状 [B, 5, 32]
        """
        # 频谱分支处理
        spec_out = self.spec_branch(spec)  # 输入: [B, 7, 48]
        spec_out = spec_out.view(spec.size(0), -1)  # 输出: [B, 256]
        """
        spec_out 是经过频谱分支处理后的特征张量
        输入: spec = [B, 7, 48]
            ↓ 经过spec_branch的每一层
        第1层Conv1d: [B, 7, 48] → [B, 16, 48]
        第2层Conv1d: [B, 16, 48] → [B, 32, 48]
        AdaptiveAvgPool1d: [B, 32, 48] → [B, 32, 8]
        输出: spec_out = [B, 32, 8]

        view() 是PyTorch中用于改变张量形状的函数
        spec.size(0) = B（批量大小）
        -1 = "自动计算这个维度的大小"
        原始spec_out形状: [B, 32, 8]
        想要的新形状: [B, ?]
        自动计算: 32 × 8 = 256
        所以: [B, 32, 8] → [B, 256]
        """

        # 包络分支处理
        env_out = self.env_branch(env)  # 输入: [B, 5, 32]
        env_out = env_out.view(env.size(0), -1)  # 输出: [B, 256]

        # 特征拼接
        combined = torch.cat([spec_out, env_out], dim=1)  # 输出: [B, 512]
        """
        样本1: [频谱256个值] + [包络256个值] = [512个值]
        ...
        样本B: [频谱256个值] + [包络256个值] = [512个值]
        """

        # 特征融合
        features = self.feature_fusion(combined)  # 输出: [B, 64]

        # 分类
        logits = self.classifier(features)  # 输出: [B, num_classes]

        """
        卷积操作：
        时间点:   1   2   3
        频带1: [○, ○, ○] × [w₁₁, w₁₂, w₁₃]
        频带2: [○, ○, ○] × [w₂₁, w₂₂, w₂₃]
        ...
        频带7: [○, ○, ○] × [w₇₁, w₇₂, w₇₃]
        输出 = (所有乘积之和) + 偏置

        然后滑动到下一个位置

        时间点:   2   3   4
        频带1: [○, ○, ○] × [w₁₁, w₁₂, w₁₃]
        频带2: [○, ○, ○] × [w₂₁, w₂₂, w₂₃]
        ...
        频带7: [○, ○, ○] × [w₇₁, w₇₂, w₇₃]
        输出 = (所有乘积之和) + 偏置

        输出通道数量是16，那么就有16个卷积核去做特征提取
        输出长度 = (输入长度 + 2×padding - kernel_size) / stride + 1，stride默认是1
        padding = (kernel_size - 1) // 2 保持输入输出长度相同
        """

        if return_features:
            return logits, features
        else:
            return logits


# ==================== Optuna优化函数（*** 修改版 ***）====================
def objective_cv(trial, augmented_folds, n_spec_bands, n_env_centers,
                 num_classes, device, optimization_dir):
    """
    *** 修改版 ***
    Optuna优化目标函数
    不再使用 StratifiedKFold，而是遍历传入的 augmented_folds 列表
    """

    # 超参数搜索空间
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # 优化器选择
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])

    # 学习率调度器参数
    scheduler_factor = trial.suggest_float('scheduler_factor', 0.3, 0.8)
    scheduler_patience = trial.suggest_int('scheduler_patience', 5, 20)

    # 梯度裁剪
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 2.0)

    # 数据增强参数
    noise_factor = trial.suggest_float('noise_factor', 0.005, 0.05)
    scale_factor = trial.suggest_float('scale_factor', 0.05, 0.3)

    # 损失函数选择
    use_focal_loss = trial.suggest_categorical('use_focal_loss', [True, False])
    use_label_smoothing = trial.suggest_categorical('use_label_smoothing', [True, False])
    use_center_loss = trial.suggest_categorical('use_center_loss', [True, False])

    # 损失函数参数
    focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.5) if use_focal_loss else 0.25
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0) if use_focal_loss else 2.0
    label_smoothing_factor = trial.suggest_float('label_smoothing_factor', 0.05, 0.2) if use_label_smoothing else 0.1
    center_loss_weight = trial.suggest_float('center_loss_weight', 0.001, 0.01, log=True) if use_center_loss else 0.003

    try:
        # *** 修改点: 不再使用 SKF，而是遍历传入的 folds ***
        fold_accuracies = []
        best_fold_acc = 0
        best_model_state = None
        n_folds = len(augmented_folds)

        print(f"\n  Trial {trial.number}: 开始 {n_folds} 折交叉验证 (使用预划分数据)...")

        for fold_idx, fold_data in enumerate(augmented_folds):
            # *** 修改点: 直接从 fold_data 加载数据 ***
            X_spec_train = fold_data['train_spec']
            X_env_train = fold_data['train_env']
            y_train = fold_data['train_labels']

            X_spec_val = fold_data['val_spec']
            X_env_val = fold_data['val_env']
            y_val = fold_data['val_labels']

            # 归一化 (基于当前折的训练数据)
            spec_mean = X_spec_train.mean(axis=(0, 2), keepdims=True)
            spec_std = X_spec_train.std(axis=(0, 2), keepdims=True) + 1e-8
            X_spec_train_norm = (X_spec_train - spec_mean) / spec_std
            X_spec_val_norm = (X_spec_val - spec_mean) / spec_std

            env_mean = X_env_train.mean(axis=(0, 2), keepdims=True)
            env_std = X_env_train.std(axis=(0, 2), keepdims=True) + 1e-8
            X_env_train_norm = (X_env_train - env_mean) / env_std
            X_env_val_norm = (X_env_val - env_mean) / env_std

            # 计算当前折的类别权重
            class_weights = calculate_class_weights(y_train)

            # 创建模型
            model = FullPModel(
                n_spec_bands=n_spec_bands,
                n_env_centers=n_env_centers,
                signal_length=0,  # 不重要
                num_classes=num_classes,
                dropout=dropout_rate
            ).to(device)

            # 创建数据加载器
            train_dataset = TensorDataset(
                torch.FloatTensor(X_spec_train_norm),
                torch.FloatTensor(X_env_train_norm),
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_spec_val_norm),
                torch.FloatTensor(X_env_val_norm),
                torch.LongTensor(y_val)
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 创建损失函数
            criterion = CombinedLoss(
                num_classes=num_classes,
                feature_dim=64,  # FullPModel的最终特征维度
                use_focal=use_focal_loss,
                use_label_smoothing=use_label_smoothing,
                use_center_loss=use_center_loss,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                label_smoothing_factor=label_smoothing_factor,
                center_loss_weight=center_loss_weight,
                class_weights=class_weights.to(device) if torch.cuda.is_available() else class_weights
            )

            # 创建优化器
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:  # SGD
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
            )

            # 训练当前折
            fold_best_acc = 0
            # *** 修改点: 调优阶段使用更少的 Epochs, e.g., 50 ***
            # (原代码为 min(50, NUM_EPOCHS), 保持一致)
            optuna_epochs = min(50, NUM_EPOCHS)
            for epoch in range(optuna_epochs):
                # 训练阶段
                model.train()
                for spec_batch, env_batch, y_batch in train_loader:
                    spec_batch = spec_batch.to(device)
                    env_batch = env_batch.to(device)
                    y_batch = y_batch.to(device)

                    # 数据增强
                    if np.random.random() < 0.5:
                        spec_batch = spec_batch + torch.randn_like(spec_batch) * noise_factor
                        env_batch = env_batch + torch.randn_like(env_batch) * noise_factor
                    if np.random.random() < 0.3:
                        scale = torch.rand(spec_batch.size(0), 1, 1).to(device) * scale_factor + (1 - scale_factor / 2)
                        spec_batch = spec_batch * scale
                        env_batch = env_batch * scale

                    optimizer.zero_grad()
                    outputs, features = model(spec_batch, env_batch, return_features=True)
                    loss, _ = criterion(outputs, features, y_batch)
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                    optimizer.step()

                # 验证阶段
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for spec_batch, env_batch, y_batch in val_loader:
                        spec_batch = spec_batch.to(device)
                        env_batch = env_batch.to(device)
                        y_batch = y_batch.to(device)

                        outputs, features = model(spec_batch, env_batch, return_features=True)
                        loss, _ = criterion(outputs, features, y_batch)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()

                val_acc = 100 * val_correct / val_total
                scheduler.step(val_loss)

                if val_acc > fold_best_acc:
                    fold_best_acc = val_acc

            # 报告中间结果 (用于 Pruning)
            trial.report(fold_best_acc, fold_idx)
            if trial.should_prune():
                print(f"    折 {fold_idx + 1}/{n_folds}: {fold_best_acc:.2f}% (Trial Pruned)")
                raise optuna.exceptions.TrialPruned()

            fold_accuracies.append(fold_best_acc)
            print(f"    折 {fold_idx + 1}/{n_folds}: {fold_best_acc:.2f}%")

            # 保存最佳折的模型
            if fold_best_acc > best_fold_acc:
                best_fold_acc = fold_best_acc
                best_model_state = model.state_dict().copy()

        # 计算平均准确率
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        print(f"  Trial {trial.number} 结果: 平均={mean_acc:.2f}% (±{std_acc:.2f}%), 最佳={best_fold_acc:.2f}%")

        # 保存最佳折的模型
        trial_model_path = f"{optimization_dir}/trial_{trial.number}_model.pth"
        if best_model_state:
            torch.save(best_model_state, trial_model_path)

        return mean_acc

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def hyperparameter_optimization(augmented_folds, num_classes, device):
    """
    *** 修改版 ***
    超参数优化主函数
    接收 augmented_folds 列表
    """
    print("\n" + "=" * 60)
    print("🔍 开始超参数优化（使用预划分的5折数据）")
    print("=" * 60)

    # *** 修改点: 从 augmented_folds 获取数据信息 ***
    # 加载数据（使用全部训练数据进行交叉验证）
    try:
        sample_fold = augmented_folds[0]
        sample_spec = sample_fold['train_spec']
        sample_env = sample_fold['train_env']
        n_folds = len(augmented_folds)
    except (IndexError, KeyError) as e:
        print(f"错误: 'augmented_folds' 结构不正确或为空: {e}")
        return None, ""

    # 模型参数
    n_spec_bands = sample_spec.shape[1]
    n_env_centers = sample_env.shape[1]
    # signal_length 不再需要

    print(f"\n数据集信息:")
    print(f"  交叉验证折数: {n_folds} (来自 pkl 文件)")
    print(f"  频谱通道数: {n_spec_bands}")
    print(f"  包络通道数: {n_env_centers}")
    print(f"  类别数: {num_classes}")

    # 创建优化结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # *** 修改点: 使用相对路径 ***
    optimization_dir = f"./hyperparameter_optimization_{timestamp}"
    Path(optimization_dir).mkdir(parents=True, exist_ok=True)

    # 创建Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # 全局最佳模型追踪
    best_trial_number = [None]  # 使用列表以便在回调函数中修改

    # 定义回调函数：追踪并保存全局最佳模型
    def save_best_model_callback(study, trial):
        if study.best_trial.number == trial.number:
            # 当前试验是新的最佳试验
            best_trial_number[0] = trial.number
            trial_model_path = f"{optimization_dir}/trial_{trial.number}_model.pth"
            best_model_path = f"{optimization_dir}/best_model.pth"

            # 复制当前试验的模型为全局最佳模型
            if os.path.exists(trial_model_path):
                shutil.copy2(trial_model_path, best_model_path)
                print(f"\n✨ 新的最佳试验 #{trial.number}: 验证准确率 = {trial.value:.2f}%")

    # 定义目标函数
    def objective_wrapper(trial):
        # *** 修改点: 传递 augmented_folds 列表 ***
        return objective_cv(
            trial, augmented_folds,
            n_spec_bands, n_env_centers, num_classes, device, optimization_dir
        )

    # 开始优化
    print(f"\n开始优化 (共{N_TRIALS}次试验，每次{n_folds}折交叉验证)...")
    study.optimize(
        objective_wrapper,
        n_trials=N_TRIALS,
        timeout=OPTIMIZATION_TIMEOUT,
        show_progress_bar=True,
        callbacks=[save_best_model_callback]
    )

    print(f"\n✅ 优化完成!")
    print(f"最佳试验: {study.best_trial.number}")
    print(f"最佳平均验证准确率（{n_folds}折交叉验证）: {study.best_value:.2f}%")
    print(f"\n最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存最佳参数
    with open(f"{optimization_dir}/best_params.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)

    # 保存所有试验结果
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        })

    with open(f"{optimization_dir}/all_trials.json", 'w') as f:
        json.dump(trials_data, f, indent=2)

    # 保存模型架构信息
    model_info = {
        'best_trial_number': study.best_trial.number,
        'best_accuracy': study.best_value,
        'optimization_method': f'{n_folds}-fold cross-validation (pre-folded)',
        'n_folds': n_folds,
        'n_spec_bands': n_spec_bands,
        'n_env_centers': n_env_centers,
        'num_classes': num_classes,
        'model_architecture': 'FullPModel'
    }

    with open(f"{optimization_dir}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    # 保存优化历史图
    try:
        if len(study.trials) > 0:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(f"{optimization_dir}/optimization_history.png")

            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(f"{optimization_dir}/param_importances.png")
    except Exception as e:
        print(f"无法生成优化可视化图表: {e}")

    # 清理非最佳试验的模型文件（节省空间）
    print(f"\n清理中间模型文件...")
    cleaned_count = 0
    for trial in study.trials:
        if trial.number != study.best_trial.number:
            trial_model_path = f"{optimization_dir}/trial_{trial.number}_model.pth"
            if os.path.exists(trial_model_path):
                os.remove(trial_model_path)
                cleaned_count += 1

    print(f"已删除 {cleaned_count} 个非最佳模型文件")

    print(f"\n优化结果已保存到: {optimization_dir}")
    print(f"  - best_model.pth: 全局最佳模型权重 (试验 #{study.best_trial.number}, 预划分 {n_folds} 折CV中的最佳折)")
    print(f"  - best_params.json: 最佳超参数")
    print(f"  - model_info.json: 模型架构信息")
    print(f"  - all_trials.json: 所有试验记录")
    print(f"\n说明: 模型权重来自最佳试验中准确率最高的折")

    return study.best_params, optimization_dir


# ==================== 训练函数 ====================
def train_model(dataset, num_classes, save_dir, best_params=None):
    """
    训练模型
    *** 修改点: 接收 num_classes, 并创建 id_to_label ***
    """
    print("\n" + "=" * 60)
    print("训练最终模型")
    print("=" * 60)

    # 使用最佳参数或默认参数
    if best_params:
        print("\n使用优化后的最佳参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        learning_rate = best_params.get('learning_rate', LEARNING_RATE)
        batch_size = best_params.get('batch_size', BATCH_SIZE)
        dropout_rate = best_params.get('dropout_rate', DROPOUT_RATE)
        weight_decay = best_params.get('weight_decay', 0.0)
        optimizer_name = best_params.get('optimizer', 'Adam')
        scheduler_factor = best_params.get('scheduler_factor', 0.5)
        scheduler_patience = best_params.get('scheduler_patience', 10)
        max_grad_norm = best_params.get('max_grad_norm', 1.0)
        noise_factor = best_params.get('noise_factor', 0.01)
        scale_factor = best_params.get('scale_factor', 0.1)

        # 损失函数参数
        use_focal_loss = best_params.get('use_focal_loss', False)
        use_label_smoothing = best_params.get('use_label_smoothing', False)
        use_center_loss = best_params.get('use_center_loss', False)
        focal_alpha = best_params.get('focal_alpha', 0.25)
        focal_gamma = best_params.get('focal_gamma', 2.0)
        label_smoothing_factor = best_params.get('label_smoothing_factor', 0.1)
        center_loss_weight = best_params.get('center_loss_weight', 0.003)
    else:
        print("\n使用默认参数")
        learning_rate = LEARNING_RATE
        batch_size = BATCH_SIZE
        dropout_rate = DROPOUT_RATE
        weight_decay = 0.0
        optimizer_name = 'Adam'
        scheduler_factor = 0.5
        scheduler_patience = 10
        max_grad_norm = 1.0
        noise_factor = 0.01
        scale_factor = 0.1

        use_focal_loss = False
        use_label_smoothing = False
        use_center_loss = False
        focal_alpha = 0.25
        focal_gamma = 2.0
        label_smoothing_factor = 0.1
        center_loss_weight = 0.003

    # 加载数据
    spec_train_all = dataset['x_train']['spec']
    env_train_all = dataset['x_train']['env']
    labels_all = dataset['y_train']

    # *** 修改点: 创建 id_to_label ***
    id_to_label = dataset.get('id_to_label', {i: f'Class_{i}' for i in range(num_classes)})

    print(f"\n数据集信息:")
    print(f"  训练集频谱特征: {spec_train_all.shape}")
    print(f"  训练集包络特征: {env_train_all.shape}")
    print(f"  训练集标签: {labels_all.shape}")
    print(f"  类别数: {num_classes}")

    # 划分训练验证集
    X_spec_train, X_spec_val, X_env_train, X_env_val, y_train, y_val = train_test_split(
        spec_train_all, env_train_all, labels_all, test_size=0.2, random_state=42, stratify=labels_all
    )

    print(f"\n训练验证集划分 (80/20 split):")
    print(f"  训练集: {len(y_train)} 样本")
    print(f"  验证集: {len(y_val)} 样本")

    # 标签分布
    print(f"\n训练集标签分布:")
    for label_id in range(num_classes):
        count = np.sum(y_train == label_id)
        print(f"  类别 {label_id} ({id_to_label.get(label_id, 'N/A')}): {count} 样本")

    # 归一化
    spec_mean = X_spec_train.mean(axis=(0, 2), keepdims=True)
    spec_std = X_spec_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_spec_train_norm = (X_spec_train - spec_mean) / spec_std
    X_spec_val_norm = (X_spec_val - spec_mean) / spec_std

    env_mean = X_env_train.mean(axis=(0, 2), keepdims=True)
    env_std = X_env_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_env_train_norm = (X_env_train - env_mean) / env_std
    X_env_val_norm = (X_env_val - env_mean) / env_std

    # 保存归一化参数
    norm_params = {
        'spec_mean': spec_mean,
        'spec_std': spec_std,
        'env_mean': env_mean,
        'env_std': env_std
    }

    # 计算类别权重
    class_weights = calculate_class_weights(y_train)

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_spec_train_norm),
        torch.FloatTensor(X_env_train_norm),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_spec_val_norm),
        torch.FloatTensor(X_env_val_norm),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    n_spec_bands = spec_train_all.shape[1]
    n_env_centers = env_train_all.shape[1]
    # signal_length = 0 # 不重要

    model = FullPModel(
        n_spec_bands=n_spec_bands,
        n_env_centers=n_env_centers,
        signal_length=0,  # 不重要
        num_classes=num_classes,
        dropout=dropout_rate
    ).to(device)

    print(f"\n模型架构:")
    print(f"  频谱通道数: {n_spec_bands}")
    print(f"  包络通道数: {n_env_centers}")
    print(f"  Dropout率: {dropout_rate}")

    # 损失函数和优化器
    criterion = CombinedLoss(
        num_classes=num_classes,
        feature_dim=64,
        use_focal=use_focal_loss,
        use_label_smoothing=use_label_smoothing,
        use_center_loss=use_center_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        label_smoothing_factor=label_smoothing_factor,
        center_loss_weight=center_loss_weight,
        class_weights=class_weights.to(device) if torch.cuda.is_available() else class_weights
    )

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)

    # 训练循环
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    patience_counter = 0

    print(f"\n开始训练 (共{NUM_EPOCHS}轮)...")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for spec_batch, env_batch, y_batch in train_loader:
            spec_batch, env_batch, y_batch = spec_batch.to(device), env_batch.to(device), y_batch.to(device)

            # 数据增强
            if np.random.random() < 0.5:
                spec_batch = spec_batch + torch.randn_like(spec_batch) * noise_factor
                env_batch = env_batch + torch.randn_like(env_batch) * noise_factor
            if np.random.random() < 0.3:
                scale = torch.rand(spec_batch.size(0), 1, 1).to(device) * scale_factor + (1 - scale_factor / 2)
                spec_batch = spec_batch * scale
                env_batch = env_batch * scale

            optimizer.zero_grad()
            outputs, features = model(spec_batch, env_batch, return_features=True)
            loss, _ = criterion(outputs, features, y_batch)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for spec_batch, env_batch, y_batch in val_loader:
                spec_batch, env_batch, y_batch = spec_batch.to(device), env_batch.to(device), y_batch.to(device)

                outputs, features = model(spec_batch, env_batch, return_features=True)
                loss, _ = criterion(outputs, features, y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 每5轮打印一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1:3d}/{NUM_EPOCHS}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        scheduler.step(val_loss)

        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n早停于第 {epoch + 1} 轮")
            break

    print("=" * 60)

    # 加载最佳模型
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))

    # 最终评估
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for spec_batch, env_batch, y_batch in val_loader:
            spec_batch, env_batch = spec_batch.to(device), env_batch.to(device)
            outputs = model(spec_batch, env_batch, return_features=False)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 打印结果
    print(f"\n{'=' * 60}")
    print("训练完成！")
    print(f"{'=' * 60}")
    print(f"\n最佳验证集准确率: {best_val_acc:.2f}%")

    print(f"\n分类报告:")
    class_names = [id_to_label.get(i, f'Class_{i}') for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 保存结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'norm_params': norm_params,
        'y_true': y_true,
        'y_pred': y_pred,
        'class_names': class_names
    }

    with open(f"{save_dir}/training_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    # 绘制训练曲线
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(train_accs, label='Train Acc', linewidth=2)
        axes[1].plot(val_accs, label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"无法绘制训练曲线: {e}")

    # 绘制混淆矩阵
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=13)
        plt.ylabel('True Label', fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"无法绘制混淆矩阵: {e}")

    print(f"\n结果已保存到: {save_dir}")
    print(f"  - best_model.pth: 最佳模型")
    print(f"  - training_results.pkl: 训练结果")
    print(f"  - training_history.png: 训练曲线")
    print(f"  - confusion_matrix.png: 混淆矩阵")

    # *** 修改点: 返回 norm_params ***
    return model, results, norm_params


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("使用优化数据集训练模型 (支持5折交叉验证超参数优化)")
    print("=" * 60)

    # 检查数据集路径
    if not OPTIMIZED_DATASET_PATH or not os.path.exists(OPTIMIZED_DATASET_PATH):
        print(f"\n错误: 请设置正确的数据集路径")
        print(f"当前路径: {OPTIMIZED_DATASET_PATH}")
        print(f"请确保 'augmented_dataset_5fold.pkl' 位于此路径")
        return None, None

    # 加载数据集
    print(f"\n加载数据集: {OPTIMIZED_DATASET_PATH}")
    try:
        with open(OPTIMIZED_DATASET_PATH, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"加载 pkl 文件失败: {e}")
        return None, None

    # --- *** 修改点: 从新 'data' 结构中提取数据 *** ---
    try:
        augmented_folds = data['augmented_folds']
        x_test_data = data['x_test']
        num_classes = data['num_classes']
        n_folds = data['n_splits']

        print(f"\n已从 '{OPTIMIZED_DATASET_PATH}' 加载数据:")
        print(f"  发现 {n_folds} 折预划分数据用于调优。")
        print(f"  发现 {len(x_test_data['spec'])} 条全局测试样本。")
        print(f"  类别数: {num_classes}")

    except (KeyError, IndexError, TypeError) as e:
        print(f"\n错误: 'augmented_dataset_5fold.pkl' 文件结构不正确。")
        print(f"  需要 'augmented_folds' (列表), 'x_test' (字典), 'num_classes' (整数).")
        print(f"  错误详情: {e}")
        return None, None
    # --- *** 修改点结束 *** ---

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 超参数优化
    best_params = None
    optimization_dir = ""  # 初始化
    if ENABLE_HYPERPARAMETER_TUNING:
        print(f"\n启用超参数优化...")
        # *** 修改点: 传递 augmented_folds 列表 ***
        best_params, optimization_dir = hyperparameter_optimization(augmented_folds, num_classes, device)
        print(f"\n超参数优化完成，最佳参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n跳过超参数优化，使用默认参数")

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # *** 修改点: 使用相对路径 ***
    save_dir = f"./training_results_{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 保存最佳参数（如果有）
    if best_params:
        with open(f"{save_dir}/best_hyperparams.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n最佳超参数已保存: {save_dir}/best_hyperparams.json")

    # --- *** 修改点: 准备 'train_model' 的数据 *** ---
    # 我们将使用第1折(索引0)的增强数据作为 "总训练集"
    # 'train_model' 将在这个数据上进行自己的 80/20 划分
    print(f"\n准备 'train_model' 的数据 (使用第 1 折的增强数据)...")
    fold_1_data = augmented_folds[0]

    # 重新构建 'dataset' 字典，以适应 'train_model' 函数的格式
    dataset_for_final_train = {
        'x_train': {'spec': fold_1_data['train_spec'], 'env': fold_1_data['train_env']},
        'y_train': fold_1_data['train_labels'],
        # 创建标签映射
        'id_to_label': {i: f'Class_{i}' for i in range(num_classes)},
        'label_map': {f'Class_{i}': i for i in range(num_classes)}
    }

    # 训练最终模型
    model, results, norm_params = train_model(dataset_for_final_train, num_classes, save_dir, best_params)

    # --- *** 新增: 在全局测试集上进行预测 *** ---
    print(f"\n{'=' * 60}")
    print("🚀 开始在全局测试集上进行最终预测...")
    print(f"{'=' * 60}")

    # 加载 'train_model' 保存的最佳模型
    best_model_path = f"{save_dir}/best_model.pth"
    if not os.path.exists(best_model_path):
        print(f"错误: 找不到训练好的模型 {best_model_path}")
        return model, results

    # 归一化参数已从 train_model 返回
    spec_mean = norm_params['spec_mean']
    spec_std = norm_params['spec_std']
    env_mean = norm_params['env_mean']
    env_std = norm_params['env_std']

    # 加载测试数据
    x_test_spec = data['x_test']['spec']
    x_test_env = data['x_test']['env']

    # 归一化测试数据
    x_test_spec_norm = (x_test_spec - spec_mean) / spec_std
    x_test_env_norm = (x_test_env - env_mean) / env_std

    # 创建测试数据加载器
    test_dataset = TensorDataset(
        torch.FloatTensor(x_test_spec_norm),
        torch.FloatTensor(x_test_env_norm)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    n_spec_bands = dataset_for_final_train['x_train']['spec'].shape[1]
    n_env_centers = dataset_for_final_train['x_train']['env'].shape[1]

    # 使用 'best_params' (如果存在)
    final_dropout = best_params.get('dropout_rate', DROPOUT_RATE) if best_params else DROPOUT_RATE

    model = FullPModel(
        n_spec_bands=n_spec_bands,
        n_env_centers=n_env_centers,
        signal_length=0,  # 不重要
        num_classes=num_classes,
        dropout=final_dropout
    ).to(device)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print("已成功加载最佳模型和归一化参数。")

    # 开始预测
    test_preds = []
    with torch.no_grad():
        for spec_batch, env_batch in test_loader:
            spec_batch = spec_batch.to(device)
            env_batch = env_batch.to(device)

            outputs = model(spec_batch, env_batch, return_features=False)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())

    test_preds = np.array(test_preds)

    # 保存预测结果
    pred_save_path = f"{save_dir}/test_predictions.npy"
    np.save(pred_save_path, test_preds)

    # 同时保存为 .txt (如果需要)
    pred_txt_path = f"{save_dir}/test_predictions.txt"
    np.savetxt(pred_txt_path, test_preds, fmt='%d')

    print(f"\n预测完成! {len(test_preds)} 条预测结果已保存。")
    print(f"  - {pred_save_path}")
    print(f"  - {pred_txt_path}")

    # 打印一些预测示例
    print(f"\n预测结果示例 (前20条):")
    print(test_preds[:20])

    print(f"\n{'=' * 60}")
    print("✅ 任务全部完成!")
    print(f"{'=' * 60}")
    print(f"\n最终模型和训练结果保存位置: {save_dir}")
    if best_params:
        print(f"超参数优化结果 (基于预划分5折数据): {optimization_dir}")

    return model, results


if __name__ == "__main__":
    model, results = main()