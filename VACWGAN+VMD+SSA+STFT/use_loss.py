# ===================================================================================
# 文件名: use_loss.py
# 描述: 组合损失函数 (Focal + Label Smoothing + Center Loss)
# 核心改动: CombinedLoss 中的 feature_dim 需与 Simple2DCNN 匹配
# ===================================================================================
# 核心逻辑不变，但为了运行正确，CombinedLoss 中 CenterLoss 的 feature_dim 
# 应为 Simple2DCNN 的 fc1 输出维度 128。

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np


def calculate_class_weights(y_train):
    """计算类别权重用于处理类别不平衡"""
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    num_classes = len(class_counts)

    # 计算每个类别的权重（逆频率）
    class_weights = {}
    for class_id, count in class_counts.items():
        # 这里使用常见的逆频率加权
        class_weights[class_id] = total_samples / (num_classes * count)
        # 归一化（可选，但通常不必）

    # 转换为tensor
    weights = torch.zeros(num_classes)
    sorted_class_ids = sorted(class_counts.keys())
    for i, class_id in enumerate(sorted_class_ids):
        weights[i] = class_weights[class_id]

    return weights


class FocalLoss(nn.Module):
    # ... (FocalLoss 实现不变)
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # 应用类别权重
        if self.class_weights is not None:
            # 确保 targets 索引正确
            weights = self.class_weights[targets]
            ce_loss = ce_loss * weights

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    # ... (LabelSmoothingCrossEntropy 实现不变)
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


class CenterLoss(nn.Module):
    """Center Loss用于增强特征的类内紧密度"""

    def __init__(self, num_classes, feature_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu

        if self.use_gpu and torch.cuda.is_available():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu and torch.cuda.is_available():
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

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
        self.class_weights = class_weights  # 存储类别权重

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
            # feature_dim 必须是 128 (fc1层的输出维度)
            self.center_loss = CenterLoss(num_classes, feature_dim)

        self.center_loss_weight = center_loss_weight

    def forward(self, logits, features, targets):
        total_loss = 0
        loss_dict = {}

        # 1. 主要分类损失 (CE/Focal/LS)
        if self.use_label_smoothing:
            # 标签平滑交叉熵不直接支持类别权重，但可以与 FocalLoss 或 CE 配合使用（此处简化为只用 LS）
            main_loss = self.label_smoothing_loss(logits, targets)
        elif self.use_focal:
            main_loss = self.focal_loss(logits, targets)
        else:
            main_loss = self.ce_loss(logits, targets)

        total_loss += main_loss
        loss_dict['main_loss'] = main_loss.item()

        # 2. Center Loss
        if self.use_center_loss:
            center_loss = self.center_loss(features, targets)
            total_loss += self.center_loss_weight * center_loss
            loss_dict['center_loss'] = center_loss.item()

        return total_loss, loss_dict