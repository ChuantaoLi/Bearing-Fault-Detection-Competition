"""
1. Optuna 启动一次试验。
2. 试验调用 cross_validate 函数开始 5 折交叉验证。
3. Fold 1:
   a. 划分 Fold 1 的 train_data (80%) 和 val_data (20%)。
   b. 归一化 train_data 和 val_data。
   c. 启动 VACWGAN 模块进行数据增强:
      i.   加载 Fold 1 专属的 GAN 参数。
      ii.  在 Fold 1 的 train_data 上训练 GAN。
      iii. 生成 (syn_spec, syn_env) 样本以平衡 train_data。
      iv.  返回增强后的 aug_train_data。
   d. 分类器训练:
      i.   使用 aug_train_data 训练 MultiFeatureFusionModel。
      ii.  使用 val_data 评估模型，得到 acc_1。
5. Fold 2:
   a. ...
   c. [数据增强]:
      i.   加载 Fold 2 专属的 GAN 参数。
   ...
6. Fold 5: 重复...
7. 计算 5 折的 mean_acc，将其作为 Optuna 的评估分数。
8. TopKTracker 跟踪并保存最佳模型。
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from datetime import datetime
import json
import optuna
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

"""配置参数"""
DATASET_PATH = "optimized_features_dataset.pkl"

# 优化配置
N_TRIALS = 50  # Optuna 试验次数
N_FOLDS = 5  # 交叉验证折数
TRAIN_EPOCHS = 100  # 分类器训练轮数
EARLY_STOPPING_PATIENCE = 10  # 早停幅度
TOP_K_MODELS = 5  # 保存Top-5模型

# 数据增强配置
GAN_EPOCHS = 100  # 每一折训练GAN的轮数
FOLD_SPECIFIC_GAN_PARAMS = [
    # --- Fold 1 (索引 0) ---
    {'latent_dim': 96, 'lr': 0.0008809783391095071, 'd_iters': 4, 'lambda_cls': 7.2824335524798745, 'lambda_kl': 1.4669130637900507},

    # --- Fold 2 (索引 1) ---
    {'latent_dim': 192, 'lr': 9.333635381796446e-05, 'd_iters': 7, 'lambda_cls': 19.950858751074374, 'lambda_kl': 1.0697610228678371},

    # --- Fold 3 (索引 2) ---
    {'latent_dim': 160, 'lr': 0.0002427050090622415, 'd_iters': 3, 'lambda_cls': 9.898693858628008, 'lambda_kl': 1.633001343548243},

    # --- Fold 4 (索引 3) ---
    {'latent_dim': 192, 'lr': 9.792715480227e-05, 'd_iters': 3, 'lambda_cls': 13.785436683742514, 'lambda_kl': 1.3301469032381479},

    # --- Fold 5 (索引 4) ---
    {'latent_dim': 160, 'lr': 0.00017586959251670548, 'd_iters': 5, 'lambda_cls': 8.664599779776182, 'lambda_kl': 0.33742629946541564}
]

"""第 1 部分: VACWGAN 数据增强模块"""


class MultiModalDataset(Dataset):
    """多模态数据集"""

    def __init__(self, x_spec, x_env, y):
        self.x_spec = torch.FloatTensor(x_spec)
        self.x_env = torch.FloatTensor(x_env)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_spec[idx], self.x_env[idx], self.y[idx]


class MultiModalEncoder(nn.Module):
    """多模态编码器，处理 spec 和 env 两个输入"""

    def __init__(self, latent_dim=128):
        super(MultiModalEncoder, self).__init__()
        # Spec分支: (7, 48) -> feature
        self.spec_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 4, 24)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 2, 12)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 1, 6)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        # Env分支: (5, 32) -> feature
        self.env_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 3, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 2, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 1, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.spec_flat_dim = 256 * 1 * 6
        self.env_flat_dim = 256 * 1 * 4
        self.combined_dim = self.spec_flat_dim + self.env_flat_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x_spec, x_env):
        x_spec = x_spec.unsqueeze(1)
        x_env = x_env.unsqueeze(1)
        spec_feat = self.spec_branch(x_spec)
        env_feat = self.env_branch(x_env)
        spec_feat = spec_feat.view(spec_feat.size(0), -1)
        env_feat = env_feat.view(env_feat.size(0), -1)
        # L2 归一化以缓解模态主导
        spec_feat = F.normalize(spec_feat, p=2, dim=1)
        env_feat = F.normalize(env_feat, p=2, dim=1)
        combined = torch.cat([spec_feat, env_feat], dim=1)
        fused = self.fusion_fc(combined)
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        return mu, logvar


class MultiModalGenerator(nn.Module):
    """多模态生成器，生成 spec 和 env 两个输出"""

    def __init__(self, latent_dim=128, num_classes=10):
        super(MultiModalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.shared_trunk = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.spec_fc = nn.Sequential(
            nn.Linear(1024, 256 * 2 * 6),
            nn.BatchNorm1d(256 * 2 * 6),
            nn.ReLU(True),
        )
        self.spec_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.env_fc = nn.Sequential(
            nn.Linear(1024, 256 * 2 * 4),
            nn.BatchNorm1d(256 * 2 * 4),
            nn.ReLU(True),
        )
        self.env_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 4), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c_emb = self.label_embedding(labels)
        z_c = torch.cat([z, c_emb], dim=1)
        shared = self.shared_trunk(z_c)
        spec_feat = self.spec_fc(shared)
        spec_feat = spec_feat.view(-1, 256, 2, 6)
        spec_out = self.spec_deconv(spec_feat)
        spec_out = F.interpolate(spec_out, size=(7, 48), mode='bilinear', align_corners=False)
        env_feat = self.env_fc(shared)
        env_feat = env_feat.view(-1, 256, 2, 4)
        env_out = self.env_deconv(env_feat)
        env_out = F.interpolate(env_out, size=(5, 32), mode='bilinear', align_corners=False)
        return spec_out.squeeze(1), env_out.squeeze(1)


class MultiModalDiscriminator(nn.Module):
    """多模态判别器"""

    def __init__(self):
        super(MultiModalDiscriminator, self).__init__()
        self.spec_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.env_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.spec_feat_dim = 128 * 1 * 6
        self.env_feat_dim = 128 * 1 * 4
        self.adv_layer = nn.Sequential(
            nn.Linear(self.spec_feat_dim + self.env_feat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x_spec, x_env):
        x_spec = x_spec.unsqueeze(1)
        x_env = x_env.unsqueeze(1)
        spec_feat = self.spec_branch(x_spec)
        env_feat = self.env_branch(x_env)
        spec_feat = spec_feat.view(spec_feat.size(0), -1)
        env_feat = env_feat.view(env_feat.size(0), -1)
        spec_feat = F.normalize(spec_feat, p=2, dim=1)
        env_feat = F.normalize(env_feat, p=2, dim=1)
        combined = torch.cat([spec_feat, env_feat], dim=1)
        validity = self.adv_layer(combined)
        return validity


class GAN_MultiModalClassifier(nn.Module):
    """多模态分类器"""

    def __init__(self, num_classes=10):
        super(GAN_MultiModalClassifier, self).__init__()
        self.spec_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.env_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.spec_feat_dim = 128 * 1 * 6
        self.env_feat_dim = 128 * 1 * 4
        self.classification_layer = nn.Sequential(
            nn.Linear(self.spec_feat_dim + self.env_feat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_spec, x_env):
        x_spec = x_spec.unsqueeze(1)
        x_env = x_env.unsqueeze(1)
        spec_feat = self.spec_branch(x_spec)
        env_feat = self.env_branch(x_env)
        spec_feat = spec_feat.view(spec_feat.size(0), -1)
        env_feat = env_feat.view(env_feat.size(0), -1)
        spec_feat = F.normalize(spec_feat, p=2, dim=1)
        env_feat = F.normalize(env_feat, p=2, dim=1)
        combined = torch.cat([spec_feat, env_feat], dim=1)
        class_output = self.classification_layer(combined)
        return class_output


def reparameterization(mu, logvar):
    """重参数化"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def compute_gradient_penalty(D, real_spec, real_env, fake_spec, fake_env, device, gamma=10):
    """计算梯度惩罚"""
    batch_size = real_spec.size(0)
    alpha_spec = torch.rand(batch_size, 1, 1).to(device)
    alpha_env = torch.rand(batch_size, 1, 1).to(device)
    interpolates_spec = (alpha_spec * real_spec + (1 - alpha_spec) * fake_spec).requires_grad_(True)
    interpolates_env = (alpha_env * real_env + (1 - alpha_env) * fake_env).requires_grad_(True)
    d_interpolates = D(interpolates_spec, interpolates_env)
    fake = torch.ones(batch_size, 1, device=device).requires_grad_(False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=[interpolates_spec, interpolates_env],
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients_spec = gradients[0].view(batch_size, -1)
    gradients_env = gradients[1].view(batch_size, -1)
    gradients_combined = torch.cat([gradients_spec, gradients_env], dim=1)
    gradient_penalty = ((gradients_combined.norm(2, dim=1) - 1) ** 2).mean() * gamma
    return gradient_penalty


def compute_kl_loss(mu, logvar):
    """计算KL散度"""
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    return kl_loss


def train_vacwgan(data_loader, num_classes, device, params, epochs=100):
    """训练多模态 VACWGAN"""
    # 从传入的 params 字典中解包参数
    latent_dim = params['latent_dim']
    lr = params['lr']
    d_iters = params['d_iters']
    lambda_cls = params['lambda_cls']
    lambda_kl = params['lambda_kl']

    E = MultiModalEncoder(latent_dim=latent_dim).to(device)
    G = MultiModalGenerator(latent_dim=latent_dim, num_classes=num_classes).to(device)
    D = MultiModalDiscriminator().to(device)
    C = GAN_MultiModalClassifier(num_classes=num_classes).to(device)

    opt_E = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_C = optim.Adam(C.parameters(), lr=lr, betas=(0.5, 0.999))

    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        E.train()
        G.train()
        D.train()
        C.train()
        for i, (real_spec, real_env, labels) in enumerate(data_loader):
            real_spec = real_spec.to(device)
            real_env = real_env.to(device)
            labels = labels.to(device)
            opt_D.zero_grad()
            opt_C.zero_grad()
            d_real = D(real_spec, real_env)
            c_real = C(real_spec, real_env)
            loss_D_real = -torch.mean(d_real)
            loss_C_real = ce_loss(c_real, labels)
            mu, logvar = E(real_spec, real_env)
            z = reparameterization(mu, logvar)
            fake_spec, fake_env = G(z, labels)
            fake_spec_d, fake_env_d = fake_spec.detach(), fake_env.detach()
            d_fake = D(fake_spec_d, fake_env_d)
            c_fake = C(fake_spec_d, fake_env_d)
            loss_D_fake = torch.mean(d_fake)
            loss_C_fake = ce_loss(c_fake, labels)
            gp = compute_gradient_penalty(D, real_spec, real_env,
                                          fake_spec_d, fake_env_d, device)
            loss_D = loss_D_fake + loss_D_real + gp
            loss_C = loss_C_real + loss_C_fake
            loss_D.backward(retain_graph=True)
            loss_C.backward()
            opt_D.step()
            opt_C.step()
            if i % d_iters == 0:
                opt_G.zero_grad()
                opt_E.zero_grad()
                mu, logvar = E(real_spec, real_env)
                z = reparameterization(mu, logvar)
                fake_spec, fake_env = G(z, labels)
                d_fake_g = D(fake_spec, fake_env)
                c_fake_g = C(fake_spec, fake_env)
                loss_G_adv = -torch.mean(d_fake_g)
                loss_G_cls = ce_loss(c_fake_g, labels)
                loss_E_kl = compute_kl_loss(mu, logvar)
                loss_G_total = loss_G_adv + lambda_cls * loss_G_cls + lambda_kl * loss_E_kl
                loss_G_total.backward()
                opt_G.step()
                opt_E.step()
        if (epoch + 1) % 50 == 0:
            print(f"    [GAN Epoch {epoch + 1}/{epochs}] G_Loss: {loss_G_total.item():.4f}, D_Loss: {loss_D.item():.4f}")

    return E, G, D, C


def generate_synthetic_samples(E, G, x_spec_minority, x_env_minority, y_minority, num_to_generate, device):
    """生成合成样本"""
    if len(y_minority) == 0 or num_to_generate == 0:
        return np.array([]), np.array([]), np.array([])

    minority_dataset = MultiModalDataset(x_spec_minority, x_env_minority, y_minority)
    minority_loader = DataLoader(minority_dataset, batch_size=min(len(y_minority), 64), shuffle=True)

    synthetic_spec_list = []
    synthetic_env_list = []
    current_generated = 0

    E.eval()
    G.eval()

    with torch.no_grad():
        while current_generated < num_to_generate:
            for spec_inputs, env_inputs, labels in minority_loader:
                spec_inputs = spec_inputs.to(device)
                env_inputs = env_inputs.to(device)
                labels = labels.to(device)
                mu, logvar = E(spec_inputs, env_inputs)
                z = reparameterization(mu, logvar)
                fake_spec, fake_env = G(z, labels)
                fake_spec = fake_spec.cpu().numpy()
                fake_env = fake_env.cpu().numpy()
                num_batch = fake_spec.shape[0]
                if current_generated + num_batch > num_to_generate:
                    needed = num_to_generate - current_generated
                    fake_spec = fake_spec[:needed]
                    fake_env = fake_env[:needed]
                    num_batch = needed
                synthetic_spec_list.append(fake_spec)
                synthetic_env_list.append(fake_env)
                current_generated += num_batch
                if current_generated >= num_to_generate:
                    break
            if current_generated >= num_to_generate:
                break

    if not synthetic_spec_list:
        return np.array([]), np.array([]), np.array([])

    synthetic_spec = np.concatenate(synthetic_spec_list, axis=0)
    synthetic_env = np.concatenate(synthetic_env_list, axis=0)
    synthetic_y = np.full(synthetic_spec.shape[0], y_minority[0])

    return synthetic_spec, synthetic_env, synthetic_y


"""第 2 部分: 分类器与损失函数模"""


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation Block for 1D"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 4), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class CBAM1D(nn.Module):
    """Convolutional Block Attention Module for 1D"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_att(sa_input)
        x = x * sa
        return x


class SelfAttention1D(nn.Module):
    """Self-Attention for 1D sequences"""

    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.size()
        Q = self.query(x).permute(0, 2, 1)  # (B, L, C/8)
        K = self.key(x)  # (B, C/8, L)
        V = self.value(x)  # (B, C, L)
        attn = torch.bmm(Q, K)  # (B, L, L)
        attn = F.softmax(attn / (K.size(1) ** 0.5), dim=-1)
        out = torch.bmm(V, attn.permute(0, 2, 1))  # (B, C, L)
        out = self.gamma * out + x
        return out


class FlexibleBranch1D(nn.Module):
    """1D CNN分支"""

    def __init__(self, in_channels, hidden_dim, attention_type='none', dropout=0.3):
        super().__init__()
        self.attention_type = attention_type
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if attention_type == 'se':
            self.attention = SEBlock1D(hidden_dim)
        elif attention_type == 'cbam':
            self.attention = CBAM1D(hidden_dim)
        elif attention_type == 'self':
            self.attention = SelfAttention1D(hidden_dim)
        else:
            self.attention = None
        self.pool = nn.AdaptiveAvgPool1d(8)

    def forward(self, x):
        x = self.conv_layers(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.pool(x)
        return x


class MultiFeatureFusionModel(nn.Module):
    """支持多特征融合的模型"""

    def __init__(self, n_spec_channels, n_env_channels,
                 spec_hidden=32, env_hidden=32, num_classes=6,
                 spec_attention='none', env_attention='none',
                 fusion_method='concat', dropout=0.3):
        super().__init__()
        self.fusion_method = fusion_method
        self.spec_branch = FlexibleBranch1D(
            n_spec_channels, spec_hidden, spec_attention, dropout
        )
        self.env_branch = FlexibleBranch1D(
            n_env_channels, env_hidden, env_attention, dropout
        )

        if fusion_method == 'concat' or fusion_method == 'weighted':  # 'weighted' 在此实现为 concat
            fusion_dim = spec_hidden * 8 + env_hidden * 8
        elif fusion_method == 'add':
            raise NotImplementedError("Fusion method 'add' is not supported in this config")

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, spec, env, return_features=False):
        spec_out = self.spec_branch(spec).flatten(1)
        env_out = self.env_branch(env).flatten(1)

        fused = torch.cat([spec_out, env_out], dim=1)
        features = self.fusion(fused)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            ce_loss = ce_loss * weights
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵"""

    def __init__(self, smoothing=0.1, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        if self.class_weights is not None:
            loss = loss * self.class_weights[target]
        return loss.mean()


class CenterLoss(nn.Module):
    """Center Loss"""

    def __init__(self, num_classes, feature_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_classes, device=features.device).long()
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


class CombinedLoss(nn.Module):
    """组合损失函数"""

    def __init__(self, num_classes, feature_dim, device,
                 loss_type='ce', use_label_smoothing=False, use_center_loss=False,
                 focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.1,
                 center_loss_weight=0.003, class_weights=None):
        super().__init__()
        self.use_center_loss = use_center_loss
        self.center_loss_weight = center_loss_weight
        if loss_type == 'focal':
            self.main_loss = FocalLoss(focal_alpha, focal_gamma, class_weights)
        elif use_label_smoothing:
            self.main_loss = LabelSmoothingCrossEntropy(label_smoothing, class_weights)
        else:
            self.main_loss = nn.CrossEntropyLoss(weight=class_weights)
        if use_center_loss:
            self.center_loss = CenterLoss(num_classes, feature_dim, device)

    def forward(self, logits, features, targets):
        main_loss = self.main_loss(logits, targets)
        if self.use_center_loss:
            center_loss = self.center_loss(features, targets)
            total_loss = main_loss + self.center_loss_weight * center_loss
            return total_loss
        return main_loss


def calculate_class_weights(y_train):
    """计算类别权重"""
    class_counts = Counter(y_train)
    total = len(y_train)
    num_classes = len(class_counts)
    weights = torch.zeros(num_classes)
    for class_id, count in class_counts.items():
        if count > 0:
            weights[class_id] = total / (num_classes * count)
    return weights


"""第 3 部分: 融合的训练与评估逻辑"""


def run_augmentation_for_fold(spec_train, env_train, y_train, device, num_classes, gan_params, gan_epochs):
    """在单折的训练集上执行数据增强"""
    # print(f"  [Augmentation] 启动折内数据增强...")

    # 1. 找到目标样本量 (该折训练集中的最大类)
    class_counts = np.bincount(y_train)
    max_samples = np.max(class_counts)
    # print(f"  [Augmentation] 平衡所有类别至 {max_samples} 个样本。")

    # 2. 准备存储列表 (首先包含所有原始数据)
    final_spec_list = [spec_train]
    final_env_list = [env_train]
    final_y_list = [y_train]

    # 3. 遍历每个类别进行增强 (占位符，实际逻辑在下面)
    for class_label in np.unique(y_train):
        class_mask = (y_train == class_label)
        num_samples = np.sum(class_mask)
        num_to_generate = max_samples - num_samples

        # if num_to_generate > 0:
        #     print(f"    - 类别 {class_label}: 样本数 {num_samples}, 需生成 {num_to_generate} ...")
        # else:
        #     print(f"    - 类别 {class_label}: 样本数 {num_samples} (多数类)，无需生成。")

    # 5. 在整个折训练集上训练一次 VACWGAN
    print(f"  [Augmentation] 正在训练 VACWGAN (Epochs={gan_epochs})...")
    # print(f"  [Augmentation] 使用参数: {gan_params}")
    fold_dataset = MultiModalDataset(spec_train, env_train, y_train)
    # 确保 batch_size 不大于总样本数
    fold_batch_size = min(len(y_train), 64)
    fold_loader = DataLoader(fold_dataset, batch_size=fold_batch_size, shuffle=True)

    # 传入专属的 gan_params 和 gan_epochs
    E, G, D, C = train_vacwgan(fold_loader, num_classes, device, gan_params, epochs=gan_epochs)
    # print(f"  [Augmentation] GAN 训练完毕。")

    # 6. 再次循环，这次仅用于生成
    # print(f"  [Augmentation] 正在生成样本...")
    for class_label in np.unique(y_train):
        class_mask = (y_train == class_label)
        x_spec_class = spec_train[class_mask]
        x_env_class = env_train[class_mask]
        y_class = y_train[class_mask]

        num_samples = len(y_class)
        num_to_generate = max_samples - num_samples

        if num_to_generate > 0:
            # 7. 生成 (spec, env)
            syn_spec, syn_env, syn_y = generate_synthetic_samples(
                E, G, x_spec_class, x_env_class, y_class,
                num_to_generate, device
            )

            if len(syn_spec) > 0:
                # 8. (已移除 time_stats 插补)
                # 9. 添加到最终列表
                final_spec_list.append(syn_spec)
                final_env_list.append(syn_env)
                final_y_list.append(syn_y)

    # 10. 拼接所有数据
    final_spec_train = np.concatenate(final_spec_list, axis=0)
    final_env_train = np.concatenate(final_env_list, axis=0)
    final_y_train = np.concatenate(final_y_list, axis=0)

    # print(f"  [Augmentation] 折内增强完成。训练集规模: {len(y_train)} -> {len(final_y_train)}")

    return final_spec_train, final_env_train, final_y_train


def train_single_fold(spec_train, env_train, y_train,
                      spec_val, env_val, y_val,
                      config, device, num_classes):
    """训练单个折的 MultiFeatureFusionModel"""

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(spec_train),
        torch.FloatTensor(env_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(spec_val),
        torch.FloatTensor(env_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 创建模型
    model = MultiFeatureFusionModel(
        n_spec_channels=spec_train.shape[1],
        n_env_channels=env_train.shape[1],
        spec_hidden=config['spec_hidden'],
        env_hidden=config['env_hidden'],
        num_classes=num_classes,
        spec_attention=config['spec_attention'],
        env_attention=config['env_attention'],
        fusion_method=config['fusion_method'],
        dropout=config['dropout']
    ).to(device)

    # 损失函数
    class_weights = calculate_class_weights(y_train)
    criterion = CombinedLoss(
        num_classes=num_classes,
        feature_dim=64,  # 融合后的特征维度
        device=device,
        loss_type=config['loss_type'],
        use_label_smoothing=config['use_label_smoothing'],
        use_center_loss=config['use_center_loss'],
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        label_smoothing=config.get('label_smoothing', 0.1),
        center_loss_weight=config.get('center_loss_weight', 0.003),
        class_weights=class_weights.to(device)
    )

    # 优化器
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler_factor'], patience=config['scheduler_patience']
    )

    # 训练
    best_val_acc = 0
    best_model_state = None
    patience = 0

    for epoch in range(TRAIN_EPOCHS):
        # 训练阶段
        model.train()
        for spec, env, labels in train_loader:
            spec = spec.to(device)
            env = env.to(device)
            labels = labels.to(device)

            if np.random.random() < 0.5:
                noise_level = config.get('noise_level', 0.01)
                spec = spec + torch.randn_like(spec) * noise_level
                env = env + torch.randn_like(env) * noise_level

            optimizer.zero_grad()
            outputs, features = model(spec, env, return_features=True)
            loss = criterion(outputs, features, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            optimizer.step()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for spec, env, labels in val_loader:
                spec = spec.to(device)
                env = env.to(device)
                labels = labels.to(device)
                outputs, features = model(spec, env, return_features=True)
                loss = criterion(outputs, features, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOPPING_PATIENCE:
            break

    return best_val_acc, best_model_state


def cross_validate(spec_all, env_all, y_all, config, device, num_classes):
    """5 折交叉验证, 内部包含数据增强"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=1024) # 一定要是1024
    fold_accs = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(spec_all, y_all)):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        # 1. 划分数据
        spec_train = spec_all[train_idx]
        env_train = env_all[train_idx]
        y_train = y_all[train_idx]

        spec_val = spec_all[val_idx]
        env_val = env_all[val_idx]
        y_val = y_all[val_idx]

        # 2. 归一化
        # print("  [Normalization] 正在归一化...")
        spec_mean = spec_train.mean(axis=(0, 2), keepdims=True)
        spec_std = spec_train.std(axis=(0, 2), keepdims=True) + 1e-8
        spec_train_norm = (spec_train - spec_mean) / spec_std
        spec_val_norm = (spec_val - spec_mean) / spec_std

        env_mean = env_train.mean(axis=(0, 2), keepdims=True)
        env_std = env_train.std(axis=(0, 2), keepdims=True) + 1e-8
        env_train_norm = (env_train - env_mean) / env_std
        env_val_norm = (env_val - env_mean) / env_std

        # 获取当前折的专属GAN参数
        gan_params_for_fold = FOLD_SPECIFIC_GAN_PARAMS[fold_idx]

        # 3. 数据增强
        spec_train_aug, env_train_aug, y_train_aug = run_augmentation_for_fold(
            spec_train_norm, env_train_norm, y_train, device, num_classes,
            gan_params_for_fold, GAN_EPOCHS
        )

        # 4. 使用增强后的训练集和原始验证集训练
        print(f"  [Classifier Training] 正在训练分类器 (Epochs={TRAIN_EPOCHS})...")
        fold_acc, fold_model_state = train_single_fold(
            spec_train_aug, env_train_aug, y_train_aug,
            spec_val_norm, env_val_norm, y_val,
            config, device, num_classes
        )
        print(f"  [Result] Fold {fold_idx + 1} 验证集准确率: {fold_acc:.2f}%")

        fold_accs.append(fold_acc)
        fold_models.append({
            'fold': fold_idx + 1,
            'accuracy': fold_acc,
            'model_state': fold_model_state,
            'n_train': len(y_train_aug),  # 增强后的训练集大小
            'n_val': len(y_val)
        })

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    best_acc = np.max(fold_accs)

    return mean_acc, std_acc, best_acc, fold_accs, fold_models


class TopKTracker:
    """追踪并保存 Top-K 模型"""

    def __init__(self, k, save_dir):
        self.k = k
        self.save_dir = save_dir
        self.top_trials = []
        self.top_folds = []

    def update(self, mean_acc, trial_number, trial_info, fold_models):
        self.top_trials.append((mean_acc, trial_number, trial_info, fold_models))
        self.top_trials.sort(reverse=True, key=lambda x: x[0])
        self.top_trials = self.top_trials[:self.k]

        for fold_info in fold_models:
            self.top_folds.append((
                fold_info['accuracy'],
                trial_number,
                fold_info['fold'],
                fold_info
            ))
        self.top_folds.sort(reverse=True, key=lambda x: x[0])
        self.top_folds = self.top_folds[:self.k]
        self.save_top_k_info()

    def save_top_k_info(self):
        top_k_trials_info = []
        for rank, (mean_acc, trial_num, trial_info, fold_models) in enumerate(self.top_trials, 1):
            top_k_trials_info.append({
                'rank': rank,
                'trial_number': trial_num,
                'mean_accuracy': mean_acc,
                'std_accuracy': trial_info.get('std_acc', 0),
                'best_fold_accuracy': trial_info.get('best_acc', 0),
                'config': trial_info['config']
            })
        with open(f"{self.save_dir}/top_k_trials_info.json", 'w') as f:
            json.dump(top_k_trials_info, f, indent=2)

        top_k_folds_info = []
        for rank, (fold_acc, trial_num, fold_num, fold_info) in enumerate(self.top_folds, 1):
            top_k_folds_info.append({
                'rank': rank,
                'accuracy': fold_acc,
                'trial_number': trial_num,
                'fold_number': fold_num,
                'n_train': fold_info.get('n_train', 0),
                'n_val': fold_info.get('n_val', 0)
            })
        with open(f"{self.save_dir}/top_k_best_folds_info.json", 'w') as f:
            json.dump(top_k_folds_info, f, indent=2)

    def save_models(self):
        print(f"\n{'=' * 60}")
        print(f"保存Top-{self.k}模型")
        print(f"{'=' * 60}")

        print(f"\n1️⃣ Top-{self.k}试验的最佳折模型:")
        for rank, (mean_acc, trial_num, trial_info, fold_models) in enumerate(self.top_trials, 1):
            best_fold = max(fold_models, key=lambda x: x['accuracy'])
            model_path = f"{self.save_dir}/top_trial{rank}_trial{trial_num}_mean{mean_acc:.2f}_best{best_fold['accuracy']:.2f}.pth"
            torch.save(best_fold['model_state'], model_path)
            print(f"  {rank}. Trial #{trial_num}: 平均={mean_acc:.2f}%, 最佳折={best_fold['accuracy']:.2f}%")
            print(f"     文件: {os.path.basename(model_path)}")

        if len(self.top_trials) > 0:
            mean_acc, trial_num, trial_info, fold_models = self.top_trials[0]
            print(f"\n2️⃣ 最佳试验 #{trial_num} 的5折模型:")
            print(f"  {'折数':<8} {'准确率':<12} {'训练集':<10} {'验证集':<10} {'模型路径':<60}")
            print(f"  {'-' * 100}")
            fold_details = []
            for fold_info in fold_models:
                fold_model_path = f"{self.save_dir}/best_trial_{trial_num}_fold{fold_info['fold']}_acc{fold_info['accuracy']:.2f}.pth"
                torch.save(fold_info['model_state'], fold_model_path)
                print(f"  Fold {fold_info['fold']:<3} {fold_info['accuracy']:>6.2f}%     "
                      f"{fold_info['n_train']:>6}样本   {fold_info['n_val']:>6}样本   "
                      f"{os.path.basename(fold_model_path)}")
                fold_details.append({
                    'fold': fold_info['fold'],
                    'accuracy': fold_info['accuracy'],
                    'n_train': fold_info['n_train'],
                    'n_val': fold_info['n_val'],
                    'model_path': fold_model_path
                })
            print(f"  {'-' * 100}")
            print(f"  平均准确率: {mean_acc:.2f}%")
            print(f"  标准差: {trial_info.get('std_acc', 0):.2f}%")
            best_trial_folds_info = {
                'trial_number': trial_num,
                'mean_accuracy': mean_acc,
                'std_accuracy': trial_info.get('std_acc', 0),
                'best_fold_accuracy': max(f['accuracy'] for f in fold_models),
                'config': trial_info['config'],
                'fold_details': fold_details
            }
            with open(f"{self.save_dir}/best_trial_all_folds_info.json", 'w') as f:
                json.dump(best_trial_folds_info, f, indent=2)

        print(f"\n3️⃣ 全局Top-{self.k}最佳验证集准确率的fold模型:")
        print(f"  {'排名':<6} {'准确率':<12} {'来源':<25} {'模型路径':<60}")
        print(f"  {'-' * 110}")
        for rank, (fold_acc, trial_num, fold_num, fold_info) in enumerate(self.top_folds, 1):
            model_path = f"{self.save_dir}/top_fold{rank}_trial{trial_num}_fold{fold_num}_acc{fold_acc:.2f}.pth"
            torch.save(fold_info['model_state'], model_path)
            source_info = f"Trial #{trial_num}, Fold {fold_num}"
            print(f"  {rank:<6} {fold_acc:>6.2f}%     {source_info:<25} {os.path.basename(model_path)}")
        print(f"  {'-' * 110}")
        print(f"  说明: 这些是所有试验所有fold中验证准确率最高的{self.k}个模型")


def objective(trial, spec_all, env_all, y_all, device, save_dir, top_k_tracker, num_classes):
    """Optuna目标函数"""
    config = {
        'spec_hidden': trial.suggest_categorical('spec_hidden', [16, 32, 64]),
        'env_hidden': trial.suggest_categorical('env_hidden', [16, 32, 64]),
        'spec_attention': trial.suggest_categorical('spec_attention', ['none', 'se', 'cbam', 'self']),
        'env_attention': trial.suggest_categorical('env_attention', ['none', 'se', 'cbam', 'self']),
        'fusion_method': trial.suggest_categorical('fusion_method', ['concat']),  # 'weighted' 和 'add' 未完全实现
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD']),
        'scheduler_factor': trial.suggest_float('scheduler_factor', 0.3, 0.8),
        'scheduler_patience': trial.suggest_int('scheduler_patience', 5, 15),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 2.0),
        'noise_level': trial.suggest_float('noise_level', 0.005, 0.05),
        'loss_type': trial.suggest_categorical('loss_type', ['ce', 'focal']),
        'use_label_smoothing': trial.suggest_categorical('use_label_smoothing', [True, False]),
        'use_center_loss': trial.suggest_categorical('use_center_loss', [True, False]),
        'focal_alpha': trial.suggest_float('focal_alpha', 0.1, 0.5),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.05, 0.2),
        'center_loss_weight': trial.suggest_float('center_loss_weight', 0.001, 0.01, log=True),
    }

    try:
        print(f"\n{'=' * 60}")
        print(f"Trial {trial.number}")
        # print(f"  注意力: spec={config['spec_attention']}, env={config['env_attention']}")
        # print(f"  损失函数: {config['loss_type']}, 标签平滑={config['use_label_smoothing']}, Center Loss={config['use_center_loss']}")

        mean_acc, std_acc, best_acc, fold_accs, fold_models = cross_validate(
            spec_all, env_all, y_all, config, device, num_classes
        )

        print(f"  结果: 平均={mean_acc:.2f}% (±{std_acc:.2f}%), 最佳={best_acc:.2f}%")
        print(f"  各折: {[f'{acc:.2f}%' for acc in fold_accs]}")
        print(f"{'=' * 60}")

        trial_info = {
            'trial_number': trial.number,
            'config': config,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'best_acc': best_acc,
            'fold_accs': fold_accs
        }
        with open(f"{save_dir}/trial_{trial.number}_info.json", 'w') as f:
            json.dump(trial_info, f, indent=2)

        top_k_tracker.update(mean_acc, trial.number, trial_info, fold_models)

        return mean_acc

    except Exception as e:
        print(f"Trial {trial.number} 失败: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def main():
    try:
        with open(DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到数据集文件! 路径: {DATASET_PATH}")
        return

    spec_all = dataset['x_train']['spec']
    env_all = dataset['x_train']['env']
    y_all = dataset['y_train']
    num_classes = len(np.unique(y_all))

    print(f"\n数据集信息:")
    print(f"  频谱特征: {spec_all.shape}")
    print(f"  包络特征: {env_all.shape}")
    print(f"  标签: {y_all.shape} (共 {num_classes} 个类别)")
    print(f"  总样本数: {len(y_all)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/AUGMENTED_SpecEnv_FoldGAN_opt_{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"结果保存目录: {save_dir}")

    top_k_tracker = TopKTracker(TOP_K_MODELS, save_dir)

    print(f"\n开始优化 (共{N_TRIALS}次试验, {N_FOLDS}折交叉验证)...")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study.optimize(
        lambda trial: objective(trial, spec_all, env_all, y_all, device, save_dir, top_k_tracker, num_classes),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    print(f"\n✅ 优化完成!")
    print(f"最佳平均准确率: {study.best_value:.2f}%")
    print(f"最佳试验: #{study.best_trial.number}")

    with open(f"{save_dir}/best_params.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)
    with open(f"{save_dir}/best_trial_info.json", 'w') as f:
        json.dump({
            'trial_number': study.best_trial.number,
            'mean_acc': study.best_value,
            'params': study.best_params
        }, f, indent=2)

    print(f"\n准备保存模型...")
    top_k_tracker.save_models()

    print(f"\n{'=' * 60}")
    print("✅ 完成!")
    print(f"{'=' * 60}")
    print(f"\n结果保存在: {save_dir}")

    print(f"\n📊 Top-{TOP_K_MODELS}模型总览:")
    print(f"{'排名':<6} {'试验号':<10} {'平均准确率':<15} {'标准差':<12} {'最佳折':<12}")
    print(f"{'-' * 60}")
    if len(top_k_tracker.top_trials) > 0:
        for rank, (mean_acc, trial_num, trial_info, fold_models) in enumerate(top_k_tracker.top_trials, 1):
            print(f"{rank:<6} #{trial_num:<9} {mean_acc:>6.2f}%         "
                  f"±{trial_info.get('std_acc', 0):>4.2f}%      {trial_info.get('best_acc', 0):>5.2f}%")

        best_mean_acc, best_trial_num, best_trial_info, best_fold_models = top_k_tracker.top_trials[0]
        print(f"\n⭐ 最佳试验 #{best_trial_num} 的5折详细准确率:")
        for fold_info in best_fold_models:
            print(f"  Fold {fold_info['fold']}: {fold_info['accuracy']:.2f}% (增强后训练集大小: {fold_info['n_train']})")
    else:
        print("未找到有效的试验结果。")

    if len(top_k_tracker.top_folds) > 0:
        print(f"\n📈 全局Top-{TOP_K_MODELS}最佳fold来源:")
        for rank, (fold_acc, trial_num, fold_num, _) in enumerate(top_k_tracker.top_folds, 1):
            print(f"  {rank}. {fold_acc:.2f}% ← Trial #{trial_num}, Fold {fold_num}")
    else:
        print("未找到最佳折模型。")


if __name__ == "__main__":
    main()
