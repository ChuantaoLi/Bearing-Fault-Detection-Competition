# ===================================================================================
# 文件名: vacwgan.py (新增损失函数和优化器定义)
# ===================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. 编码器 (Encoder) ---
class Encoder(nn.Module):
    """VACWGAN的编码器E，用于提取潜在空间参数 (μ, log(σ^2))"""

    def __init__(self, latent_dim=100):
        super(Encoder, self).__init__()
        # Input: 1 x 64 x 32
        self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # 16 x 32 x 16
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)  # 32 x 16 x 8
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 64 x 8 x 4
        self.bn3 = nn.BatchNorm2d(64)
        self.fc_size = 64 * 8 * 4  # 2048

        self.fc_mu = nn.Linear(self.fc_size, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_size, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = x.view(-1, self.fc_size)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# --- 2. 生成器 (Generator) ---
class Generator(nn.Module):
    """VACWGAN的生成器G，接收潜在变量z和类别c生成样本"""

    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.init_H = 8
        self.init_W = 4
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim * 2, 128 * self.init_H * self.init_W),
            nn.BatchNorm1d(128 * self.init_H * self.init_W),
            nn.ReLU(True)
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c_emb = self.label_embedding(labels)
        z_c = torch.cat((z, c_emb), dim=1)

        out = self.l1(z_c)
        out = out.view(out.size(0), 128, self.init_H, self.init_W)
        img = self.conv_blocks(out)
        return img

    # --- 3. 判别器 (Discriminator) ---


class Discriminator(nn.Module):
    """VACWGAN的判别器D，仅负责真/假二分类判别"""

    def __init__(self):
        super(Discriminator, self).__init__()
        # 简化结构以匹配 Encoder
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc_size = 64 * 4 * 2  # 512

        self.adv_layer = nn.Sequential(nn.Linear(self.fc_size, 1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity

    # --- 4. 分类器 (Classifier) ---


class Classifier(nn.Module):
    """VACWGAN的独立分类器C，仅负责多类别分类"""

    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc_size = 64 * 4 * 2

        self.classification_layer = nn.Sequential(nn.Linear(self.fc_size, num_classes))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)
        class_output = self.classification_layer(out)
        return class_output


# --- 5. 损失函数和辅助函数 ---

def reparameterization(mu, logvar):
    """重参数化技巧"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def compute_gradient_penalty(D, real_samples, fake_samples, device, gamma=10):
    """计算 WGAN-GP 中的梯度惩罚项"""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=device).requires_grad_(False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gamma
    return gradient_penalty


def compute_kl_loss(mu, logvar):
    """计算 VAE 的 KL 散度损失"""
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss
