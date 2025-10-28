import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import optuna
import os
from sklearn.model_selection import StratifiedKFold


# ==================== 数据集类 ====================
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


# ==================== 编码器 ====================
class MultiModalEncoder(nn.Module):
    """多模态编码器，处理spec和env两个输入"""

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

        # 计算展平后的特征维度
        self.spec_flat_dim = 256 * 1 * 6
        self.env_flat_dim = 256 * 1 * 4
        self.combined_dim = self.spec_flat_dim + self.env_flat_dim

        # 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        # 输出层
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x_spec, x_env):
        # 添加通道维度
        x_spec = x_spec.unsqueeze(1)  # (B, 1, 7, 48)
        x_env = x_env.unsqueeze(1)  # (B, 1, 5, 32)

        # 分支处理
        spec_feat = self.spec_branch(x_spec)
        env_feat = self.env_branch(x_env)

        # 展平
        spec_feat = spec_feat.view(spec_feat.size(0), -1)
        env_feat = env_feat.view(env_feat.size(0), -1)

        # 融合
        combined = torch.cat([spec_feat, env_feat], dim=1)
        fused = self.fusion_fc(combined)

        # 输出mu和logvar
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)

        return mu, logvar


# ==================== 生成器 ====================
class MultiModalGenerator(nn.Module):
    """多模态生成器，生成spec和env两个输出"""

    def __init__(self, latent_dim=128, num_classes=10):
        super(MultiModalGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # 共享主干
        self.shared_trunk = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )

        # Spec生成分支
        self.spec_fc = nn.Sequential(
            nn.Linear(1024, 256 * 2 * 6),
            nn.BatchNorm1d(256 * 2 * 6),
            nn.ReLU(True),
        )
        self.spec_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 4, 12)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Env生成分支
        self.env_fc = nn.Sequential(
            nn.Linear(1024, 256 * 2 * 4),
            nn.BatchNorm1d(256 * 2 * 4),
            nn.ReLU(True),
        )
        self.env_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 4, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 4), stride=1, padding=1),  # (64, 4, 9)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 标签嵌入
        c_emb = self.label_embedding(labels)
        z_c = torch.cat([z, c_emb], dim=1)

        # 共享特征
        shared = self.shared_trunk(z_c)

        # Spec生成
        spec_feat = self.spec_fc(shared)
        spec_feat = spec_feat.view(-1, 256, 2, 6)
        spec_out = self.spec_deconv(spec_feat)
        # 裁剪到目标尺寸 (1, 7, 48)
        spec_out = F.interpolate(spec_out, size=(7, 48), mode='bilinear', align_corners=False)

        # Env生成
        env_feat = self.env_fc(shared)
        env_feat = env_feat.view(-1, 256, 2, 4)
        env_out = self.env_deconv(env_feat)
        # 裁剪到目标尺寸 (1, 5, 32)
        env_out = F.interpolate(env_out, size=(5, 32), mode='bilinear', align_corners=False)

        return spec_out.squeeze(1), env_out.squeeze(1)


# ==================== 判别器 ====================
class MultiModalDiscriminator(nn.Module):
    """多模态判别器"""

    def __init__(self):
        super(MultiModalDiscriminator, self).__init__()

        # Spec分支
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

        # Env分支
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

        # 计算特征维度
        self.spec_feat_dim = 128 * 1 * 6
        self.env_feat_dim = 128 * 1 * 4

        # 融合和输出
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

        combined = torch.cat([spec_feat, env_feat], dim=1)
        validity = self.adv_layer(combined)

        return validity


# ==================== 分类器 ====================
class MultiModalClassifier(nn.Module):
    """多模态分类器"""

    def __init__(self, num_classes=10):
        super(MultiModalClassifier, self).__init__()

        # Spec分支
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

        # Env分支
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

        # 分类层
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

        combined = torch.cat([spec_feat, env_feat], dim=1)
        class_output = self.classification_layer(combined)

        return class_output


# ==================== 辅助函数 ====================
def reparameterization(mu, logvar):
    """重参数化技巧"""
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


# ==================== 训练函数 ====================
def train_vacwgan(data_loader, num_classes, device, params, epochs=100):
    """训练多模态VACWGAN"""
    latent_dim = params['latent_dim']
    lr = params['lr']
    d_iters = params['d_iters']
    lambda_cls = params['lambda_cls']
    lambda_kl = params['lambda_kl']

    E = MultiModalEncoder(latent_dim=latent_dim).to(device)
    G = MultiModalGenerator(latent_dim=latent_dim, num_classes=num_classes).to(device)
    D = MultiModalDiscriminator().to(device)
    C = MultiModalClassifier(num_classes=num_classes).to(device)

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

            # 训练D和C
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

            # 训练G和E
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

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, D: {loss_D.item():.4f}, "
                  f"C: {loss_C.item():.4f}, G: {loss_G_total.item():.4f}")

    return E, G, D, C


# ==================== 数据生成函数 ====================
def generate_synthetic_samples(E, G, x_spec_minority, x_env_minority,
                               y_minority, num_to_generate, device):
    """生成合成样本"""
    if len(y_minority) == 0 or num_to_generate == 0:
        return np.array([]), np.array([]), np.array([])

    minority_dataset = MultiModalDataset(x_spec_minority, x_env_minority, y_minority)
    minority_loader = DataLoader(minority_dataset, batch_size=64, shuffle=True)

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


# ==================== Optuna目标函数 ====================
def objective(trial, train_loader, val_loader, num_classes, device):
    """Optuna优化目标"""
    params = {
        'latent_dim': trial.suggest_int('latent_dim', 64, 256, step=32),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'd_iters': trial.suggest_int('d_iters', 3, 7),
        'lambda_cls': trial.suggest_float('lambda_cls', 5.0, 20.0),
        'lambda_kl': trial.suggest_float('lambda_kl', 0.1, 2.0),
    }

    E, G, D, C = train_vacwgan(train_loader, num_classes, device, params, epochs=100)

    # 在验证集上评估
    C.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for spec, env, labels in val_loader:
            spec, env, labels = spec.to(device), env.to(device), labels.to(device)
            outputs = C(spec, env)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


# ==================== 主训练流程 ====================
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.zeros(1, device=device)

    # 加载数据
    pkl_path = r'dataset_best_cv.pkl'

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {pkl_path}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pkl_path}")
        print("Please update 'pkl_path' in the main() function.")
        return  # 找不到文件则退出

    x_train_spec = data['x_train']['spec']
    x_train_env = data['x_train']['env']
    y_train = data['y_train']
    x_test_spec = data['x_test']['spec']
    x_test_env = data['x_test']['env']

    num_classes = len(np.unique(y_train))
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1024)

    # 存储结果
    fold_results = []
    augmented_data = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train_spec, y_train), 1):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold_idx}/{n_splits}")
        print(f"{'=' * 50}")

        # 划分数据
        x_spec_train = x_train_spec[train_idx]
        x_env_train = x_train_env[train_idx]
        y_train_fold = y_train[train_idx]

        x_spec_val = x_train_spec[val_idx]
        x_env_val = x_train_env[val_idx]
        y_val_fold = y_train[val_idx]

        # 创建数据加载器
        train_dataset = MultiModalDataset(x_spec_train, x_env_train, y_train_fold)
        val_dataset = MultiModalDataset(x_spec_val, x_env_val, y_val_fold)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Optuna超参数优化
        print(f"\nStarting Optuna optimization for Fold {fold_idx}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_loader, val_loader,
                                               num_classes, device),
                       n_trials=20, show_progress_bar=True)  # n_trials=20 是调参试验次数

        best_params = study.best_params
        print(f"\nBest params for Fold {fold_idx}: {best_params}")
        print(f"Best validation accuracy: {study.best_value:.4f}")

        # 使用最优参数训练最终模型
        print(f"\nTraining final model with best params...")
        E, G, D, C = train_vacwgan(train_loader, num_classes, device,
                                   best_params, epochs=100)  # epochs=100 是最终训练轮数

        # 保存模型
        os.makedirs('models', exist_ok=True)
        torch.save({
            'encoder': E.state_dict(),
            'generator': G.state_dict(),
            'discriminator': D.state_dict(),
            'classifier': C.state_dict(),
            'params': best_params,
            'fold': fold_idx
        }, f'models/fold_{fold_idx}_best_model.pth')

        # ==========================================================
        # --- 数据生成与输出逻辑修改 ---
        # ==========================================================
        print(f"\nGenerating augmented data for balancing...")
        augmented_spec_fold = []
        augmented_env_fold = []
        augmented_y_fold = []

        # 1. 找到当前折训练集中，样本量最大的类的数量
        class_counts = np.bincount(y_train_fold)
        max_samples = np.max(class_counts)
        print(f"Balancing all classes up to {max_samples} samples.")

        for class_label in np.unique(y_train_fold):
            class_mask = y_train_fold == class_label
            x_spec_class = x_spec_train[class_mask]
            x_env_class = x_env_train[class_mask]
            y_class = y_train_fold[class_mask]

            num_samples = len(y_class)

            # 2. 计算需要生成的样本数
            if num_samples < max_samples:
                num_to_generate = max_samples - num_samples
                print(f"Class {class_label}: Has {num_samples}, generating {num_to_generate} new samples...")
            else:
                num_to_generate = 0
                print(f"Class {class_label}: Is majority class with {num_samples} samples. No generation needed.")

            # 3. 调用生成函数
            syn_spec, syn_env, syn_y = generate_synthetic_samples(
                E, G, x_spec_class, x_env_class, y_class,
                num_to_generate, device
            )

            if len(syn_spec) > 0:
                augmented_spec_fold.append(syn_spec)
                augmented_env_fold.append(syn_env)
                augmented_y_fold.append(syn_y)

        # --- 逻辑修改结束 ---

        # 合并原始数据和增强数据
        if augmented_spec_fold:
            augmented_spec_fold = np.concatenate(augmented_spec_fold, axis=0)
            augmented_env_fold = np.concatenate(augmented_env_fold, axis=0)
            augmented_y_fold = np.concatenate(augmented_y_fold, axis=0)

            final_spec = np.concatenate([x_spec_train, augmented_spec_fold], axis=0)
            final_env = np.concatenate([x_env_train, augmented_env_fold], axis=0)
            final_y = np.concatenate([y_train_fold, augmented_y_fold], axis=0)
        else:
            # 如果没有生成任何数据 (例如数据集本身是平衡的)
            final_spec = x_spec_train
            final_env = x_env_train
            final_y = y_train_fold

        # 保存该折的增强数据
        fold_data = {
            'train_spec': final_spec,
            'train_env': final_env,
            'train_labels': final_y,
            'val_spec': x_spec_val,
            'val_env': x_env_val,
            'val_labels': y_val_fold,
            'best_params': best_params,
            'best_val_acc': study.best_value,
            'fold': fold_idx
        }

        augmented_data.append(fold_data)
        fold_results.append({
            'fold': fold_idx,
            'best_params': best_params,
            'best_val_acc': study.best_value
        })

        print(f"\nFold {fold_idx} completed!")
        print(f"Original training size: {len(x_spec_train)}")
        print(f"Augmented (Balanced) training size: {len(final_spec)}")

    # 保存所有折的增强数据
    print(f"\n{'=' * 50}")
    print("Saving augmented data...")
    print(f"{'=' * 50}")

    output_data = {
        'augmented_folds': augmented_data,
        'fold_results': fold_results,
        'x_test': {
            'spec': x_test_spec,
            'env': x_test_env
        },
        'num_classes': num_classes,
        'n_splits': n_splits
    }

    with open('augmented_dataset_5fold.pkl', 'wb') as f:
        pickle.dump(output_data, f)

    print("\nAll folds completed and saved!")
    print(f"Augmented data saved to: augmented_dataset_5fold.pkl")

    # 打印每折的结果摘要
    print(f"\n{'=' * 50}")
    print("Results Summary")
    print(f"{'=' * 50}")
    for result in fold_results:
        print(f"\nFold {result['fold']}:")
        print(f"  Best Validation Accuracy: {result['best_val_acc']:.4f}")
        print(f"  Best Parameters: {result['best_params']}")


if __name__ == '__main__':
    main()
