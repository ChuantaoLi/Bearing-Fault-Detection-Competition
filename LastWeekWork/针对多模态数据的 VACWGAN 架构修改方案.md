# 针对多模态数据的 VACWGAN 架构修改方案

​		现在有一个用于数据增强的 VACWGAN，其实现框架如下面的代码所示：

```python
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

```

​		其调用的训练代码框架如下面所示：

```python
def train_vacwgan_for_augmentation(data_loader, num_classes, device, epochs):
    """VACWGAN 训练函数"""
    E = Encoder(latent_dim=LATENT_DIM).to(device)
    G = Generator(latent_dim=LATENT_DIM, num_classes=num_classes).to(device)
    D = Discriminator().to(device)
    C = Classifier(num_classes=num_classes).to(device)
    opt_E = optim.Adam(E.parameters(), lr=VACWGAN_LR, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=VACWGAN_LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=VACWGAN_LR, betas=(0.5, 0.999))
    opt_C = optim.Adam(C.parameters(), lr=VACWGAN_LR, betas=(0.5, 0.999))
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        E.train()
        G.train()
        D.train()
        C.train()
        for i, (real_imgs, labels) in enumerate(data_loader):
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            # --- 训练 D, C ---
            opt_D.zero_grad()
            opt_C.zero_grad()
            d_real = D(real_imgs)
            c_real = C(real_imgs)
            loss_D_real = -torch.mean(d_real)
            loss_C_real = ce_loss(c_real, labels)
            mu, logvar = E(real_imgs)
            z = reparameterization(mu, logvar)
            fake_imgs = G(z, labels).detach()
            d_fake = D(fake_imgs)
            c_fake = C(fake_imgs)
            loss_D_fake = torch.mean(d_fake)
            loss_C_fake = ce_loss(c_fake, labels)
            gp = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data, device)
            loss_D = loss_D_fake + loss_D_real + gp
            loss_C = loss_C_real + loss_C_fake
            loss_D.backward(retain_graph=True)
            loss_C.backward()
            opt_D.step()
            opt_C.step()
            # --- 训练 G, E ---
            if i % D_ITERS == 0:
                opt_G.zero_grad()
                opt_E.zero_grad()
                mu, logvar = E(real_imgs)
                z = reparameterization(mu, logvar)
                fake_imgs = G(z, labels)
                d_fake_g = D(fake_imgs)
                c_fake_g = C(fake_imgs)
                loss_G_adv = -torch.mean(d_fake_g)
                loss_G_cls = ce_loss(c_fake_g, labels)
                loss_E_kl = compute_kl_loss(mu, logvar)
                loss_G_total = loss_G_adv + 10 * loss_G_cls + 1 * loss_E_kl
                loss_G_total.backward()
                opt_G.step()
                opt_E.step()
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}/{epochs}, D_Loss: {loss_D.item():.4f}, C_Loss: {loss_C.item():.4f}, G_Loss: {loss_G_total.item():.4f}")
    return E, G


def generate_synthetic_samples_from_trained(E_trained, G_trained, X_minority, y_minority, num_to_generate, device):
    """使用训练好的 E 和 G 进行样本生成"""
    if X_minority.size == 0 or num_to_generate == 0: return np.array([]), np.array([])
    minority_data = PreprocessedDataset(X_minority, y_minority)

    batch_size = min(64, len(minority_data))
    if batch_size == 0: return np.array([]), np.array([])
    minority_loader = DataLoader(minority_data, batch_size=batch_size, shuffle=True)

    synthetic_X = []
    current_generated = 0
    E_trained.eval()
    G_trained.eval()
    with torch.no_grad():
        while current_generated < num_to_generate:
            for inputs, labels in minority_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                mu, logvar = E_trained(inputs)
                z = reparameterization(mu, logvar)
                fake_samples = G_trained(z, labels).cpu().numpy()
                num_batch = fake_samples.shape[0]
                if current_generated + num_batch > num_to_generate:
                    needed = num_to_generate - current_generated
                    fake_samples = fake_samples[:needed]
                    num_batch = needed
                synthetic_X.append(fake_samples.squeeze(1))
                current_generated += num_batch
                if current_generated >= num_to_generate: break
            if current_generated >= num_to_generate: break
            if batch_size == 0: break

    if not synthetic_X: return np.array([]), np.array([])
    synthetic_X = np.concatenate(synthetic_X, axis=0)
    synthetic_y = np.full(synthetic_X.shape[0], y_minority[0])
    return synthetic_X, synthetic_y


def load_pure_data_and_train_gan():
    """加载、预处理数据，并训练 VACWGAN"""
    f_paths, raw_labels = [], []
    raw_class_dirs = sorted([d for d in os.listdir(TRAIN_ROOT_DIR) if os.path.isdir(os.path.join(TRAIN_ROOT_DIR, d))])
    num_classes = len(CLEAN_CLASS_NAMES)

    for raw_name in raw_class_dirs:
        class_dir = os.path.join(TRAIN_ROOT_DIR, raw_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(".xlsx"):
                f_paths.append(os.path.join(class_dir, fname))
                raw_labels.append(raw_name)

    print(f"使用 {cpu_count()} 个CPU核心")
    input_tuples = list(zip(f_paths, raw_labels))
    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(process_single_signal_worker, input_tuples), total=len(input_tuples), desc="并行处理信号"))

    X_processed = np.array([res[0] for res in results])
    y_labels = np.array([res[1] for res in results])
    valid_mask = y_labels != -1
    X_processed = X_processed[valid_mask]
    y_labels = y_labels[valid_mask]

    full_dataset = PreprocessedDataset(X_processed, y_labels)
    full_data_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    E_trained, G_trained = train_vacwgan_for_augmentation(full_data_loader, num_classes, DEVICE, VACWGAN_EPOCHS)

    return X_processed, y_labels, E_trained, G_trained, CLEAN_CLASS_NAMES
```

​		现在，由于我的需求变化，对于上面的代码你需要给我改进：

1. 具体的网络结构请你给我设计得更加复杂一些，使得能够学习到更加复杂的信息，当然，需要合理。

2. 以上的代码只是实现的参考，你并不需要强制按照上面的细节进行改进，但是我需要你做一个重大的修改，那就是我的输入不再是时频图了，而是多模态的输入。具体而言，我的训练集和测试集的数据格式是这样的：

   ```python
   --- 训练数据 ---
   x_train 'spec' Shape: (860, 7, 48), Dtype: float32
   x_train 'env'  Shape: (860, 5, 32), Dtype: float32
   y_train        Shape: (860,), Dtype: int64
   
   --- 测试数据 ---
   x_test 'spec' Shape: (1140, 7, 48), Dtype: float32
   x_test 'env'  Shape: (1140, 5, 32), Dtype: float32
   ```

   由于测试集是不带标签的，在训练的过程中，我首先划分了5折交叉验证所用的各折的训练集和验证集，如下面的代码所示：

   ```python
   import pickle
   from sklearn.model_selection import StratifiedKFold
   
   pkl_path = r'dataset_best_cv.pkl'
   
   """对于pkl文件来说，读取后就是一个字典"""
   with open(pkl_path, 'rb') as f:
       data = pickle.load(f)
   
   """取出训练集"""
   x_train_spec = data['x_train']['spec']  # 频谱特征
   x_train_env = data['x_train']['env']  # 包络特征
   y_train = data['y_train']  # 训练集标签
   
   """取出测试集"""
   x_test_spec = data['x_test']['spec']  # 频谱特征
   x_test_env = data['x_test']['env']  # 包络特征
   
   """打印形状"""
   print("\n--- 训练数据 ---")
   print(f"x_train 'spec' Shape: {x_train_spec.shape}, Dtype: {x_train_spec.dtype}")
   print(f"x_train 'env'  Shape: {x_train_env.shape}, Dtype: {x_train_env.dtype}")
   print(f"y_train        Shape: {y_train.shape}, Dtype: {y_train.dtype}")
   
   print("\n--- 测试数据 ---")
   print(f"x_test 'spec' Shape: {x_test_spec.shape}, Dtype: {x_test_spec.dtype}")
   print(f"x_test 'env'  Shape: {x_test_env.shape}, Dtype: {x_test_env.dtype}")
   
   """准备好5折交叉验证的数据集"""
   n_splits = 5
   skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1024)
   cv_splits = []
   
   for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train_spec, y_train), 1):
       fold_data = {
           'train_indices': train_idx,
           'val_indices': val_idx,
           'fold': fold_idx
       }
       cv_splits.append(fold_data)
   
   save_data = {
       'cv_splits': cv_splits,
       'n_splits': n_splits,
       'x_train': {
           'spec': x_train_spec,
           'env': x_train_env
       },
       'y_train': y_train,
       'x_test': {
           'spec': x_test_spec,
           'env': x_test_env
       },
       'random_state': 1024,
       'stratified': True
   }
   
   ```

   ​		可见，我的输入不再是一种时频图模态，而是频谱特征通道和包络特征通道，它们的形状分别为(860, 7, 48)和(860, 5, 32)，我现在给你一种可以参考的改进思路，你可以基于这种思路，实现我的需求：将两种模态的数据融合起来输入给 VACWGAN 进行数据合成。

   ​		我给你的一种可以参考的思路是：采用特征层融合策略，即修改网络的四个核心组件，使其能够并行处理两个输入流，并在网络内部进行信息融合。

---

## 1. 编码器 (Encoder, E) 的修改

**目标：** 将两个模态 `(x_spec, x_env)` 压缩为一个共享的潜变量 $z$ （通过其均值 $\mu$ 和方差 $\sigma$）。

* **输入：**
    * `x_spec`: (BatchSize, 7, 48) 或 (BatchSize, 7, 48, 1)
    * `x_env`: (BatchSize, 5, 32) 或 (BatchSize, 5, 32, 1)
* **结构：**
    1.  **分支1 (Spec Branch):** 使用一系列2D卷积层（`Conv2D`）处理 `x_spec`，提取特征。
    2.  **分支2 (Env Branch):** 使用另一系列 `Conv2D` 层处理 `x_env`，提取特征。
    3.  **融合 (Fusion):** 将两个分支的输出展平（`Flatten`），然后拼接（`Concatenate`）成一个单一的特征向量。
    4.  **输出 (Output Head):** 将这个融合向量通过全连接层（`Dense`），最终分别输出 $\mu$ 和 $\sigma$ 向量。

---

## 2. 生成器 (Generator, G) 的修改

**目标：** 从潜变量 $z$ 和类别标签 $c$ 出发，同时生成两个模态的虚假样本$(\hat{x}_{spec}, \hat{x}_{env})$。

* **输入：**
    * `z`: (BatchSize, LatentVariableDimensions)
    * `c`: (BatchSize, NumClasses) (通常是 one-hot 编码)
* **结构：**
    1.  **输入融合：** 将 $z$ 和 $c$ 拼接在一起。
    2.  **共享层 (Shared Trunk):** 将融合后的输入通过一个共享的“主干”网络（例如，`Dense` 层，然后 `Reshape` 为一个小的特征图）。
    3.  **分裂 (Split):** 在某个中间层后，网络分成两个分支。
    4.  **分支1 (Spec Branch):** 使用一系列反卷积层（`Conv2DTranspose`）将共享特征上采样，最终重塑为 `(7, 48, 1)`，生成 $\hat{x}_{spec}$。
    5.  **分支2 (Env Branch):** 使用另一系列 `Conv2DTranspose` 层将共享特征上采样，最终重塑为 `(5, 32, 1)`，生成 $\hat{x}_{env}$。

---

## 3. 判别器 (D) 和 分类器 (C) 的修改

**目标：** 这两个网络都需要判断一对样本 `(x_spec, x_env)` 是真是假（D）或属于哪一类（C）。它们的结构几乎相同，只是输出头不同。

* **输入：**
    * `x_spec`: (BatchSize, 7, 48, 1)
    * `x_env`: (BatchSize, 5, 32, 1)
* **结构（以D为例）：**
    1.  **分支1 (Spec Branch):** `Conv2D` 层处理 `x_spec`。
    2.  **分支2 (Env Branch):** `Conv2D` 层处理 `x_env`。
    3.  **融合 (Fusion):** 展平（`Flatten`）并拼接（`Concatenate`）两个分支的特征。
    4.  **输出 (Output Head):**
        * **判别器 (D):** 将融合向量通过 `Dense` 层，最后输出一个标量值（代表“真/假”）。
        * **分类器 (C):** 将融合向量通过 `Dense` 层，最后通过 `Softmax` 输出类别概率。

---

## 4. 训练流程的变化

​		训练循环大体保持不变，但输入和输出会相应调整：

1.  `E` 接收 `(x_spec, x_env)`，输出 $\mu, \sigma$。
2.  `G` 接收 `(z, c)`，输出$ (\hat{x}_{spec}, \hat{x}_{env})$。
3.  **** `D` 接收 `(x_spec, x_env)`（真实）和 $(\hat{x}_{spec}, \hat{x}_{env})$（虚假），计算 $L_D$。
4.  `C` 接收 `(x_spec, x_env)`（真实）和 $(\hat{x}_{spec}, \hat{x}_{env})$（虚假），计算 $L_C$ 。
5.  **损失函数 $L_E$, $L_G$:**
    * $L_E$ 的计算不变（仅依赖 $\mu, \sigma$）。
    * $L_G$ 的计算也不变，因为它依赖于 `D` 和 `C` 对虚假样本 $(\hat{x}_{spec}, \hat{x}_{env})$ 的输出 。

---

​		要注意的是，以上是一种可以参考的思路，你不一定需要全部按照这些思路来进行代码的编写，你需要给我的改进结果应该是合理的，并且是可行的。此外，要把合成好的5折数据导出成pkl文件哦，在进行改进 VANWGAN 网络的训练中，要使用Optuna进行网络参数的调优，并且保存每一折的最优参数和模型pth。请给我所有的实现的代码。