# ===================================================================================
# 文件名: main_train_eval.py (最终修正 - 集成学习 + 正确的CV增强)
# 描述:
# 1. [修复数据泄露] 在 K-Fold 循环内部进行数据增强，在纯净验证集上评估。
# 2. [集成学习] Optuna 仅用于寻找最佳超参数。
# 3. [集成学习] 找到最佳参数后，重新训练 5 个 K-Fold 模型用于集成。
# 4. [集成学习] 推理时加载 5 个模型，使用平均 Logits 进行预测。
# ===================================================================================
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool, cpu_count
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from preprocessing import VMD_Sorter_SSA_Processor, stft_transform
from tqdm import tqdm
import numpy as np
import pandas as pd
from model import Simple2DCNN
from use_loss import CombinedLoss, calculate_class_weights
# 导入所有 VACWGAN 组件
from vacwgan import (Encoder, Generator, Discriminator, Classifier,
                     reparameterization, compute_gradient_penalty, compute_kl_loss)
# 导入更全面的评估工具
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

import optuna
import shutil
import json
from pathlib import Path
from datetime import datetime
import random

# --- 全局常量和设置 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TRAIN_ROOT_DIR = r"D:\轴承故障检测竞赛\初赛训练集"
TEST_ROOT_DIR = r"D:\轴承故障检测竞赛\初赛测试集"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 类别映射 ---
RAW_TO_CLEAN_MAP = {
    'inner_broken_train150': 'inner_broken',
    'inner_wear_train120': 'inner_wear',
    'normal_train160': 'normal',
    'outer_missing_train180': 'outer_missing',
    'roller_broken_train150': 'roller_broken',
    'roller_wear_train100': 'roller_wear'
}
CLEAN_CLASS_NAMES = sorted(list(set(RAW_TO_CLEAN_MAP.values())))
CLEAN_LABEL_MAP = {name: i for i, name in enumerate(CLEAN_CLASS_NAMES)}

# --- 超参数设置 ---
VACWGAN_EPOCHS = 300
VACWGAN_LR = 1e-4
VACWGAN_SAMPLES_PER_CLASS = 50
LATENT_DIM = 100
N_FOLDS = 5  # 交叉验证折数

N_TRIALS = 300  # (减少试验次数以便演示，你可以调回 300)
NUM_EPOCHS = 100
OPTIMIZATION_TIMEOUT = 10800
D_ITERS = 5

# --- 全局预处理器参数 ---
GLOBAL_PREPROCESSOR_PARAMS = {
    'K': 5, 'alpha': 2000, 'window_len': 128, 'ssa_threshold': 32, 'denoise_indices': [0, 1]
}


# --- 多进程工作函数（训练集） ---
def process_single_signal_worker(path_label_tuple):
    path, raw_label_name = path_label_tuple
    clean_label_name = RAW_TO_CLEAN_MAP.get(raw_label_name, 'unknown')
    label_int = CLEAN_LABEL_MAP.get(clean_label_name, -1)
    preprocessor = VMD_Sorter_SSA_Processor(**GLOBAL_PREPROCESSOR_PARAMS)
    signal = pd.read_excel(path, header=None, usecols=[1]).squeeze("columns").to_numpy()
    denoised_signal = preprocessor.process(signal)
    stft_map = stft_transform(denoised_signal)
    return stft_map, label_int


# --- 测试集的多进程工作函数 ---
def process_single_test_signal_worker(path):
    preprocessor = VMD_Sorter_SSA_Processor(**GLOBAL_PREPROCESSOR_PARAMS)
    signal = pd.read_excel(path, header=None, usecols=[1]).squeeze("columns").to_numpy()
    denoised_signal = preprocessor.process(signal)
    stft_map = stft_transform(denoised_signal)
    return stft_map


# --- 数据集类 ---
class PreprocessedDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # 注意：这里添加了通道维度 (C=1)
        signal = torch.from_numpy(self.signals[idx]).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label


# --- VACWGAN 完整训练函数 ---
def train_vacwgan_for_augmentation(data_loader, num_classes, device, epochs):
    print("\n--- 开始 VACWGAN 对抗训练 ---")
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
        E.train();
        G.train();
        D.train();
        C.train()
        for i, (real_imgs, labels) in enumerate(data_loader):
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            # --- 训练 D, C ---
            opt_D.zero_grad();
            opt_C.zero_grad()
            d_real = D(real_imgs);
            c_real = C(real_imgs)
            loss_D_real = -torch.mean(d_real);
            loss_C_real = ce_loss(c_real, labels)
            mu, logvar = E(real_imgs);
            z = reparameterization(mu, logvar)
            fake_imgs = G(z, labels).detach()
            d_fake = D(fake_imgs);
            c_fake = C(fake_imgs)
            loss_D_fake = torch.mean(d_fake);
            loss_C_fake = ce_loss(c_fake, labels)
            gp = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data, device)
            loss_D = loss_D_fake + loss_D_real + gp
            loss_C = loss_C_real + loss_C_fake
            loss_D.backward(retain_graph=True);
            loss_C.backward()
            opt_D.step();
            opt_C.step()
            # --- 训练 G, E ---
            if i % D_ITERS == 0:
                opt_G.zero_grad();
                opt_E.zero_grad()
                mu, logvar = E(real_imgs);
                z = reparameterization(mu, logvar)
                fake_imgs = G(z, labels)
                d_fake_g = D(fake_imgs);
                c_fake_g = C(fake_imgs)
                loss_G_adv = -torch.mean(d_fake_g);
                loss_G_cls = ce_loss(c_fake_g, labels)
                loss_E_kl = compute_kl_loss(mu, logvar)
                loss_G_total = loss_G_adv + 10 * loss_G_cls + 1 * loss_E_kl
                loss_G_total.backward()
                opt_G.step();
                opt_E.step()
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}/{epochs}, D_Loss: {loss_D.item():.4f}, C_Loss: {loss_C.item():.4f}, G_Loss: {loss_G_total.item():.4f}")
    print("--- VACWGAN 对抗训练结束 ---")
    return E, G


# --- [NEW] 样本生成函数（从 load_and_augment_data 移出） ---
def generate_synthetic_samples_from_trained(E_trained, G_trained, X_minority, y_minority, num_to_generate, device):
    """使用训练好的 E 和 G 进行样本生成"""
    if X_minority.size == 0 or num_to_generate == 0: return np.array([]), np.array([])
    # 使用 PreprocessedDataset 以匹配训练时的输入 (N, C, H, W)
    minority_data = PreprocessedDataset(X_minority, y_minority)
    # 确保 batch_size 不大于样本数
    batch_size = min(64, len(minority_data))
    if batch_size == 0: return np.array([]), np.array([])
    minority_loader = DataLoader(minority_data, batch_size=batch_size, shuffle=True)

    synthetic_X = []
    current_generated = 0
    E_trained.eval();
    G_trained.eval()
    with torch.no_grad():
        while current_generated < num_to_generate:
            for inputs, labels in minority_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                mu, logvar = E_trained(inputs);
                z = reparameterization(mu, logvar)
                fake_samples = G_trained(z, labels).cpu().numpy()
                num_batch = fake_samples.shape[0]
                if current_generated + num_batch > num_to_generate:
                    needed = num_to_generate - current_generated
                    fake_samples = fake_samples[:needed]
                    num_batch = needed
                synthetic_X.append(fake_samples.squeeze(1));  # (N, 1, H, W) -> (N, H, W)
                current_generated += num_batch
                if current_generated >= num_to_generate: break
            # 如果样本量太少，loader 循环一次就结束了，需要手动 break
            if current_generated >= num_to_generate: break
            # 防止因 batch_size=0 导致的死循环
            if batch_size == 0: break

    if not synthetic_X: return np.array([]), np.array([])
    synthetic_X = np.concatenate(synthetic_X, axis=0)
    synthetic_y = np.full(synthetic_X.shape[0], y_minority[0])
    return synthetic_X, synthetic_y


# --- [MODIFIED] 重命名：只加载纯净数据并训练 GAN ---
def load_pure_data_and_train_gan():
    """加载、预处理纯净数据，并训练 VACWGAN"""
    f_paths, raw_labels = [], []
    raw_class_dirs = sorted([d for d in os.listdir(TRAIN_ROOT_DIR) if os.path.isdir(os.path.join(TRAIN_ROOT_DIR, d))])
    num_classes = len(CLEAN_CLASS_NAMES)

    for raw_name in raw_class_dirs:
        class_dir = os.path.join(TRAIN_ROOT_DIR, raw_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(".xlsx"):
                f_paths.append(os.path.join(class_dir, fname))
                raw_labels.append(raw_name)

    print(f"开始对 {len(f_paths)} 个[纯净]样本进行 VMD+SSA+STFT 预处理 (并行, {cpu_count()} 核心)...")
    input_tuples = list(zip(f_paths, raw_labels))
    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(process_single_signal_worker, input_tuples), total=len(input_tuples), desc="并行处理信号"))

    X_processed = np.array([res[0] for res in results])
    y_labels = np.array([res[1] for res in results])
    valid_mask = y_labels != -1
    X_processed = X_processed[valid_mask]
    y_labels = y_labels[valid_mask]
    print(f"纯净数据预处理完成。样本形状: {X_processed.shape}，标签形状: {y_labels.shape}")

    # 3. 训练 VACWGAN (在所有纯净数据上训练)
    full_dataset = PreprocessedDataset(X_processed, y_labels)
    full_data_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    E_trained, G_trained = train_vacwgan_for_augmentation(full_data_loader, num_classes, DEVICE, VACWGAN_EPOCHS)

    # --- [MODIFIED] ---
    # 移除样本合成步骤，只返回纯净数据和训练好的 GAN
    print(f"纯净数据加载和 GAN 训练完毕。")
    return X_processed, y_labels, E_trained, G_trained, CLEAN_CLASS_NAMES


# --- 详细指标函数 (无修改) ---
def output_detailed_metrics(y_true, y_pred, class_names, output_filepath=None):
    report_lines = []
    report_lines.append("--- 详细性能报告 ---")
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    report_lines.append("\n--- 整体宏平均指标 ---")
    report_lines.append(f"总体准确率 (Accuracy): {accuracy:.3f}")
    report_lines.append(f"宏平均F1分数 (Macro-F1): {macro_f1:.3f}")
    report_lines.append(f"宏平均精确率 (Macro-Precision): {macro_precision:.3f}")
    report_lines.append(f"宏平均召回率 (Macro-Recall): {macro_recall:.3f}")
    report_lines.append("\n--- 各类别详细指标 (Precision, Recall, F1-Score) ---")
    class_report = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    report_lines.append(class_report)
    cm = confusion_matrix(y_true, y_pred)
    report_lines.append("\n--- 混淆矩阵 ---")
    report_lines.append(f"类别索引: {CLEAN_LABEL_MAP}")
    report_lines.append("矩阵 (行: 真实标签, 列: 预测标签):")
    report_lines.append(str(cm))
    print("\n".join(report_lines))
    if output_filepath:
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            print(f"\n✅ 详细性能报告已导出至: {output_filepath}")
        except Exception as e:
            print(f"\n⚠️ 导出性能报告失败: {e}")


# --- [MODIFIED] Optuna 优化目标函数 (在循环内增强，在纯净集验证) ---
def objective_cv(trial, X_all_pure, y_all_pure, E_trained, G_trained, num_classes, device):
    """
    Optuna优化目标函数（K折交叉验证）
    [MODIFIED] 传入纯净数据和训练好的 GAN
    """
    # --- 超参数搜索空间 (不变) ---
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 2.0)
    use_focal_loss = trial.suggest_categorical('use_focal', [True, False])
    use_label_smoothing = trial.suggest_categorical('use_ls', [True, False])
    use_center_loss = trial.suggest_categorical('use_center', [True, False])
    focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.5) if use_focal_loss else 0.25
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0) if use_focal_loss else 2.0
    label_smoothing_factor = trial.suggest_float('ls_factor', 0.05, 0.2) if use_label_smoothing else 0.1
    center_loss_weight = trial.suggest_float('center_w', 0.001, 0.01, log=True) if use_center_loss else 0.003

    try:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_accuracies = []
        global_step = 0

        # --- [MODIFIED] ---
        # 拆分纯净数据
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all_pure, y_all_pure)):
            X_train_pure, y_train_pure = X_all_pure[train_idx], y_all_pure[train_idx]
            X_val_pure, y_val_pure = X_all_pure[val_idx], y_all_pure[val_idx]

            # --- [NEW] 在 K-Fold 循环内部进行数据增强 ---
            X_synthetic_list = [X_train_pure]
            y_synthetic_list = [y_train_pure]
            class_counts_fold = Counter(y_train_pure)

            # 找到需要增强的类别（基于当前训练折）
            for label_int in np.unique(y_train_pure):
                if class_counts_fold.get(label_int, 0) < VACWGAN_SAMPLES_PER_CLASS:
                    X_minority = X_train_pure[y_train_pure == label_int]
                    y_minority = y_train_pure[y_train_pure == label_int]

                    num_to_generate = VACWGAN_SAMPLES_PER_CLASS  # (简单起见，统一生成50个)

                    X_synth, y_synth = generate_synthetic_samples_from_trained(
                        E_trained, G_trained, X_minority, y_minority, num_to_generate, DEVICE
                    )

                    if X_synth.size > 0:
                        X_synthetic_list.append(X_synth)
                        y_synthetic_list.append(y_synth)

            X_train_aug = np.concatenate(X_synthetic_list, axis=0)
            y_train_aug = np.concatenate(y_synthetic_list, axis=0)
            # --- 增强结束 ---

            # --- [MODIFIED] ---
            # 归一化：Mean/Std 必须来自增强后的训练集 (X_train_aug)
            current_fold_mean = X_train_aug.mean(axis=(0, 1, 2), keepdims=True)
            current_fold_std = X_train_aug.std(axis=(0, 1, 2), keepdims=True) + 1e-8
            X_train_norm = (X_train_aug - current_fold_mean) / current_fold_std
            # 验证集使用训练集的 Mean/Std，并且是纯净数据
            X_val_norm = (X_val_pure - current_fold_mean) / current_fold_std

            model = Simple2DCNN(num_classes=num_classes, dropout=dropout_rate).to(device)
            center_feature_dim = model.center_feature_dim

            # 使用 TensorDataset (不自动加通道)
            train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.LongTensor(y_train_aug))
            val_dataset = TensorDataset(torch.FloatTensor(X_val_norm), torch.LongTensor(y_val_pure))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 权重计算应基于增强后的训练集标签
            class_weights = calculate_class_weights(y_train_aug)

            criterion = CombinedLoss(
                num_classes=num_classes, feature_dim=center_feature_dim, use_focal=use_focal_loss,
                use_label_smoothing=use_label_smoothing, use_center_loss=use_center_loss,
                focal_alpha=focal_alpha, focal_gamma=focal_gamma, label_smoothing_factor=label_smoothing_factor,
                center_loss_weight=center_loss_weight, class_weights=class_weights.to(device)
            )

            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            fold_best_acc = 0

            for epoch in range(NUM_EPOCHS):
                model.train()
                for X_batch, y_batch in train_loader:
                    # 手动添加通道维度 (C=1)，因为用的是 TensorDataset
                    X_batch = X_batch.to(device).unsqueeze(1);
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    outputs, features = model(X_batch, return_features=True)
                    loss, _ = criterion(outputs, features, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()

                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        # 手动添加通道维度 (C=1)
                        X_batch = X_batch.to(device).unsqueeze(1);
                        y_batch = y_batch.to(device)
                        outputs, features = model(X_batch, return_features=True)
                        val_loss += criterion(outputs, features, y_batch)[0].item() * X_batch.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()

                # [MODIFIED] 验证集现在是纯净的，分数更真实
                val_acc = 100 * val_correct / val_total
                val_loss /= val_total
                scheduler.step(val_loss)
                trial.report(val_acc, global_step)

                if val_acc > fold_best_acc:
                    fold_best_acc = val_acc
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                global_step += 1

            fold_accuracies.append(fold_best_acc)

        mean_acc = np.mean(fold_accuracies)

        # --- [MODIFIED] ---
        # 不再保存模型状态到 trial.user_attrs。Optuna 只负责返回平均精度。
        return mean_acc

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0


# --- [NEW] 用于训练最终集成模型的独立函数 ---
def train_final_ensemble(X_all_pure, y_all_pure, E_trained, G_trained, best_params, num_classes, optimization_dir):
    """
    使用最佳超参数，训练 N_FOLDS 个模型用于集成，并保存它们。
    """
    print("\n" + "=" * 60)
    print(f"🚀 开始训练最终 {N_FOLDS}-Fold 集成模型...")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # 用于最终验证的指标
    all_val_labels = []
    all_val_preds = []

    # 从 best_params 中提取参数
    lr = best_params['lr']
    batch_size = best_params['batch_size']
    dropout = best_params['dropout']
    weight_decay = best_params['weight_decay']
    optimizer_name = best_params['optimizer']
    max_grad_norm = best_params['max_grad_norm']

    # 提取损失函数参数
    loss_params = {
        'use_focal': best_params['use_focal'],
        'use_label_smoothing': best_params['use_ls'],
        'use_center_loss': best_params['use_center'],
        'focal_alpha': best_params.get('focal_alpha', 0.25),
        'focal_gamma': best_params.get('focal_gamma', 2.0),
        'label_smoothing_factor': best_params.get('ls_factor', 0.1),
        'center_loss_weight': best_params.get('center_w', 0.003)
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all_pure, y_all_pure)):
        print(f"\n--- 正在训练 Fold {fold_idx + 1}/{N_FOLDS} ---")

        X_train_pure, y_train_pure = X_all_pure[train_idx], y_all_pure[train_idx]
        X_val_pure, y_val_pure = X_all_pure[val_idx], y_all_pure[val_idx]

        # 1. 数据增强 (与 objective_cv 逻辑相同)
        X_synthetic_list = [X_train_pure]
        y_synthetic_list = [y_train_pure]
        class_counts_fold = Counter(y_train_pure)
        for label_int in np.unique(y_train_pure):
            if class_counts_fold.get(label_int, 0) < VACWGAN_SAMPLES_PER_CLASS:
                X_minority = X_train_pure[y_train_pure == label_int]
                y_minority = y_train_pure[y_train_pure == label_int]
                num_to_generate = VACWGAN_SAMPLES_PER_CLASS
                X_synth, y_synth = generate_synthetic_samples_from_trained(
                    E_trained, G_trained, X_minority, y_minority, num_to_generate, DEVICE
                )
                if X_synth.size > 0:
                    X_synthetic_list.append(X_synth)
                    y_synthetic_list.append(y_synth)
        X_train_aug = np.concatenate(X_synthetic_list, axis=0)
        y_train_aug = np.concatenate(y_synthetic_list, axis=0)

        # 2. 归一化 (与 objective_cv 逻辑相同)
        current_fold_mean = X_train_aug.mean(axis=(0, 1, 2), keepdims=True)
        current_fold_std = X_train_aug.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        X_train_norm = (X_train_aug - current_fold_mean) / current_fold_std
        X_val_norm = (X_val_pure - current_fold_mean) / current_fold_std

        # 3. 保存归一化统计数据
        mean_path = os.path.join(optimization_dir, f"best_model_mean_fold_{fold_idx}.npy")
        std_path = os.path.join(optimization_dir, f"best_model_std_fold_{fold_idx}.npy")
        np.save(mean_path, current_fold_mean)
        np.save(std_path, current_fold_std)
        print(f"Fold {fold_idx} 归一化统计数据已保存。")

        # 4. 初始化模型、数据和损失
        model = Simple2DCNN(num_classes=num_classes, dropout=dropout).to(DEVICE)
        center_feature_dim = model.center_feature_dim
        train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.LongTensor(y_train_aug))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_norm), torch.LongTensor(y_val_pure))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        class_weights = calculate_class_weights(y_train_aug)

        criterion = CombinedLoss(
            num_classes=num_classes, feature_dim=center_feature_dim,
            class_weights=class_weights.to(DEVICE), **loss_params
        )
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # 5. 训练模型
        fold_best_acc = 0.0
        best_model_state_fold = None

        for epoch in range(NUM_EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE).unsqueeze(1);
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs, features = model(X_batch, return_features=True)
                loss, _ = criterion(outputs, features, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            model.eval()
            val_loss_epoch = 0;
            val_correct_epoch = 0;
            val_total_epoch = 0
            current_val_labels = [];
            current_val_preds = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE).unsqueeze(1);
                    y_batch = y_batch.to(DEVICE)
                    outputs, features = model(X_batch, return_features=True)
                    val_loss_epoch += criterion(outputs, features, y_batch)[0].item() * X_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total_epoch += y_batch.size(0)
                    val_correct_epoch += (predicted == y_batch).sum().item()
                    current_val_labels.extend(y_batch.cpu().numpy())
                    current_val_preds.extend(predicted.cpu().numpy())

            val_acc_epoch = 100 * val_correct_epoch / val_total_epoch
            val_loss_epoch /= val_total_epoch
            scheduler.step(val_loss_epoch)

            if val_acc_epoch > fold_best_acc:
                fold_best_acc = val_acc_epoch
                best_model_state_fold = model.state_dict().copy()
                # 保存这一折的最佳预测结果
                all_val_labels_fold = current_val_labels
                all_val_preds_fold = current_val_preds

            if (epoch + 1) % 20 == 0:
                print(
                    f"  Epoch {epoch + 1}/{NUM_EPOCHS}, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%")

        print(f"Fold {fold_idx} 训练完毕。最佳验证准确率: {fold_best_acc:.3f}%")

        # 6. 保存当前折的最佳模型
        if best_model_state_fold:
            model_path = os.path.join(optimization_dir, f"best_model_fold_{fold_idx}.pth")
            torch.save(best_model_state_fold, model_path)
            print(f"Fold {fold_idx} 最佳模型已保存。")
            all_val_labels.extend(all_val_labels_fold)
            all_val_preds.extend(all_val_preds_fold)
        else:
            print(f"⚠️ 警告: Fold {fold_idx} 未能保存模型 (未找到最佳状态)。")

    print(f"\n✅ {N_FOLDS}-Fold 集成模型训练完毕。")
    return all_val_labels, all_val_preds


# --- [MODIFIED] 主函数，执行调优和集成训练 ---
def hyperparameter_optimization():
    """主函数，负责驱动 GAN 训练、Optuna 调优和最终集成模型训练"""
    print("\n" + "=" * 60)
    print("🔍 阶段 1: 加载纯净数据并训练 VACWGAN")
    print("=" * 60)

    # 1. 加载纯净数据并训练 GAN (一次性)
    X_all_pure, y_all_pure, E_trained, G_trained, class_names = load_pure_data_and_train_gan()
    num_classes = len(class_names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimization_dir = f"./hyperparameter_optimization_{timestamp}"
    Path(optimization_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("🔍 阶段 2: Optuna 超参数优化")
    print("=" * 60)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Optuna 目标函数需要 GAN 和纯净数据
    def objective_wrapper(trial):
        return objective_cv(
            trial, X_all_pure, y_all_pure, E_trained, G_trained, num_classes, DEVICE
        )

    print(f"\n开始优化 (共{N_TRIALS}次试验)...")
    study.optimize(
        objective_wrapper,
        n_trials=N_TRIALS,
        timeout=OPTIMIZATION_TIMEOUT,
        show_progress_bar=True
    )

    best_trial = study.best_trial
    best_params = best_trial.params

    print(f"\n✅ 优化完成! 最佳平均验证准确率 (来自Optuna): {study.best_value:.3f}%")
    print("最佳参数:", best_params)

    # 3. 保存最佳超参数
    best_params_path = os.path.join(optimization_dir, "best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✅ 最佳超参数已保存至: {best_params_path}")

    # 4. --- [NEW] 训练最终的 5-Fold 集成模型 ---
    all_val_labels, all_val_preds = train_final_ensemble(
        X_all_pure, y_all_pure, E_trained, G_trained, best_params, num_classes, optimization_dir
    )

    # 5. --- [NEW] 输出集成模型的 K-Fold 验证性能 ---
    if all_val_labels and all_val_preds:
        print("\n" + "=" * 60)
        print(f"📊 {N_FOLDS}-Fold 集成模型在 [纯净验证集] 上的聚合性能报告")
        print("=" * 60)
        val_report_path = os.path.join(optimization_dir, "ensemble_validation_performance_report.txt")
        output_detailed_metrics(np.array(all_val_labels), np.array(all_val_preds), class_names, val_report_path)
    else:
        print("\n⚠️ 未能生成集成模型的验证报告 (没有收集到标签或预测)。")

    return best_params, optimization_dir, class_names


# --- [MODIFIED] 测试集推理函数 (支持集成) ---
def run_test_inference(optimization_dir, class_names, best_params):
    """加载 5-Fold 集成模型，处理最终测试集，并输出 TXT 格式结果"""

    num_classes = len(class_names)
    dropout_rate = best_params.get('dropout', 0.5)

    # 1. 加载 5-Fold 集成模型
    models_list = []
    stats_list = []
    print("\n--- 加载 5-Fold 集成模型及统计数据 ---")

    for i in range(N_FOLDS):
        try:
            model = Simple2DCNN(num_classes=num_classes, dropout=dropout_rate).to(DEVICE)
            model_path = os.path.join(optimization_dir, f"best_model_fold_{i}.pth")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models_list.append(model)

            mean_path = os.path.join(optimization_dir, f"best_model_mean_fold_{i}.npy")
            std_path = os.path.join(optimization_dir, f"best_model_std_fold_{i}.npy")
            data_mean = np.load(mean_path)
            data_std = np.load(std_path)
            stats_list.append({'mean': data_mean, 'std': data_std})

            print(f"成功加载 Fold {i} 模型及统计数据。")
        except FileNotFoundError as e:
            print(f"\n错误: 未找到 Fold {i} 的模型或统计文件: {e}")
            print("请确保已成功运行 train_final_ensemble。")
            return

    if not models_list:
        print("错误：未能加载任何模型，推理中止。")
        return

    # 2. 预处理测试集数据 (并行)
    test_files = [f for f in os.listdir(TEST_ROOT_DIR) if f.endswith(".xlsx")]
    final_test_paths = [os.path.join(TEST_ROOT_DIR, f) for f in test_files]
    inference_input_paths = final_test_paths

    print(f"\n开始处理最终测试集数据 (并行, {cpu_count()} 核心)...")
    with Pool(cpu_count()) as pool:
        X_final_test_processed = list(
            tqdm(pool.imap(process_single_test_signal_worker, inference_input_paths), total=len(inference_input_paths),
                 desc="并行处理测试信号"))
    X_final_test_processed = np.array(X_final_test_processed)  # 形状 (N_test, H, W)

    # 3. 运行集成推理
    # 累加 logits
    all_ensemble_logits = np.zeros((len(X_final_test_processed), num_classes))

    print("\n--- 运行 5-Fold 集成推理 ---")
    for i in range(N_FOLDS):
        model = models_list[i]
        stats = stats_list[i]

        # 使用当前折的统计数据归一化
        X_final_test_norm_i = (X_final_test_processed - stats['mean']) / stats['std']

        # PreprocessedDataset 会自动添加通道维度
        final_test_dataset_i = PreprocessedDataset(X_final_test_norm_i, [0] * len(X_final_test_norm_i))
        final_test_loader_i = DataLoader(final_test_dataset_i, batch_size=64, shuffle=False)

        model_logits = []
        with torch.no_grad():
            for inputs, _ in final_test_loader_i:
                inputs = inputs.to(DEVICE)  # (B, 1, H, W)
                logits = model(inputs)
                model_logits.append(logits.cpu().numpy())

        model_logits_np = np.concatenate(model_logits, axis=0)
        all_ensemble_logits += model_logits_np
        print(f"Fold {i} 推理完成。")

    # 4. 计算平均 Logits 并获取最终预测
    avg_logits = all_ensemble_logits / N_FOLDS
    y_pred_final = np.argmax(avg_logits, axis=1)

    # 5. 导出预测结果 (使用 CLEAN_CLASS_NAMES)
    output_filename = "test_prediction_results.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("测试集名称\t故障类型\n")
        for path, pred_index in zip(final_test_paths, y_pred_final):
            base_name = os.path.basename(path)
            test_set_name = os.path.splitext(base_name)[0]
            predicted_fault_type = class_names[pred_index]
            f.write(f"{test_set_name}\t{predicted_fault_type}\n")

    print(f"\n✅ 测试集预测结果已导出至: '{output_filename}'")
    pred_counts = Counter(y_pred_final)
    print(f"\n--- 测试集预测分布概览 (共 {len(y_pred_final)} 个样本) ---")
    for pred_idx, count in pred_counts.items():
        print(f"  {class_names[pred_idx]:<20}: {count} 个")


if __name__ == '__main__':
    # --- 阶段 1 & 2 & 3: GAN训练、超参数优化、集成模型训练 ---
    best_params, optimization_dir, class_names = hyperparameter_optimization()

    # --- 阶段 4: 最终测试集集成推理 ---
    print("\n" + "=" * 60)
    print("🚀 运行最终测试集集成推理")
    print("=" * 66)
    run_test_inference(optimization_dir, class_names, best_params)