# ===================================================================================
# 文件名: main_train_eval.py (最终修正 - 解决测试集预处理并行问题 + 修复推理Bug)
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
from evaluation import evaluate_model  # 导入评估函数
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

# --- 类别映射 (解决问题 2) ---
RAW_TO_CLEAN_MAP = {
    'inner_broken_train150': 'inner_broken',
    'inner_wear_train120': 'inner_wear',
    'normal_train160': 'normal',
    'outer_missing_train180': 'outer_missing',
    'roller_broken_train150': 'roller_broken',
    'roller_wear_train100': 'roller_wear'
}
CLEAN_CLASS_NAMES = sorted(list(set(RAW_TO_CLEAN_MAP.values())))
CLEAN_LABEL_MAP = {name: i for i, name in enumerate(CLEAN_CLASS_NAMES)}  # 干净标签的索引映射

# --- 超参数设置 ---
VACWGAN_EPOCHS = 300
VACWGAN_LR = 1e-4
VACWGAN_SAMPLES_PER_CLASS = 50
LATENT_DIM = 100

N_TRIALS = 300
NUM_EPOCHS = 100
OPTIMIZATION_TIMEOUT = 3600
D_ITERS = 5

# --- 全局预处理器参数 (用于多进程 worker) ---
GLOBAL_PREPROCESSOR_PARAMS = {
    'K': 5, 'alpha': 2000, 'window_len': 128, 'ssa_threshold': 32, 'denoise_indices': [0, 1]
}


# --- 多进程工作函数（训练集） ---
def process_single_signal_worker(path_label_tuple):
    """
    多进程池中的工作函数，对单个训练信号文件进行预处理。
    返回: (stft_map, clean_label_int)
    """
    path, raw_label_name = path_label_tuple

    # 转换为干净标签 (在 worker 进程中安全)
    clean_label_name = RAW_TO_CLEAN_MAP.get(raw_label_name, 'unknown')
    label_int = CLEAN_LABEL_MAP.get(clean_label_name, -1)

    preprocessor = VMD_Sorter_SSA_Processor(**GLOBAL_PREPROCESSOR_PARAMS)
    signal = pd.read_excel(path, header=None, usecols=[1]).squeeze("columns").to_numpy()
    denoised_signal = preprocessor.process(signal)
    stft_map = stft_transform(denoised_signal)

    return stft_map, label_int


# --- 新增：测试集的多进程工作函数（顶级函数） ---
def process_single_test_signal_worker(path):
    """
    专门用于测试集的 worker 函数，只接收路径，只返回处理后的 STFT 图像。
    解决多进程局部对象错误。
    """
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
    """
    完整训练 VACWGAN 网络 (E, G, D, C)
    """
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

            # --- 训练 Discriminator 和 Classifier (D, C) ---
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

            # --- 训练 Generator 和 Encoder (G, E) ---
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

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, D_Loss: {loss_D.item():.4f}, C_Loss: {loss_C.item():.4f}, G_Loss: {loss_G_total.item():.4f}, KL_Loss: {loss_E_kl.item():.4f}")

    print("--- VACWGAN 对抗训练结束 ---")
    return E, G


def load_and_augment_data():
    """加载、预处理、训练 VACWGAN 并进行样本增强"""
    f_paths, raw_labels = [], []

    # 仅获取原始文件夹名称，用于映射
    raw_class_dirs = sorted([d for d in os.listdir(TRAIN_ROOT_DIR) if os.path.isdir(os.path.join(TRAIN_ROOT_DIR, d))])
    num_classes = len(CLEAN_CLASS_NAMES)

    # 1. 加载所有数据路径和原始文件夹名称
    for raw_name in raw_class_dirs:
        class_dir = os.path.join(TRAIN_ROOT_DIR, raw_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(".xlsx"):
                f_paths.append(os.path.join(class_dir, fname))
                raw_labels.append(raw_name)  # 存储原始文件夹名

    # 2. 预处理所有数据 - 使用 Multiprocessing 进行并行计算
    print(f"开始对 {len(f_paths)} 个样本进行 VMD+SSA+STFT 预处理 (并行计算, {cpu_count()} 核心)...")

    # input_tuples 包含 (路径, 原始文件夹名称)
    input_tuples = list(zip(f_paths, raw_labels))

    with Pool(cpu_count()) as pool:
        # process_single_signal_worker 返回 (stft_map, clean_label_int)
        results = list(tqdm(pool.imap(process_single_signal_worker, input_tuples), total=len(input_tuples), desc="并行处理信号"))

    X_processed = np.array([res[0] for res in results])
    y_labels = np.array([res[1] for res in results])
    print(f"数据预处理完成。样本形状: {X_processed.shape[1:]}")

    # 过滤掉标签为 -1 的异常样本
    valid_mask = y_labels != -1
    X_processed = X_processed[valid_mask]
    y_labels = y_labels[valid_mask]

    # 3. 训练 VACWGAN
    # 注意：PreprocessedDataset 会自动在 __getitem__ 中添加通道维度 (N, C, H, W)
    full_dataset = PreprocessedDataset(X_processed, y_labels)
    full_data_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    E_trained, G_trained = train_vacwgan_for_augmentation(full_data_loader, num_classes, DEVICE, VACWGAN_EPOCHS)

    # 4. 识别稀缺类并生成样本
    class_counts = Counter(y_labels)
    majority_count = max(class_counts.values())

    minority_classes_labels = {}
    for raw_name, clean_name in RAW_TO_CLEAN_MAP.items():
        label_int = CLEAN_LABEL_MAP[clean_name]
        # 只要是稀缺类，都添加到生成目标
        if class_counts.get(label_int, 0) < majority_count:
            minority_classes_labels[label_int] = clean_name

    # 5. 样本合成增强
    X_synthetic_list = [X_processed]
    y_synthetic_list = [y_labels]

    print("\n--- VACWGAN 样本合成/增强阶段 ---")

    for label_int in minority_classes_labels.keys():
        X_minority = X_processed[y_labels == label_int]
        y_minority = y_labels[y_labels == label_int]

        num_to_generate = VACWGAN_SAMPLES_PER_CLASS

        def generate_synthetic_samples_from_trained(E_trained, G_trained, X_minority, y_minority, num_to_generate, device):
            """使用训练好的 E 和 G 进行样本生成"""
            if X_minority.size == 0 or num_to_generate == 0: return np.array([]), np.array([])
            # 使用 PreprocessedDataset 以匹配训练时的输入 (N, C, H, W)
            minority_data = PreprocessedDataset(X_minority, y_minority)
            minority_loader = DataLoader(minority_data, batch_size=64, shuffle=True)
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
                        # fake_samples 已经是 (N_batch, 1, H, W)，我们需要 (N_batch, H, W) 存入Numpy
                        synthetic_X.append(fake_samples.squeeze(1));
                        current_generated += num_batch
                        if current_generated >= num_to_generate: break
            synthetic_X = np.concatenate(synthetic_X, axis=0)
            synthetic_y = np.full(synthetic_X.shape[0], y_minority[0])
            return synthetic_X, synthetic_y

        X_synth, y_synth = generate_synthetic_samples_from_trained(
            E_trained, G_trained, X_minority, y_minority, num_to_generate, DEVICE
        )

        if X_synth.size > 0:
            X_synthetic_list.append(X_synth)
            y_synthetic_list.append(y_synth)

    X_augmented = np.concatenate(X_synthetic_list, axis=0)
    y_augmented = np.concatenate(y_synthetic_list, axis=0)

    print(f"样本增强完毕。原始总数: {len(y_labels)}, 增强后总数: {len(y_augmented)}")

    return X_augmented, y_augmented, CLEAN_CLASS_NAMES


# --- Optuna 优化目标函数 (已修复警告 + 修复Bug) ---
def objective_cv(trial, X_all, y_all, num_classes, device, optimization_dir, n_folds=5):
    """Optuna优化目标函数（5折交叉验证）"""

    # --- 超参数搜索空间 (略) ---
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
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        fold_accuracies = []
        global_step = 0

        best_trial_fold_state = None
        best_trial_fold_acc = -1.0

        # 用于记录最佳 fold 的标签和预测结果 (用于详细评估)
        best_val_labels = []
        best_val_preds = []

        # --- (修复 Bug) 新增：用于存储最佳折的统计数据 ---
        best_fold_mean = None
        best_fold_std = None

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_val, y_val = X_all[val_idx], y_all[val_idx]

            # --- (修复 Bug) 关键：捕获当前折的统计数据 ---
            current_fold_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
            current_fold_std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8

            # 应用归一化
            X_train_norm = (X_train - current_fold_mean) / current_fold_std
            X_val_norm = (X_val - current_fold_mean) / current_fold_std

            model = Simple2DCNN(num_classes=num_classes, dropout=dropout_rate).to(device)
            center_feature_dim = model.center_feature_dim

            # --- 注意：使用 TensorDataset，它不会自动添加通道维度 ---
            train_dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val_norm), torch.LongTensor(y_val))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            class_weights = calculate_class_weights(y_train)

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

            # --- (修复 Bug) 新增：用于存储当前折的最佳状态 ---
            current_fold_best_state = None
            current_fold_best_labels = []
            current_fold_best_preds = []

            for epoch in range(NUM_EPOCHS):
                model.train()
                for X_batch, y_batch in train_loader:
                    # --- (修复 Bug) 手动添加通道维度 (C=1)，因为用的是 TensorDataset ---
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
                current_val_labels = [];
                current_val_preds = []

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        # --- (修复 Bug) 手动添加通道维度 (C=1) ---
                        X_batch = X_batch.to(device).unsqueeze(1);
                        y_batch = y_batch.to(device)
                        outputs, features = model(X_batch, return_features=True)
                        val_loss += criterion(outputs, features, y_batch)[0].item() * X_batch.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()
                        current_val_labels.extend(y_batch.cpu().numpy())
                        current_val_preds.extend(predicted.cpu().numpy())

                val_acc = 100 * val_correct / val_total
                val_loss /= val_total
                scheduler.step(val_loss)

                trial.report(val_acc, global_step)

                # --- (修复 Bug) 逻辑修正：保存当前折的最佳模型状态 ---
                if val_acc > fold_best_acc:
                    fold_best_acc = val_acc
                    current_fold_best_state = model.state_dict().copy()
                    current_fold_best_labels = current_val_labels
                    current_fold_best_preds = current_val_preds

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                global_step += 1

            fold_accuracies.append(fold_best_acc)

            # --- (修复 Bug) 逻辑修正：比较当前折的最佳acc与试验的全局最佳acc ---
            if fold_best_acc > best_trial_fold_acc:
                best_trial_fold_acc = fold_best_acc
                best_trial_fold_state = current_fold_best_state  # 保存最佳折的最佳状态
                best_val_labels = current_fold_best_labels  # 保存对应的标签
                best_val_preds = current_fold_best_preds  # 保存对应的预测

                # --- (修复 Bug) 保存与最佳模型匹配的归一化统计数据 ---
                best_fold_mean = current_fold_mean
                best_fold_std = current_fold_std

        mean_acc = np.mean(fold_accuracies)

        if best_trial_fold_state is not None:
            trial.set_user_attr("best_model_state", best_trial_fold_state)
            trial.set_user_attr("best_val_labels", best_val_labels)
            trial.set_user_attr("best_val_preds", best_val_preds)

            # --- (修复 Bug) 将统计数据存入试验属性 ---
            if best_fold_mean is not None and best_fold_std is not None:
                trial.set_user_attr("best_model_mean", best_fold_mean.tolist())
                trial.set_user_attr("best_model_std", best_fold_std.tolist())

        return mean_acc

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0


# --- (满足请求 1) 重写详细指标函数 ---
def output_detailed_metrics(y_true, y_pred, class_names, output_filepath=None):
    """
    输出并导出详细的分类指标和混淆矩阵 (解决问题 3)
    """
    report_lines = []

    report_lines.append("--- 验证集详细性能报告 ---")

    # 宏平均指标
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    report_lines.append("\n--- 整体宏平均指标 ---")
    report_lines.append(f"总体准确率 (Accuracy): {accuracy:.3f}")
    report_lines.append(f"宏平均F1分数 (Macro-F1): {macro_f1:.3f}")
    report_lines.append(f"宏平均精确率 (Macro-Precision): {macro_precision:.3f}")
    report_lines.append(f"宏平均召回率 (Macro-Recall): {macro_recall:.3f}")

    # 分类报告 (包含每个类别的 P, R, F1)
    report_lines.append("\n--- 各类别详细指标 (Precision, Recall, F1-Score) ---")
    # 使用 classification_report 自动生成 P, R, F1
    class_report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )
    report_lines.append(class_report)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    report_lines.append("\n--- 混淆矩阵 ---")
    report_lines.append(f"类别索引: {CLEAN_LABEL_MAP}")
    report_lines.append("矩阵 (行: 真实标签, 列: 预测标签):")
    report_lines.append(str(cm))

    # --- 打印到控制台 ---
    print("\n".join(report_lines))

    # --- (满足请求 1) 导出到文件 ---
    if output_filepath:
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            print(f"\n✅ 详细性能报告已导出至: {output_filepath}")
        except Exception as e:
            print(f"\n⚠️ 导出性能报告失败: {e}")


def hyperparameter_optimization():
    """主函数，负责驱动 VACWGAN 训练、增强和 Optuna 调优"""
    print("\n" + "=" * 60)
    print("🔍 开始超参数优化（5折交叉验证）")
    print("=" * 60)

    X_all, y_all, class_names = load_and_augment_data()
    num_classes = len(class_names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimization_dir = f"./hyperparameter_optimization_{timestamp}"
    Path(optimization_dir).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    def objective_wrapper(trial):
        return objective_cv(
            trial, X_all, y_all, num_classes, DEVICE, optimization_dir, n_folds=5
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

    # 4. 保存最佳模型
    if "best_model_state" in best_trial.user_attrs:
        best_model_state = best_trial.user_attrs["best_model_state"]
        best_model_path = os.path.join(optimization_dir, "best_model.pth")
        torch.save(best_model_state, best_model_path)
        print(f"\n✅ 最佳模型权重已保存至: {best_model_path}")

        # --- (修复 Bug) 新增：保存归一化统计数据 ---
        if "best_model_mean" in best_trial.user_attrs and "best_model_std" in best_trial.user_attrs:
            best_mean = np.array(best_trial.user_attrs["best_model_mean"])
            best_std = np.array(best_trial.user_attrs["best_model_std"])
            np.save(os.path.join(optimization_dir, "best_model_mean.npy"), best_mean)
            np.save(os.path.join(optimization_dir, "best_model_std.npy"), best_std)
            print(f"✅ 最佳模型对应的归一化统计数据已保存。")
        else:
            print(f"⚠️ 警告: 未找到最佳模型的归一化统计数据。测试集推理可能失败。")

        # --- (满足请求 1) 修正：输出并导出验证集详细性能报告 ---
        if "best_val_labels" in best_trial.user_attrs:
            y_true_val = np.array(best_trial.user_attrs["best_val_labels"])
            y_pred_val = np.array(best_trial.user_attrs["best_val_preds"])

            # 定义报告路径
            val_report_path = os.path.join(optimization_dir, "validation_performance_report.txt")
            # 调用新函数
            output_detailed_metrics(y_true_val, y_pred_val, class_names, val_report_path)
    else:
        print(f"\n⚠️ 未找到最佳模型状态字典。请检查 Optuna 运行情况。")

    # 5. 保存最佳超参数
    best_params_path = os.path.join(optimization_dir, "best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✅ 最佳超参数已保存至: {best_params_path}")

    print(f"\n✅ 优化完成! 最佳平均验证准确率: {study.best_value:.3f}%")

    return best_params, optimization_dir, class_names


# --- (修复 Bug) 修正测试集推理函数 ---
def run_test_inference(optimization_dir, class_names, best_params):
    """加载最佳模型，处理最终测试集，并输出要求的 TXT 格式结果 (已修复并行预处理和标签问题)"""

    # 1. 加载最佳模型
    num_classes = len(class_names)
    dropout_rate = best_params.get('dropout', 0.5)
    model = Simple2DCNN(num_classes=num_classes, dropout=dropout_rate).to(DEVICE)
    best_model_path = os.path.join(optimization_dir, "best_model.pth")

    try:
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"\n成功加载最佳模型: {best_model_path}")
    except FileNotFoundError:
        print(f"\n错误: 未找到最佳模型权重文件。请检查路径。")
        return

    # --- (修复 Bug) 新增：加载归一化统计数据 ---
    mean_path = os.path.join(optimization_dir, "best_model_mean.npy")
    std_path = os.path.join(optimization_dir, "best_model_std.npy")
    try:
        data_mean = np.load(mean_path)
        data_std = np.load(std_path)
        print("成功加载归一化统计数据 (mean/std)。")
    except FileNotFoundError:
        print(f"\n错误: 未找到归一化统计文件 (mean/std)。")
        print(f"请确保 'best_model_mean.npy' 和 'best_model_std.npy' 位于 {optimization_dir} 目录下。")
        return

    # 2. 预处理测试集数据 (使用 Multiprocessing 进行并行计算)
    test_files = [f for f in os.listdir(TEST_ROOT_DIR) if f.endswith(".xlsx")]
    final_test_paths = [os.path.join(TEST_ROOT_DIR, f) for f in test_files]

    # 构建多进程输入列表：只包含路径 (用于 process_single_test_signal_worker)
    inference_input_paths = final_test_paths

    print(f"\n开始处理最终测试集数据 (并行计算, {cpu_count()} 核心)...")

    # 调用顶级的 process_single_test_signal_worker
    with Pool(cpu_count()) as pool:
        X_final_test_processed = list(tqdm(pool.imap(process_single_test_signal_worker, inference_input_paths), total=len(inference_input_paths), desc="并行处理测试信号"))

    X_final_test_processed = np.array(X_final_test_processed)

    # --- (修复 Bug) 新增：对测试集应用归一化 ---
    print("应用归一化到测试集...")
    X_final_test_norm = (X_final_test_processed - data_mean) / data_std

    # 3. 运行推理
    # --- (修复 Bug) 使用归一化后的数据 (X_final_test_norm) ---
    # PreprocessedDataset 会自动添加通道维度
    final_test_dataset = PreprocessedDataset(X_final_test_norm, [0] * len(X_final_test_norm))
    final_test_loader = DataLoader(final_test_dataset, batch_size=64, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in final_test_loader:
            # inputs 已经具有 (N, 1, H, W) 形状，由 PreprocessedDataset 提供
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    y_pred_final = np.array(all_preds)

    # 4. 导出预测结果 (使用 CLEAN_CLASS_NAMES)
    output_filename = "test_prediction_results.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("测试集名称\t故障类型\n")

        for path, pred_index in zip(final_test_paths, y_pred_final):
            base_name = os.path.basename(path)
            test_set_name = os.path.splitext(base_name)[0]

            # 修正：使用 CLEAN_CLASS_NAMES 列表进行索引映射
            predicted_fault_type = class_names[pred_index]

            f.write(f"{test_set_name}\t{predicted_fault_type}\n")

    print(f"\n✅ 测试集预测结果已导出至: '{output_filename}'")

    # 额外检查：确认预测结果是否多样化
    pred_counts = Counter(y_pred_final)
    print(f"\n--- 测试集预测分布概览 (共 {len(y_pred_final)} 个样本) ---")
    for pred_idx, count in pred_counts.items():
        print(f"  {class_names[pred_idx]:<20}: {count} 个")


if __name__ == '__main__':
    # --- 阶段 1: 样本增强与超参数优化 ---
    best_params, optimization_dir, class_names = hyperparameter_optimization()

    # --- 阶段 2: 最终测试集评估 ---
    print("\n" + "=" * 60)
    print("🚀 运行最终测试集推理")
    print("=" * 60)
    run_test_inference(optimization_dir, class_names, best_params)