import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from preprocessing import VMD_Sorter_SSA_Processor, stft_transform  # <-- 修改点
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
from model import Simple2DCNN
from evaluation import calculate_metrics, plot_confusion_matrix, evaluate_model
import random

# 全局设置
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


class PreprocessedDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx]).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label


# --- 参数和路径设置 ---
ROOT_DIR = r"D:\轴承故障检测竞赛\初赛训练集"
TEST_SIZE, RANDOM_STATE = 0.2, 42
BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS = 64, 0.001, 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 【核心修改点】VMD+Sort+SSA 参数 ---
# 在这里您可以轻松地进行实验，找到最佳的去噪组合
params = {
    'K': 5,  # VMD分解数量，回到原来效果好的设置
    'alpha': 2000,
    'window_len': 128,
    'ssa_threshold': 32,  # 为所有目标模态统一设置一个阈值
    # 指定要去噪的模态索引 (基于从高到低的频率排序)
    # [0, 1] = 去噪最高频和次高频的两个模态
    'denoise_indices': [0, 1]
}

# --- 加载数据路径 ---
f_paths, labels = [], []
class_names = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
label_map = {name: i for i, name in enumerate(class_names)}
for class_name, label_int in label_map.items():
    class_dir = os.path.join(ROOT_DIR, class_name)
    for fname in os.listdir(class_dir):
        if fname.endswith(".xlsx"):
            f_paths.append(os.path.join(class_dir, fname))
            labels.append(label_int)
train_paths, test_paths, train_labels, test_labels = train_test_split(
    f_paths, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
)
print(f"数据加载完成: {len(train_paths)}个训练样本, {len(test_paths)}个测试样本。")

# --- 数据预处理 ---
preprocessor = VMD_Sorter_SSA_Processor(**params)  # <-- 修改点
X_train_processed, X_test_processed = [], []

print("\n开始处理训练数据...")
for path in tqdm(train_paths, desc="处理训练信号"):
    signal = pd.read_excel(path, header=None, usecols=[1]).squeeze("columns").to_numpy()
    denoised_signal = preprocessor.process(signal)  # <-- 修改点
    stft_map = stft_transform(denoised_signal)
    X_train_processed.append(stft_map)

print("\n开始处理测试数据...")
for path in tqdm(test_paths, desc="处理测试信号"):
    signal = pd.read_excel(path, header=None, usecols=[1]).squeeze("columns").to_numpy()
    denoised_signal = preprocessor.process(signal)  # <-- 修改点
    stft_map = stft_transform(denoised_signal)
    X_test_processed.append(stft_map)

print("\n数据预处理完成。")

# --- 创建Dataset和DataLoader ---
X_train_processed = np.array(X_train_processed)
X_test_processed = np.array(X_test_processed)
train_dataset = PreprocessedDataset(X_train_processed, train_labels)
test_dataset = PreprocessedDataset(X_test_processed, test_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"输入模型的时频谱图尺寸: {train_dataset[0][0].shape}")

# --- 模型训练与评估 ---
model = Simple2DCNN(num_classes=len(class_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    print("\n--- 开始训练 ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * signals.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
    print("--- 训练结束 ---")


print(f"\n使用设备: {DEVICE}")
start_time = time.time()
train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
end_time = time.time()
print(f"\n训练耗时 {end_time - start_time:.2f} 秒。")

y_true, y_pred = evaluate_model(model, test_loader, DEVICE)
calculate_metrics(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, class_names)
