import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from preprocessing import VMD_Sorter_Processor, stft_transform
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
from model import Simple2DCNN
from evaluation import calculate_metrics, plot_confusion_matrix, evaluate_model
import random
from multiprocessing import Pool, cpu_count
from visualization import run_visualization_for_single_signal

# 全局设置
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


class PreprocessedDataset(Dataset):
    """
    用于封装预处理完成的时频谱图和标签的 PyTorch Dataset
    """

    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # 增加一个维度作为 Channel (C, H, W)，并转为 Float Tensor
        signal = torch.from_numpy(self.signals[idx]).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label


# --- 参数和路径设置 ---
TRAIN_ROOT_DIR = r"D:\轴承故障检测竞赛\初赛训练集"
TEST_ROOT_DIR = r"D:\轴承故障检测竞赛\初赛测试集"

TEST_SIZE, RANDOM_STATE = 0.2, 42
BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS = 64, 0.001, 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- VMD 参数 ---
params = {
    'K': 5,
    'alpha': 2000,
}


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """用于执行模型训练的函数"""
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


# --- 【核心修改】数据加载和划分 ---
# 【修改点 1】: 定义显式映射字典
RAW_TO_CLEAN_MAP = {
    'inner_broken_train150': 'inner_broken',
    'inner_wear_train120': 'inner_wear',
    'normal_train160': 'normal',
    'outer_missing_train180': 'outer_missing',
    'roller_broken_train150': 'roller_broken',
    'roller_wear_train100': 'roller_wear'
}

# 确保所有原始目录都在训练集中存在
raw_class_dirs = sorted([d for d in os.listdir(TRAIN_ROOT_DIR) if os.path.isdir(os.path.join(TRAIN_ROOT_DIR, d))])

# 检查是否存在未映射的目录
if any(d not in RAW_TO_CLEAN_MAP for d in raw_class_dirs):
    unmapped_dirs = [d for d in raw_class_dirs if d not in RAW_TO_CLEAN_MAP]
    print(f"警告: 训练集中存在未映射的目录，它们将被跳过或导致错误: {unmapped_dirs}")

# 提取纯净类别名称并排序，用于模型输出
clean_class_names = sorted(list(set(RAW_TO_CLEAN_MAP.values())))

# 建立 纯净类别名 到 索引 的映射
label_map = {name: i for i, name in enumerate(clean_class_names)}

f_paths, labels = [], []
# 遍历映射字典，加载数据并赋予标签
for raw_name, clean_name in RAW_TO_CLEAN_MAP.items():
    class_dir = os.path.join(TRAIN_ROOT_DIR, raw_name)
    # 检查目录是否存在
    if not os.path.exists(class_dir):
        print(f"警告: 映射中的目录 '{raw_name}' 不存在，跳过。")
        continue

    label_int = label_map[clean_name]

    for fname in os.listdir(class_dir):
        if fname.endswith(".xlsx"):
            f_paths.append(os.path.join(class_dir, fname))
            labels.append(label_int)

# 划分训练集和验证集路径
train_paths, val_paths, train_labels, val_labels = train_test_split(
    f_paths, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
)

# 获取最终测试集文件路径 (无标签)
final_test_paths = []
for fname in sorted(os.listdir(TEST_ROOT_DIR)):
    if fname.endswith(".xlsx"):
        final_test_paths.append(os.path.join(TEST_ROOT_DIR, fname))


def worker_init(seed):
    """在每个子进程启动时，重新设置 numpy/random 的随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def process_single_file(path):
    """
    单个文件处理函数，用于 Pool.imap。
    返回: STFT 结果 (np.ndarray)
    """
    local_preprocessor = VMD_Sorter_Processor(**params)
    signal = pd.read_excel(path, header=None, usecols=[1]).squeeze("columns").to_numpy()
    final_signal, _ = local_preprocessor.process(signal)
    stft_map = stft_transform(final_signal)
    return stft_map


# --- 主执行块 ---
if __name__ == '__main__':

    # --- 1. 可视化第一个测试样本 ---
    if final_test_paths:
        first_sample_path = final_test_paths[0]
        first_sample_name = os.path.basename(first_sample_path)
        signal_to_visualize = pd.read_excel(first_sample_path, header=None, usecols=[1]).squeeze("columns").to_numpy()
        visual_preprocessor = VMD_Sorter_Processor(**params)

        run_visualization_for_single_signal(visual_preprocessor, signal_to_visualize, first_sample_name)
    else:
        print("未找到任何最终测试文件，跳过可视化。")

    # --- 2. 数据预处理 (并行化) ---
    num_cores = cpu_count()
    print(f"\n使用 {num_cores} 个核心进行并行处理...")

    all_paths = train_paths + val_paths + final_test_paths
    train_count, val_count, test_count = len(train_paths), len(val_paths), len(final_test_paths)

    with Pool(num_cores, initializer=worker_init, initargs=(SEED,)) as pool:
        all_processed = list(tqdm(
            pool.imap(process_single_file, all_paths),
            total=len(all_paths),
            desc="并行处理所有信号"
        ))

    X_train_processed = np.array(all_processed[:train_count])
    X_val_processed = np.array(all_processed[train_count:train_count + val_count])
    X_final_test_processed = np.array(all_processed[train_count + val_count:])

    print("\n所有数据预处理完成。")

    # --- 3. 创建Dataset和DataLoader ---
    train_dataset = PreprocessedDataset(X_train_processed, train_labels)
    val_dataset = PreprocessedDataset(X_val_processed, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"输入模型的时频谱图尺寸: {train_dataset[0][0].shape}")

    # --- 4. 模型训练与评估 ---
    # 【修改点 2】使用 clean_class_names 作为最终的类别名称
    num_classes = len(clean_class_names)
    model = Simple2DCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n使用设备: {DEVICE}")
    start_time = time.time()
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
    end_time = time.time()
    print(f"\n训练耗时 {end_time - start_time:.2f} 秒。")

    # 在验证集上评估性能
    print("\n--- 在验证集上评估 ---")
    y_true_val, y_pred_val = evaluate_model(model, val_loader, DEVICE)
    calculate_metrics(y_true_val, y_pred_val)
    plot_confusion_matrix(y_true_val, y_pred_val, clean_class_names)

    # 最终测试集预测
    final_test_dataset = PreprocessedDataset(X_final_test_processed, [0] * len(X_final_test_processed))
    final_test_loader = DataLoader(final_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    _, y_pred_final = evaluate_model(model, final_test_loader, DEVICE)

    # --- 5. 整理测试集输出格式并保存到 TXT 文件 ---
    output_filename = "test_prediction_results.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("测试集名称\t故障类型\n")

        for path, pred_index in zip(final_test_paths, y_pred_final):
            # 1. 提取文件名，并移除 .xlsx 后缀
            base_name = os.path.basename(path)
            test_set_name = os.path.splitext(base_name)[0]

            # 2. 映射到预测故障类型 (使用 clean_class_names)
            predicted_fault_type = clean_class_names[pred_index]

            # 3. 写入文件
            f.write(f"{test_set_name}\t{predicted_fault_type}\n")
