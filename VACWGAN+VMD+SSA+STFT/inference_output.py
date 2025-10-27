# ===================================================================================
# 文件名: inference_output.py
# 描述: 加载最佳模型，处理最终测试集，并输出要求的 TXT 格式结果
# ===================================================================================
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from preprocessing import VMD_Sorter_SSA_Processor, stft_transform
from model import Simple2DCNN
from evaluation import evaluate_model
import json
import random

# --- 全局设置 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- 路径和参数 ---
# 必须手动修改这些路径以匹配你的环境
TEST_ROOT_DIR = r"D:\轴承故障检测竞赛\初赛测试集"
TRAIN_ROOT_DIR = r"D:\轴承故障检测竞赛\初赛训练集"
# 假设 Optuna 优化结果目录名，请根据实际运行结果修改
OPTIMIZATION_DIR = r"./hyperparameter_optimization_YYYYMMDD_HHMMSS"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64


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


def run_test_inference(optimization_dir):
    # 1. 重新构建类别名称和标签映射
    train_class_names = sorted([d for d in os.listdir(TRAIN_ROOT_DIR) if os.path.isdir(os.path.join(TRAIN_ROOT_DIR, d))])
    label_map = {name: i for i, name in enumerate(train_class_names)}
    clean_class_names = train_class_names
    num_classes = len(clean_class_names)

    # 2. 确定最佳模型的参数
    try:
        with open(os.path.join(optimization_dir, 'best_params.json'), 'r') as f:
            best_params = json.load(f)
        dropout_rate = best_params.get('dropout_rate', 0.5)
    except:
        print("警告: 未找到最佳参数文件，使用默认模型参数。")
        dropout_rate = 0.5

    # 3. 加载最佳模型
    model = Simple2DCNN(num_classes=num_classes, dropout=dropout_rate).to(DEVICE)
    best_model_path = os.path.join(optimization_dir, "best_model.pth")

    try:
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"\n成功加载最佳模型: {best_model_path}")
    except FileNotFoundError:
        print(f"\n错误: 未找到最佳模型权重文件 {best_model_path}。请先运行 optuna_script.py 完成优化。")
        return

    # 4. 加载和预处理测试集数据
    test_files = [f for f in os.listdir(TEST_ROOT_DIR) if f.endswith(".xlsx")]
    final_test_paths = [os.path.join(TEST_ROOT_DIR, f) for f in test_files]

    # VMD+SSA 参数 (需要与训练时保持一致)
    params = {
        'K': 5,
        'alpha': 2000,
        'window_len': 128,
        'ssa_threshold': 32,
        'denoise_indices': [0, 1]
    }
    preprocessor = VMD_Sorter_SSA_Processor(**params)

    X_final_test_processed = []
    print(f"\n开始处理最终测试集数据: {len(final_test_paths)} 个样本...")
    for path in tqdm(final_test_paths, desc="处理测试信号"):
        # 测试集读取方式与训练集相同，只读取第二列
        signal = pd.read_excel(path, header=None, usecols=[1]).squeeze("columns").to_numpy()
        denoised_signal = preprocessor.process(signal)
        stft_map = stft_transform(denoised_signal)
        X_final_test_processed.append(stft_map)

    X_final_test_processed = np.array(X_final_test_processed)

    # 5. 运行推理
    # 由于测试集没有真实标签，我们使用虚拟标签
    final_test_dataset = PreprocessedDataset(X_final_test_processed, [0] * len(X_final_test_processed))
    final_test_loader = DataLoader(final_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in final_test_loader:
            inputs = inputs.to(DEVICE).unsqueeze(1)  # 添加通道维度
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    y_pred_final = np.array(all_preds)

    # 6. 导出预测结果 (严格按照 use_output.py 格式)
    output_filename = "test_prediction_results.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("测试集名称\t故障类型\n")

        for path, pred_index in zip(final_test_paths, y_pred_final):
            base_name = os.path.basename(path)
            test_set_name = os.path.splitext(base_name)[0]

            # 预测类别名称
            predicted_fault_type = clean_class_names[pred_index]

            f.write(f"{test_set_name}\t{predicted_fault_type}\n")

    print(f"\n✅ 测试集预测结果已导出至: '{output_filename}'")

    # 打印前几个结果示例
    print("\n--- 预测结果示例 ---")
    with open(output_filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 6:
                print(line.strip())
            if i == 5:
                break


if __name__ == '__main__':
    # 请确保 OPTIMIZATION_DIR 指向正确的 Optuna 结果目录
    run_test_inference(OPTIMIZATION_DIR)