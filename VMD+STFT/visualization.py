import matplotlib.pyplot as plt
import numpy as np
import os


def plot_vmd_results_with_original(original_signal, modes, sample_name):
    """
    可视化原始序列以及VMD筛选和频率排序后的模态。
    """
    num_modes = modes.shape[0]
    # +1 for the original signal
    fig, axes = plt.subplots(num_modes + 1, 1, figsize=(16, (num_modes + 1) * 2.5), sharex=True)
    fig.suptitle(f'VMD Decomposition for {os.path.splitext(sample_name)[0]} (Original + Sorted Modes)', fontsize=20)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 确保 axes 是一个可迭代的列表
    if num_modes == 0:
        axes = [axes]
    elif num_modes == 1:
        axes = list(axes)

    # 1. 绘制原始信号
    axes[0].plot(original_signal, color='black', linewidth=1.5, label='Original Signal')
    axes[0].set_title('1. Original Signal', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. 绘制 VMD 模态
    for i, mode in enumerate(modes):
        ax_index = i + 1
        axes[ax_index].plot(mode, color='darkcyan')
        axes[ax_index].set_title(f'Sorted Mode {i + 1} (High to Low Freq)', fontsize=14)
        axes[ax_index].grid(True, linestyle='--', alpha=0.6)

    if num_modes >= 0:
        axes[-1].set_xlabel('Time Steps', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = 'vmd_decomposition_with_original.png'
    plt.savefig(save_path, dpi=300)
    print(f"VMD分解图（含原始序列）已保存至: '{save_path}'")
    plt.close()


def run_visualization_for_single_signal(preprocessor, original_signal, sample_name):
    """
    执行并驱动单个信号的完整可视化流程（仅VMD模态）。
    """
    print(f"\n--- 开始对样本 '{sample_name}' 进行详细的可视化分析 ---")

    # 1. 执行完整的处理流程，只取排序后的模态
    _, sorted_modes = preprocessor.process(original_signal)

    # 2. 可视化筛选和排序后的模态和原始序列
    print("\n正在生成 VMD 分解图（含原始序列）...")
    plot_vmd_results_with_original(original_signal, sorted_modes, sample_name)

    print(f"--- 可视化分析完毕 ---")