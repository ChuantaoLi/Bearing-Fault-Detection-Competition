import matplotlib.pyplot as plt
import numpy as np
import os


def plot_selected_modes(modes, sample_name):
    """
    可视化DDEF筛选和频率排序后的VMD模态。
    """
    num_modes = modes.shape[0]
    fig, axes = plt.subplots(num_modes, 1, figsize=(16, num_modes * 2.5), sharex=True)
    fig.suptitle(f'Selected and Sorted VMD Modes for {os.path.splitext(sample_name)[0]} (High to Low Freq)', fontsize=20)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    for i, mode in enumerate(modes):
        ax = axes[i] if num_modes > 1 else axes
        ax.plot(mode, color='darkcyan')
        ax.set_title(f'Sorted Mode {i + 1}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

    if num_modes > 0:
        axes[-1].set_xlabel('Time Steps', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = 'selected_vmd_modes.png'
    plt.savefig(save_path, dpi=300)
    print(f"筛选并排序后的模态图已保存至: '{save_path}'")
    plt.close()


def plot_ssa_single_mode_decomposition(original_mode, denoised_mode, noise_mode, mode_rank, sample_name):
    """
    对单个模态的SSA分解结果进行可视化。
    """
    plt.figure(figsize=(16, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.plot(original_mode, color='gray', linestyle='--', alpha=0.8, label=f'Original Mode (Freq Rank {mode_rank})')
    plt.plot(denoised_mode, color='red', linewidth=1.5, label='Principal Trend (Signal Layer)')
    plt.plot(noise_mode, color='green', linewidth=1, alpha=0.7, label='Noise Layer')

    plt.title(f'SSA Decomposition of Mode Rank {mode_rank} for {os.path.splitext(sample_name)[0]}', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = f'SSA_Mode_Rank_{mode_rank}_decomposition.png'
    plt.savefig(save_path, dpi=300)
    print(f"SSA分解图已保存至: '{save_path}'")
    plt.close()


def run_visualization_for_single_signal(preprocessor, signal, sample_name):
    """
    执行并驱动单个信号的完整可视化流程（新版）。
    """
    print(f"\n--- 开始对样本 '{sample_name}' 进行详细的可视化分析 ---")

    # 1. 执行完整的处理流程
    _, sorted_modes, denoised_info = preprocessor.process(signal)

    # 2. 可视化筛选和排序后的模态
    print("\n正在生成筛选和排序后的模态图...")
    plot_selected_modes(sorted_modes, sample_name)

    # 3. 可视化频率最高的两个模态的SSA去噪效果
    print("\n正在生成高频模态的SSA分解图...")
    for i, info in enumerate(denoised_info):
        plot_ssa_single_mode_decomposition(
            original_mode=info["original"],
            denoised_mode=info["denoised"],
            noise_mode=info["noise"],
            mode_rank=i + 1,  # 频率排序1, 2
            sample_name=sample_name
        )

    print(f"--- 可视化分析完毕 ---")

