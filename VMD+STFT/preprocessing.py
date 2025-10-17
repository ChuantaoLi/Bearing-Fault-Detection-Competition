import numpy as np
from scipy.signal import stft
from vmdpy import VMD


class VMD_Sorter_Processor:
    """
    一个集成了VMD分解和频率排序的处理器（已移除SSA去噪）。
    - K: VMD分解的模态数量
    - alpha: VMD带宽约束
    """

    def __init__(self, K, alpha):
        self.K = K
        self.alpha = alpha
        # VMD固定参数
        self.tau = 0.
        self.DC = 0
        self.init = 1
        self.tol = 1e-7

    def process(self, signal):
        """
        执行 VMD -> Sort -> Reconstruct 流程
        返回:
          - final_signal: 重构的最终信号 (np.ndarray)
          - sorted_modes: 按频率从高到低排序的模态 (np.ndarray)
        """
        # 1. VMD分解
        u, _, omega = VMD(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)

        # 2. 按频率从高到低排序
        sort_order = np.argsort(omega[0, :])[::-1]
        sorted_modes = u[sort_order, :]

        # 3. 重构最终信号（所有模态的和）
        final_signal = np.sum(sorted_modes, axis=0)

        return final_signal, sorted_modes


def stft_transform(signal, nperseg=64, noverlap=32, nfft=128):
    """对信号进行短时傅里叶变换"""
    _, _, Zxx = stft(signal, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return np.abs(Zxx)