import numpy as np
from scipy.signal import stft
from vmdpy import VMD


class VMD_Sorter_SSA_Processor:
    """
    一个集成了VMD分解、频率排序和可控SSA去噪的处理器。
    - K: VMD分解的模态数量
    - alpha: VMD带宽约束
    - denoise_indices: 一个列表，指定要对哪些按频率排序后的模態进行去噪
                       例如 [0, 1] 表示对最高频和次高频的模態去噪。
    - window_len: SSA的嵌入窗口长度
    - ssa_threshold: SSA去噪的奇异值阈值
    """

    def __init__(self, K, alpha, denoise_indices: list, window_len, ssa_threshold):
        self.K = K
        self.alpha = alpha
        self.denoise_indices = denoise_indices
        self.window_len = window_len
        self.ssa_threshold = ssa_threshold
        # VMD固定参数
        self.tau = 0.
        self.DC = 0
        self.init = 1
        self.tol = 1e-7

    def _ssa_denoise_mode(self, mode):
        """对单个模态执行SSA去噪，返回去噪后的主趋势"""
        if self.ssa_threshold >= self.window_len:
            threshold = self.window_len - 1
        else:
            threshold = self.ssa_threshold

        series_len = len(mode)
        k = series_len - self.window_len + 1
        trajectory_matrix = np.array([mode[i:i + self.window_len] for i in range(k)]).T

        u, sigma, vt = np.linalg.svd(trajectory_matrix, full_matrices=False)

        # 重构主趋势
        reconstructed_matrix = u[:, :threshold] @ np.diag(sigma[:threshold]) @ vt[:threshold, :]

        # 对角平均
        denoised_series = np.zeros(series_len)
        counts = np.zeros(series_len)
        for i in range(self.window_len):
            for j in range(k):
                idx = i + j
                denoised_series[idx] += reconstructed_matrix[i, j]
                counts[idx] += 1
        denoised_series[counts > 0] /= counts[counts > 0]

        return denoised_series

    def process(self, signal):
        """
        执行 VMD -> Sort -> SSA -> Reconstruct 流程
        """
        # 1. VMD分解
        u, _, omega = VMD(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)

        # 2. 按频率从高到低排序
        sort_order = np.argsort(omega[0, :])[::-1]
        sorted_modes = u[sort_order, :]

        # 3. 对指定的模态进行SSA去噪
        reconstruction_components = list(sorted_modes)  # 复制一份用于重构
        for idx in self.denoise_indices:
            if idx < len(sorted_modes):
                denoised_mode = self._ssa_denoise_mode(sorted_modes[idx])
                reconstruction_components[idx] = denoised_mode  # 替换为去噪后的版本

        # 4. 重构最终信号
        final_signal = np.sum(reconstruction_components, axis=0)

        return final_signal


def stft_transform(signal, nperseg=64, noverlap=32, nfft=128):
    """对信号进行短时傅里叶变换"""
    _, _, Zxx = stft(signal, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return np.abs(Zxx)

