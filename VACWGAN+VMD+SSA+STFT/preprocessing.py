# ===================================================================================
# 文件名: preprocessing.py (修正 STFT 确保输出 64x32)
# ===================================================================================
import numpy as np
from scipy.signal import stft
from vmdpy import VMD


class VMD_Sorter_SSA_Processor:
    """VMD + SSA 处理器 (逻辑不变)"""

    def __init__(self, K, alpha, denoise_indices: list, window_len, ssa_threshold):
        self.K = K
        self.alpha = alpha
        self.denoise_indices = denoise_indices
        self.window_len = window_len
        self.ssa_threshold = ssa_threshold
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

        reconstructed_matrix = u[:, :threshold] @ np.diag(sigma[:threshold]) @ vt[:threshold, :]

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
        """执行 VMD -> Sort -> SSA -> Reconstruct 流程"""
        u, _, omega = VMD(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
        sort_order = np.argsort(omega[0, :])[::-1]
        sorted_modes = u[sort_order, :]
        reconstruction_components = list(sorted_modes)
        for idx in self.denoise_indices:
            if idx < len(sorted_modes):
                denoised_mode = self._ssa_denoise_mode(sorted_modes[idx])
                reconstruction_components[idx] = denoised_mode

        final_signal = np.sum(reconstruction_components, axis=0)
        return final_signal


def stft_transform(signal, nperseg=64, noverlap=32, nfft=127, target_T=32):
    """
    对信号进行短时傅里叶变换，确保输出尺寸固定为 [64, 32]。
    修复：nfft=127 确保频率维是 64 (127//2 + 1 = 64)。
    """
    # 假设标准信号长度为 1024 进行截断或填充
    signal_len = 1024
    if len(signal) > signal_len:
        signal = signal[:signal_len]
    elif len(signal) < signal_len:
        signal = np.pad(signal, (0, signal_len - len(signal)), 'constant')

    f, t, Zxx = stft(signal, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    stft_map = np.abs(Zxx) # 形状: [64, T]

    # 确保时间维是 target_T=32，截断或填充
    if stft_map.shape[1] > target_T:
        stft_map = stft_map[:, :target_T]
    elif stft_map.shape[1] < target_T:
        padding = target_T - stft_map.shape[1]
        stft_map = np.pad(stft_map, ((0, 0), (0, padding)), 'constant')

    return stft_map