import numpy as np
from PyEMD import EMD
from scipy.signal import stft


class EMD_SSA_Denoising:
    """
    一个封装了EMD分解和SSA去噪的类。
    - window_len: SSA的嵌入窗口长度。
    - threshold_imf1: 第一个IMF的奇异值阈值。
    - threshold_imf2: 第二个IMF的奇异值阈值。
    """

    def __init__(self, window_len, threshold_imf1, threshold_imf2):
        if window_len <= 1 or not isinstance(window_len, int):
            raise ValueError("window_len 必须是大于1的整数")
        self.window_len = window_len
        self.threshold_indices = [threshold_imf1, threshold_imf2]
        self.emd = EMD()

    def _ssa_denoise_imf(self, imf, threshold_index):
        """对单个IMF分量执行SSA去噪，并返回主趋势和噪声"""
        if threshold_index >= self.window_len:
            threshold_index = self.window_len - 1

        # 1. 嵌入
        series_len = len(imf)
        k = series_len - self.window_len + 1
        trajectory_matrix = np.zeros((self.window_len, k))
        for i in range(k):
            trajectory_matrix[:, i] = imf[i:i + self.window_len]

        # 2. SVD分解
        u, sigma, vt = np.linalg.svd(trajectory_matrix, full_matrices=False)

        # 3. 分组（重构主趋势）
        reconstructed_matrix = np.dot(u[:, :threshold_index], np.dot(np.diag(sigma[:threshold_index]), vt[:threshold_index, :]))

        # 4. 对角平均
        denoised_series = np.zeros(series_len)
        buffer = np.zeros(series_len)
        counts = np.zeros(series_len)
        for i in range(self.window_len):
            for j in range(k):
                idx = i + j
                buffer[idx] += reconstructed_matrix[i, j]
                counts[idx] += 1
        denoised_series = buffer / counts

        # 5. 计算噪声层 (原始信号 - 主趋势)
        noise_series = imf - denoised_series

        # ---【关键修改点】---
        # 确保函数返回两个值：去噪后的序列和噪声序列
        return denoised_series, noise_series

    def process(self, signal):
        """对输入信号执行完整的EMD-SSA去噪流程"""
        imfs_and_residue = self.emd.emd(signal, max_imf=5)
        imfs = imfs_and_residue[:-1]  # 分离IMF和残差
        residue = imfs_and_residue[-1]

        if len(imfs) < 2:
            return signal

        denoised_imf1, _ = self._ssa_denoise_imf(imfs[0], self.threshold_indices[0])
        denoised_imf2, _ = self._ssa_denoise_imf(imfs[1], self.threshold_indices[1])

        # 重构信号
        reconstructed_signal = denoised_imf1 + denoised_imf2
        for i in range(2, len(imfs)):
            reconstructed_signal += imfs[i]
        reconstructed_signal += residue

        return reconstructed_signal


def stft_transform(signal, nperseg=64, noverlap=32, nfft=128):
    """
    对信号进行短时傅里叶变换
    - signal: 输入的一维信号
    - nperseg: 每个段的长度
    - noverlap: 段之间的重叠长度
    - nfft: FFT的长度
    """
    _, _, Zxx = stft(signal, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return np.abs(Zxx)
