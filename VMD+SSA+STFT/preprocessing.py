import numpy as np
from scipy.signal import stft
from vmdpy import VMD  # 导入VMD


class VMD_SSA_Denoising:
    """
    一个封装了VMD分解和SSA去噪的类。
    - K: VMD分解的模态数量
    - alpha: VMD中的带宽约束参数
    - window_len: SSA的嵌入窗口长度
    - threshold_mode1: 第一个模态的奇异值阈值
    - threshold_mode2: 第二个模态的奇异值阈值
    """

    def __init__(self, K, alpha, window_len, threshold_mode1, threshold_mode2):
        if window_len <= 1 or not isinstance(window_len, int):
            raise ValueError("window_len 必须是大于1的整数")

        # VMD 参数
        self.K = K
        self.alpha = alpha
        self.tau = 0.  # 噪声容忍度
        self.DC = 0  # 无直流分量
        self.init = 1  # 初始化 omega
        self.tol = 1e-7  # 收敛容忍度

        # SSA 参数
        self.window_len = window_len
        self.threshold_indices = [threshold_mode1, threshold_mode2]

    def _ssa_denoise_mode(self, mode, threshold_index):
        """对单个模态分量(mode)执行SSA去噪，并返回主趋势"""
        if threshold_index >= self.window_len:
            threshold_index = self.window_len - 1

        # 1. 嵌入
        series_len = len(mode)
        k = series_len - self.window_len + 1
        trajectory_matrix = np.zeros((self.window_len, k))
        for i in range(k):
            trajectory_matrix[:, i] = mode[i:i + self.window_len]

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

        # 避免除以零
        denoised_series = np.divide(buffer, counts, out=np.zeros_like(buffer), where=counts != 0)

        return denoised_series

    def process(self, signal):
        """对输入信号执行完整的VMD-SSA去噪流程"""
        # 1. VMD分解
        # u.shape: (K, len(signal))
        u, u_hat, omega = VMD(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)

        if u.shape[0] < 2:
            # 如果分解出的模态少于2个，则直接返回原始信号或第一个模态
            return signal

        # 2. 对前两个模态应用SSA去噪
        denoised_mode1 = self._ssa_denoise_mode(u[0], self.threshold_indices[0])
        denoised_mode2 = self._ssa_denoise_mode(u[1], self.threshold_indices[1])

        # 3. 重构信号
        # 将去噪后的前两个模态与剩余的原始模态相加
        remaining_modes = np.sum(u[2:], axis=0)
        reconstructed_signal = denoised_mode1 + denoised_mode2 + remaining_modes

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
