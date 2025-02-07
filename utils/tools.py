# This file is the tools for processing CSI data.
from scipy.signal import stft
import numpy as np
import pywt

# 小波变换滤波
def wavelet_filter(csi_data, wavelet='db4', level=6):
    """
    使用小波变换对复数 CSI 数据去噪
    :param csi_data: CSI 数据 (Time x N_subcarrier)，复数矩阵
    :param wavelet: 小波类型
    :param level: 分解层数
    :return: 去噪后的复数 CSI 数据
    """
    denoised_data = np.zeros_like(csi_data, dtype=complex)

    # 分别对实部和虚部进行小波去噪
    real_part = np.real(csi_data)
    imag_part = np.imag(csi_data)

    def denoise_signal(signal):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # 对高频分量（噪声）进行阈值处理
        coeffs[1:] = [pywt.threshold(c, value=0.2 * np.max(c), mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)

    for i in range(csi_data.shape[1]):  # 遍历每个子载波
        denoised_real = denoise_signal(real_part[:, i])
        denoised_imag = denoise_signal(imag_part[:, i])
        # 合并实部和虚部
        denoised_data[:, i] = denoised_real[:csi_data.shape[0]] + 1j * denoised_imag[:csi_data.shape[0]]

    return denoised_data

# Extract DFS From CSI
def compute_dfs(csi_data, fs=121, nperseg=256):
    """
    计算 Doppler Frequency Spectrum (DFS)
    :param csi_data: CSI 数据 (Time x N_subcarrier)
    :param fs: 采样频率 (Hz)
    :param nperseg: 每段的窗口长度
    :return: 时间, 频率, 频谱强度
    """
    time, freq, spectrum = [], [], []
    num_subcarriers = csi_data.shape[1]

    # 对每个子载波计算 STFT
    for subcarrier in range(num_subcarriers):
        f, t, Zxx = stft(csi_data[:, subcarrier], fs=fs, nperseg=nperseg)
        if subcarrier == 0:
            time, freq = t, f
            spectrum = np.abs(Zxx)
        else:
            spectrum += np.abs(Zxx)  # 聚合所有子载波的频谱

    return time, freq, spectrum