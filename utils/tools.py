# This file is the tools for processing CSI data.
import numpy as np
import pywt
from scipy.signal import stft
from scipy.interpolate import interp1d, CubicSpline, splev, splrep

# 1. Create CSI-Ratio from CSI
def calculate_amplitude_variance_ratio(csi_data):
    """
    计算每个天线的所有子载波的幅度方差比之和
    :param csi_data: CSI数据，形状为 (num_packets, num_antennas, num_subcarriers)==(数据包数量,天线数量,子载波数量)
    :return: 每个天线的幅度方差比之和，形状为 (num_antennas,)
    """
    # 计算幅度
    amplitude = np.abs(csi_data)  # 形状为 (num_packets, num_antennas, num_subcarriers)

    # 计算每个天线和子载波的幅度方差
    variance = np.var(amplitude, axis=0)  # 形状为 (num_antennas, num_subcarriers)

    # 计算每个天线的幅度方差比之和
    variance_ratio_sum = np.sum(variance, axis=1)  # 形状为 (num_antennas,)

    return variance_ratio_sum

def select_reference_antenna(csi_data):
    """
    选择参考天线（幅度方差比之和最小的天线）
    :param csi_data: CSI数据，形状为 (num_packets, num_antennas, num_subcarriers)
    :return: 参考天线的索引
    """
    variance_ratio_sum = calculate_amplitude_variance_ratio(csi_data)
    reference_antenna_index = np.argmin(variance_ratio_sum)
    return reference_antenna_index

def calculate_csi_ratio(csi_data):
    """
    计算CSI比值（CSI-Ratio）
    :param csi_data: CSI数据，形状为 (num_packets, num_antennas, num_subcarriers)
    :return: CSI比值数据，形状为 (num_packets, num_antennas, num_subcarriers)
    """
    num_packets, num_antennas, num_subcarriers = csi_data.shape

    # 选择参考天线
    reference_antenna_index = select_reference_antenna(csi_data)
    reference_csi = csi_data[:, reference_antenna_index, :]  # 参考天线的CSI数据

    # 计算CSI比值
    csi_ratio = np.zeros_like(csi_data)
    for antenna in range(num_antennas):
        if antenna == reference_antenna_index:
            # 参考天线的CSI比值为1
            csi_ratio[:, antenna, :] = 1.0
        else:
            # 其他天线的CSI比值
            csi_ratio[:, antenna, :] = csi_data[:, antenna, :] / reference_csi

    return csi_ratio, reference_antenna_index

# 2. Resample CSI Sequence
def resample_csi_sequence(csi_sequence, target_length=500):
    """
    对单个CSI序列进行重采样至目标长度
    :param csi_sequence: 原始CSI序列，形状为 (original_length, num_antennas, num_subcarriers)
    :param target_length: 目标序列长度（默认为500）
    :return: 重采样后的CSI序列，形状为 (target_length, num_antennas, num_subcarriers)
    """
    original_length = csi_sequence.shape[0]
    num_antennas = csi_sequence.shape[1]
    num_subcarriers = csi_sequence.shape[2]
    
    # 创建新时间轴
    original_time = np.linspace(0, 1, original_length)  # 归一化时间轴
    new_time = np.linspace(0, 1, target_length)          # 目标时间轴
    
    # 初始化输出序列
    resampled_sequence = np.zeros((target_length, num_antennas, num_subcarriers))
    
    # 对每个天线和子载波进行插值
    for ant in range(num_antennas):
        for sc in range(num_subcarriers):
            # 提取原始复数值序列
            complex_sequence = csi_sequence[:, ant, sc]
            
            # 创建插值函数
            interp_real = interp1d(original_time, complex_sequence, kind='linear')

            # 生成新序列
            resampled_real = interp_real(new_time)
            
            resampled_sequence[:, ant, sc] = resampled_real
            
    return resampled_sequence

# 3. 小波变换滤波
def wavelet_denoise(signal, wavelet='db4', level=None, threshold_mode='soft', mode='sym'):
    """
    对一维信号进行小波去噪，支持实数和复数信号。
    """
    if np.iscomplexobj(signal):
        # 复数信号分解为实部和虚部分别处理
        real_part = np.real(signal)
        imag_part = np.imag(signal)
        denoised_real = wavelet_denoise(real_part, wavelet, level, threshold_mode, mode)
        denoised_imag = wavelet_denoise(imag_part, wavelet, level, threshold_mode, mode)
        return denoised_real + 1j * denoised_imag
    else:
        # 实数信号处理
        # 自动计算最大分解层数（如果未指定）
        if level is None:
            level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
            # 如果信号长度不足以进行小波分解，则返回原信号
            if level == 0:
                return signal
        
        try:
            coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
        except ValueError:
            # 分解层数过高，降低层数
            level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
            if level == 0:
                return signal
            coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
        
        if len(coeffs) < 2:
            return signal
        
        # 使用最高层细节系数估计噪声标准差
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # 阈值处理细节系数
        coeffs_thresholded = [coeffs[0]]  # 保留近似系数
        for i in range(1, len(coeffs)):
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode=threshold_mode))
        
        # 重构信号
        denoised_signal = pywt.waverec(coeffs_thresholded, wavelet, mode=mode)
        
        # 确保长度一致
        if len(denoised_signal) != len(signal):
            denoised_signal = np.resize(denoised_signal, len(signal))
        
        return denoised_signal

def denoise_csi_data(csi_data, wavelet='db4', level=None, threshold_mode='soft', mode='sym', n_jobs=None):
    """
    对CSI数据进行小波去噪（支持并行处理）
    
    参数:
        csi_data: 三维CSI数据，形状为(num_packets, num_antennas, num_subcarriers)
        n_jobs: 并行任务数（None表示串行处理）
    """
    num_packets, num_antennas, num_subcarriers = csi_data.shape
    csi_denoised = np.zeros_like(csi_data)
    
    # 并行处理
    if n_jobs is not None and n_jobs != 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(wavelet_denoise)(csi_data[:, i, j], wavelet, level, threshold_mode, mode)
            for i in range(num_antennas)
            for j in range(num_subcarriers)
        )
        
        idx = 0
        for i in range(num_antennas):
            for j in range(num_subcarriers):
                csi_denoised[:, i, j] = results[idx]
                idx += 1
    else:
        # 串行处理
        for i in range(num_antennas):
            for j in range(num_subcarriers):
                csi_denoised[:, i, j] = wavelet_denoise(
                    csi_data[:, i, j], 
                    wavelet=wavelet,
                    level=level,
                    threshold_mode=threshold_mode,
                    mode=mode
                )
    
    return csi_denoised

# 4. Extract DFS From CSI
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