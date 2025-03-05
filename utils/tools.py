# This file is the tools for processing CSI data.
import numpy as np
import pywt
from scipy.signal import stft
from scipy.interpolate import interp1d, CubicSpline, splev, splrep
from sklearn.decomposition import PCA

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

# 2. Denoise the Phase of CSI-Ratio
def hampel_filter(data, window_size=5, n_sigmas=3):
    """
    使用Hampel滤波器去除时间序列中的异常值。
    
    :param data: 输入的时间序列数据，形状为 (num_packets, num_antennas, num_subcarriers)
    :param window_size: 计算中位数和MAD的窗口大小
    :param n_sigmas: 阈值的倍数
    :return: 去除异常值后的时间序列数据
    """
    filtered_data = np.copy(data)
    num_packets, num_antennas, num_subcarriers = data.shape
    
    for ant in range(num_antennas):
        for sc in range(num_subcarriers):
            sequence = data[:, ant, sc]
            median = np.median(sequence)
            deviation = np.abs(sequence - median)
            mad = np.median(deviation)
            # 计算阈值
            threshold = n_sigmas * mad
            # 检测异常值
            outliers = deviation > threshold
            # 替换异常值为中位数
            filtered_data[outliers, ant, sc] = median
    return filtered_data

def phase_calibration(phase_data, epsilon=0.3):
    """
    对CSI-Ratio的相位数据进行相位校准。
    参数:
    phase_data: numpy数组，形状为(num_packets, num_antennas, num_subcarriers)，包含相位数据。
    epsilon: 浮点数，经验性设置的阈值，默认为0.3。
    返回:
    校准后的相位数据，形状与phase_data相同。
    """
    calibrated_phase_data = phase_data.copy()
    num_packets, num_antennas, num_subcarriers = phase_data.shape
    
    for ant in range(num_antennas):
        for sc in range(num_subcarriers):
            for t in range(1, num_packets):
                phase_diff = calibrated_phase_data[t, ant, sc] - calibrated_phase_data[t-1, ant, sc]
                while abs(phase_diff) > 2 * np.pi - epsilon:
                    if phase_diff > 0:
                        calibrated_phase_data[t, ant, sc] -= 2 * np.pi
                    else:
                        calibrated_phase_data[t, ant, sc] += 2 * np.pi
                    phase_diff = calibrated_phase_data[t, ant, sc] - calibrated_phase_data[t-1, ant, sc]
    
    return calibrated_phase_data


# 3. Resample CSI Sequence
def resample_csi_sequence(csi_sequence, target_length=500, sample_way='linear'):
    """
    对单个CSI序列进行重采样至目标长度
    :param csi_sequence: 原始CSI序列，形状为 (original_length, num_antennas, num_subcarriers)
    :param target_length: 目标序列长度（默认为500）
    :param sample_way: 插值方式，可选 'linear', 'cubic', 'spline'（默认为 'linear'）
    :return: 重采样后的CSI序列，形状为 (target_length, num_antennas, num_subcarriers)
    """
    original_length = csi_sequence.shape[0]
    num_antennas = csi_sequence.shape[1]
    num_subcarriers = csi_sequence.shape[2]
    
    # 创建新时间轴
    original_time = np.linspace(0, 1, original_length)  # 归一化时间轴
    new_time = np.linspace(0, 1, target_length)          # 目标时间轴
    
    # 初始化输出序列
    resampled_sequence = np.zeros((target_length, num_antennas, num_subcarriers), dtype=csi_sequence.dtype)
    
    # 对每个天线和子载波进行插值
    for ant in range(num_antennas):
        for sc in range(num_subcarriers):
            # 提取原始序列
            cur_sequence = csi_sequence[:, ant, sc]
            
            # 检查数据类型
            if np.iscomplexobj(cur_sequence):
                # 如果是复数，分别处理实部和虚部
                real_part = np.real(cur_sequence)
                imag_part = np.imag(cur_sequence)
                
                # 根据插值方式选择插值函数
                if sample_way == 'linear':
                    interp_real = interp1d(original_time, real_part, kind='linear')
                    interp_imag = interp1d(original_time, imag_part, kind='linear')
                elif sample_way == 'cubic':
                    interp_real = CubicSpline(original_time, real_part)
                    interp_imag = CubicSpline(original_time, imag_part)
                elif sample_way == 'spline':
                    tck_real = splrep(original_time, real_part, k=3)
                    tck_imag = splrep(original_time, imag_part, k=3)
                    interp_real = lambda x: splev(x, tck_real)
                    interp_imag = lambda x: splev(x, tck_imag)
                else:
                    raise ValueError("Unsupported sample_way. Choose 'linear', 'cubic', or 'spline'.")
                
                # 生成新序列
                resampled_real = interp_real(new_time)
                resampled_imag = interp_imag(new_time)
                
                # 合并实部和虚部
                resampled_sequence[:, ant, sc] = resampled_real + 1j * resampled_imag
            else:
                # 如果是实数，直接插值
                if sample_way == 'linear':
                    interp_fun = interp1d(original_time, cur_sequence, kind='linear')
                elif sample_way == 'cubic':
                    interp_fun = CubicSpline(original_time, cur_sequence)
                elif sample_way == 'spline':
                    tck = splrep(original_time, cur_sequence, k=3)
                    interp_fun = lambda x: splev(x, tck)
                else:
                    raise ValueError("Unsupported sample_way. Choose 'linear', 'cubic', or 'spline'.")
                
                # 生成新序列
                resampled_sequence[:, ant, sc] = interp_fun(new_time)
            
    return resampled_sequence

# 4. 小波变换滤波
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

# 5. 短时傅里叶变换
def apply_stft(pca_data, fs=1.0, window='hann', nperseg=256, noverlap=None):
    """
    对降维后的数据进行短时傅里叶变换。
    :param pca_data: 降维后的数据，形状为 (num_packets, num_antennas, n_components)
    :param fs: 采样频率
    :param window: 窗口函数
    :param nperseg: 每个分段的长度
    :param noverlap: 分段之间的重叠长度
    :return: STFT结果，形状为 (num_antennas, n_components, n_freqs, n_segments)
    """
    num_packets, num_antennas, n_components = pca_data.shape
    stft_results = []

    for ant in range(num_antennas):
        for comp in range(n_components):
            f, t, Zxx = stft(pca_data[:, ant, comp], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            stft_results.append(Zxx)
    
    stft_results = np.array(stft_results).reshape(num_antennas, n_components, *Zxx.shape)
    return f, t, stft_results
# 6. PCA降维
def apply_pca(amplitude_data, n_components=10):
    """
    对幅度数据进行PCA降维。
    
    :param amplitude_data: 幅度数据，形状为 (num_packets, num_antennas, num_subcarriers)
    :param n_components: 降维后的主成分数量
    :return: 降维后的数据，形状为 (num_packets, num_antennas, n_components)
    """
    num_packets, num_antennas, num_subcarriers = amplitude_data.shape
    pca = PCA(n_components=n_components)
    
    # 对每个天线和子载波进行PCA降维
    pca_data = np.zeros((num_packets, num_antennas, n_components))
    for ant in range(num_antennas):
        pca_data[:, ant, :] = pca.fit_transform(amplitude_data[:, ant, :])
    
    return pca_data

# 6. Extract DFS From CSI
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