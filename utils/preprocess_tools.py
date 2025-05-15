# This file is the tools for processing CSI data.
import numpy as np
from numpy.linalg import svd
import pywt
from scipy.signal import stft, get_window, windows, filtfilt, lfilter, butter
from scipy.interpolate import interp1d, CubicSpline, splev, splrep
from sklearn.preprocessing import StandardScaler
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

def calculate_csi_ratio(csi_data, eps=1e-6):
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
            continue
        else:
            # 其他天线的CSI比值
            csi_ratio[:, antenna, :] = csi_data[:, antenna, :] / ((np.abs(reference_csi) + eps)*np.exp(1j*np.angle(reference_csi)))

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
    half_window = (window_size-1)//2
    for ant in range(num_antennas):
        for sc in range(num_subcarriers):
            sequence = data[:, ant, sc]
            for i in range(half_window, num_packets-half_window):
                window = sequence[max(0, i-window_size//2) : min(num_packets, i+window_size//2+1)]
                median = np.median(window)
                mad = np.median(np.abs(window - median))
                if np.abs(sequence[i] - median) > n_sigmas * mad:
                    filtered_data[i, ant,sc] = median
    
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

def DWT_Denoise(csi_data, wavelet='db3', level=5, threshold_mode='soft', mode='sym', n_jobs=None):
    """
    对CSI数据进行小波去噪（支持并行处理）
    
    参数:
        csi_data: 三维CSI数据，形状为(num_packets, num_antennas, num_subcarriers)
        n_jobs: 并行任务数（None表示串行处理）
    """
    _, num_antennas, num_subcarriers = csi_data.shape
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

# 5. PCA去噪
def butter_bandpass_filter(csi_data, highcut=200, fs=1000, order=5):
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs  # Nyquist频率
        normal_cutoff = cutoff / nyq  # 归一化截止频率
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    b, a = butter_lowpass(highcut, fs, order)
    
    _, num_antennas, num_subcarriers = csi_data.shape
    ans_csi = np.zeros_like(csi_data)
    for ant in range(num_antennas):
        for sub in range(num_subcarriers):
            data = csi_data[:, ant, sub]
            # 应用滤波器
            filtered_data = filtfilt(b, a, data)
            ans_csi[:, ant, sub] = filtered_data
    return ans_csi

def pca_denoise(csi_data, highcut=100, fs=1000, order=5, n_components=0.90):
    num_packets, num_antennas, num_subcarriers = csi_data.shape
    # Butterworth filter
    butter_csi = butter_bandpass_filter(csi_data, highcut=highcut, fs=fs, order=order)
    # 标准化
    butter_csi_re = butter_csi.reshape(num_packets, -1)
    scaler = StandardScaler()
    scale_csi = scaler.fit_transform(butter_csi_re)
    # PCA
    pca = PCA(n_components=n_components)
    pca_csi = pca.fit_transform(scale_csi)
    # PCA Inverse
    csi_pca_inverse = pca.inverse_transform(pca_csi)
    csi_scale_inverse = scaler.inverse_transform(csi_pca_inverse)
    csi_scale_inverse = csi_scale_inverse.reshape(num_packets, num_antennas, num_subcarriers)
    return csi_scale_inverse


# 6. Extract DFS From CSI
def dfs_pca(X, n_components=None, centered=True, algorithm='svd',
        weights=None, variable_weights=None, missing_rows='complete',
        economy=True, tol=1e-6):
    """
    Principal Component Analysis (PCA) for complex-valued data matrices
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The complex-valued data matrix
    n_components : int, optional
        Number of components to keep
    centered : bool, default=True
        Whether to center the data before PCA
    algorithm : {'svd'}, default='svd'
        Algorithm to use (currently only SVD supported)
    weights : array-like of shape (n_samples,), optional
        Observation weights
    variable_weights : {'variance', array-like}, optional
        Variable weights specification
    missing_rows : {'complete'}, default='complete'
        Missing value handling method
    economy : bool, default=True
        Whether to return economy-size results
    tol : float, default=1e-6
        Tolerance for determining effective rank
        
    Returns
    -------
    coeff : ndarray of shape (n_features, n_components)
        Principal component coefficients (loadings)
    score : ndarray of shape (n_samples, n_components)
        Principal component scores
    latent : ndarray of shape (n_components,)
        Variance explained by each component
    tsquared : ndarray of shape (n_samples,)
        Hotelling's T-squared statistic
    explained : ndarray of shape (n_components,)
        Percentage of variance explained
    mu : ndarray of shape (n_features,)
        Column means (if centered)
    """
    
    X = np.asarray(X)
    n, p = X.shape
    
    # ================ Missing Value Handling ================
    nan_mask = np.isnan(X).any(axis=1)
    if missing_rows == 'complete':
        valid_rows = ~nan_mask
        X_clean = X[valid_rows]
        if weights is not None:
            weights = np.asarray(weights)[valid_rows]
    else:
        raise ValueError("Only 'complete' missing value handling supported")
    
    n_clean, p_clean = X_clean.shape
    
    # ================ Weight Initialization ================
    if weights is None:
        weights = np.ones(n_clean, dtype=X.dtype)
    else:
        weights = np.asarray(weights).astype(X.dtype)
    
    # ================ Variable Weights Handling ================
    if variable_weights == 'variance':
        if not centered:
            raise ValueError("Variance weights require centering")
        # Match MATLAB's wnanvar with ddof=1
        mu_clean = np.nanmean(X_clean, axis=0)
        X_centered_clean = X_clean - mu_clean
        var = np.nansum((X_centered_clean * np.conj(X_centered_clean)) * weights.reshape(-1,1), axis=0) / (np.sum(weights) - 1)
        variable_weights = 1 / np.real(var)
    elif variable_weights is None:
        variable_weights = np.ones(p, dtype=np.float64)
    else:
        variable_weights = np.asarray(variable_weights)
    
    # ================ Data Centering ================
    if centered:
        mu = np.nansum(X_clean * weights.reshape(-1,1), axis=0) / np.sum(weights)
        X_centered = X_clean - mu
    else:
        mu = np.zeros(p, dtype=X.dtype)
        X_centered = X_clean.copy()
    
    # ================ Weight Application ================
    sqrt_weights = np.sqrt(weights).reshape(-1,1)
    sqrt_var_weights = np.sqrt(variable_weights).reshape(1,-1)
    
    X_weighted = X_centered * sqrt_weights * sqrt_var_weights
    
    # ================ SVD Decomposition ================
    U, S, Vt = svd(X_weighted, full_matrices=False)
    coeff = (Vt.T / sqrt_var_weights.reshape(-1,1)).astype(X.dtype)
    
    # ================ Scores Calculation ================
    score = U * S
    score = score / sqrt_weights
    
    # ================ Sign Convention (MATLAB Compatibility) ================
    for i in range(coeff.shape[1]):
        col = coeff[:, i]
        max_idx = np.argmax(np.abs(col))
        max_val = col[max_idx]
        phase = np.angle(max_val)
        rotation = np.exp(-1j * phase)
        coeff[:, i] = (col * rotation).astype(X.dtype)
        score[:, i] = (score[:, i] * rotation).astype(X.dtype)
    
    # ================ Variance Explained ================
    dof = n_clean - (1 if centered else 0)
    latent = (S**2) / dof if dof > 0 else np.zeros_like(S)
    
    # ================ Economy Size Handling ================
    if economy:
        max_components = min(dof, p)
        coeff = coeff[:, :max_components]
        score = score[:, :max_components]
        latent = latent[:max_components]
    
    # ================ Component Selection ================
    if n_components is not None:
        coeff = coeff[:, :n_components]
        score = score[:, :n_components]
        latent = latent[:n_components]
    
    # ================ Explained Variance ================
    total_var = np.sum(np.real(latent))
    explained = (np.real(latent) / total_var * 100) if total_var > 0 else np.zeros_like(latent)
    
    # ================ Hotelling's T-Squared ================
    effective_rank = np.sum(latent > tol * np.max(latent))
    if effective_rank > 0:
        stand_scores = score[:, :effective_rank] / np.sqrt(latent[:effective_rank])
        tsquared = np.sum(np.abs(stand_scores)**2, axis=1)
    else:
        tsquared = np.zeros(n_clean, dtype=np.float64)
    
    # ================ NaN Reinsertion ================
    full_score = np.full((n, score.shape[1]), np.nan, dtype=score.dtype)
    full_score[valid_rows] = score
    score = full_score
    
    full_tsquared = np.full(n, np.nan, dtype=np.float64)
    full_tsquared[valid_rows] = tsquared
    
    return coeff, score, latent, full_tsquared, explained, mu

def tfrsp(x, t=None, N=None, h=None, trace=0):
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    xrow, xcol = x.shape
    if xcol != 1:
        raise ValueError("X must have one column")
    
    # Handle default parameters
    if t is None:
        t = np.arange(1, xrow + 1)
    else:
        t = np.array(t).flatten()
        if t.ndim != 1:
            raise ValueError("T must be a 1-D array")
    
    if N is None:
        N = xrow
    elif N <= 0:
        raise ValueError("N must be greater than zero")
    
    # Generate default window h
    if h is None:
        hlength = int(np.floor(N / 4))
        hlength += 1 - (hlength % 2)  # Ensure odd length
        if hlength <= 0:
            hlength = 1
        h = get_window('hamming', hlength)
    else:
        h = np.array(h).flatten()
        if len(h) % 2 == 0:
            raise ValueError("H must have odd length")
    
    # Normalize window to unit energy
    h = h / np.linalg.norm(h)
    Lh = (len(h) - 1) // 2
    
    if (N & (N - 1)) != 0:
        print("Note: For faster computation, N should be a power of two")
    
    tcol = len(t)
    tfr = np.zeros((N, tcol), dtype=np.complex128)
    
    for icol in range(tcol):
        ti = t[icol]
        ti_py = ti - 1  # Convert to 0-based index
        
        # Calculate valid tau range
        max_neg = min(int(round(N/2 - 1)), Lh, ti_py)
        max_pos = min(int(round(N/2 - 1)), Lh, xrow - 1 - ti_py)
        tau = np.arange(-max_neg, max_pos + 1)
        
        if len(tau) == 0:
            continue
        
        # Signal segment
        indices_x = ti_py + tau
        if np.any(indices_x < 0) or np.any(indices_x >= xrow):
            raise IndexError("Signal indices out of bounds")
        x_segment = x[indices_x, 0]
        
        # Window segment
        h_segment = h[Lh + tau]
        # Normalize the window segment
        h_norm = np.linalg.norm(h_segment)
        if h_norm == 0:
            h_segment_normalized = h_segment
        else:
            h_segment_normalized = h_segment / h_norm
        
        # Compute product and place in tfr
        product = x_segment * h_segment_normalized.conj()
        indices_tfr = np.mod(N + tau, N)
        tfr[indices_tfr, icol] = product
        
        if trace and (icol + 1) % max(1, tcol // 10) == 0:
            print(f"Progress: {icol + 1}/{tcol}")
    
    # Compute FFT and magnitude squared
    tfr = np.abs(np.fft.fft(tfr, axis=0)) ** 2
    
    # Frequency vector
    f = np.fft.fftfreq(N).reshape(-1, 1)
    
    return tfr, t, f

def get_doppler_spectrum(csi_data):
    '''
    param: csi_data: 输入的CSI数据, shape为(time_length, rx_acnt, subcarrier_num)
    param: rx_acnt: 接收天线数量
    '''
    time, rx_acnt, subcarrier_num = csi_data.shape

    # Set-Up Parameters
    samp_rate = 1000
    half_rate = samp_rate / 2
    uppe_orde = 6
    uppe_stop = 60
    lowe_orde = 3
    lowe_stop = 2

    # Butterworth滤波器设计
    lu, ld = butter(uppe_orde, uppe_stop / half_rate, 'low')
    hu, hd = butter(lowe_orde, lowe_stop / half_rate, 'high')

    # 频率分箱
    freq_bins_unwrap = np.concatenate([np.arange(0, samp_rate/2), np.arange(-samp_rate/2, 0)])
    freq_bins_unwrap = freq_bins_unwrap / samp_rate
    freq_lpf_sele = (freq_bins_unwrap <= uppe_stop / samp_rate) & (freq_bins_unwrap >= -uppe_stop / samp_rate)
    freq_lpf_positive_max = np.sum(freq_lpf_sele[1:int(len(freq_lpf_sele)/2)])
    freq_lpf_negative_min = np.sum(freq_lpf_sele[int(len(freq_lpf_sele)/2):])

    # 多普勒频谱初始化
    # csi_data = pca_denoise(csi_data)
    doppler_spectrum = np.zeros((1 + int(freq_lpf_positive_max) + int(freq_lpf_negative_min), int(np.floor(csi_data.shape[0]))))

    # 降采样
    csi_data = csi_data[::1, :, :]

    # 选择天线对
    csi_mean = np.mean(np.abs(csi_data), axis=0)
    csi_var = np.sqrt(np.var(np.abs(csi_data), axis=0))
    csi_mean_var_ratio = csi_mean / csi_var
    idx = np.argmax(np.mean(csi_mean_var_ratio, axis=-1))
    csi_data_ref = np.tile(csi_data[:, idx, :], (1, rx_acnt)).reshape(time, rx_acnt, -1)

    # 幅度调整
    csi_data_adj = np.zeros_like(csi_data)
    csi_data_ref_adj = np.zeros_like(csi_data_ref)
    alpha_sum = 0
    for acnt in range(rx_acnt):
        for subcarrier in range(subcarrier_num):
            amp = np.abs(csi_data[:, acnt, subcarrier])
            alpha = np.min(amp[amp != 0])
            alpha_sum += alpha
            csi_data_adj[:, acnt, subcarrier] = np.abs(np.abs(csi_data[:, acnt, subcarrier]) - alpha) * np.exp(1j * np.angle(csi_data[:, acnt, subcarrier]))
    beta = 1000 * alpha_sum / (subcarrier_num * rx_acnt)
    for acnt in range(rx_acnt):
        for subcarrier in range(subcarrier_num):
            csi_data_ref_adj[:, acnt, subcarrier] = (np.abs(csi_data_ref[:, acnt, subcarrier]) + beta) * np.exp(1j * np.angle(csi_data_ref[:, acnt, subcarrier]))

    # 共轭乘法
    conj_mult = csi_data_adj * np.conj(csi_data_ref_adj)
    conj_mult = np.concatenate([conj_mult[:, :idx, :], conj_mult[:, idx+1:, :]], axis=1)

    # 滤波
    for acnt in range(rx_acnt-1):
        for subcarrier in range(subcarrier_num):
            conj_mult[:, acnt, subcarrier] = lfilter(lu, ld, conj_mult[:, acnt, subcarrier])
            conj_mult[:, acnt, subcarrier] = lfilter(hu, hd, conj_mult[:, acnt, subcarrier])

    # PCA分析
    conj_mult_temp = conj_mult.reshape(conj_mult.shape[0], -1)
    coeff, *_ = dfs_pca(conj_mult_temp)
    conj_mult_pca = np.dot(conj_mult_temp,coeff[:, 0])
    
    # 时频分析
    freq_time_prof_allfreq, t, f = tfrsp(conj_mult_pca, np.arange(conj_mult_pca.shape[0]), 1000, windows.gaussian(125,18))

    # 选择关心的频率
    freq_time_prof = freq_time_prof_allfreq[freq_lpf_sele, :]
    freq_time_prof = freq_time_prof / np.sum(freq_time_prof, axis=0)

    # 频率分箱
    freq_bin = np.concatenate([np.arange(0, freq_lpf_positive_max+1), np.arange(-freq_lpf_negative_min, 0)])

    # 存储多普勒频谱
    if freq_time_prof.shape[1] >= doppler_spectrum.shape[1]:
        doppler_spectrum[:, :] = freq_time_prof[:, :doppler_spectrum.shape[1]]
    else:
        doppler_spectrum[:, :] = np.concatenate([freq_time_prof, np.zeros((doppler_spectrum.shape[0], doppler_spectrum.shape[1] - freq_time_prof.shape[1]))], axis=1)

    return doppler_spectrum, t, freq_bin