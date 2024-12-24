import torch  

def time2timefreq(signal,n_fft = 256,hop_length = 16,sampling_rate =125,valid_freq_range = (0, 30)):
    # # 使用STFT进行时频分析 
    # n_fft = 256  # FFT的窗口大小
    # hop_length = 16  # 滑动窗口步长
    # sampling_rate =125
    stft_result = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    

    # 1. 计算信号强度 (幅值)
    signal_strength = torch.abs(stft_result)
    phase = torch.angle(stft_result)    # 相位

    # 2. 定义有效频率范围
    # valid_freq_range = (0, 30)  # 以Hz为单位的有效频率范围
    frequencies = torch.fft.fftfreq(n_fft, 1 / sampling_rate)[:n_fft // 2 + 1]
    times = torch.arange(stft_result.size(-1)) * hop_length / sampling_rate

    # 3. 获取有效频率的索引
    valid_indices = (frequencies >= valid_freq_range[0]) & (frequencies <= valid_freq_range[1])
    valid_indices = valid_indices.nonzero(as_tuple=True)[0]  # 获取有效频率的索引

    # 4. 过滤频谱信号
    filtered_signal_strength = signal_strength[:, valid_indices, :]  # 只保留有效频率的部分
    filtered_phase = phase[:, valid_indices, :]  # 只保留有效频率的部分
    filtered_frequencies = frequencies[  valid_indices ]  
    print('频率过滤后的信号',filtered_frequencies.shape) 
    return filtered_signal_strength,filtered_frequencies,times,filtered_phase


def add_gaussian_noise_all(signal,   mean=0, std_dev=0.15,phase= 1 ):
    noise = phase * torch.normal(mean=torch.full(signal.shape[-1], mean,dtype=torch.float32), 
                                 std=torch.full(signal.shape[-1], std_dev,dtype=torch.float32))  # 生成高斯噪声
    signal +=  noise
    return signal

def add_pink_noise_all(signal, alpha=1.0, std_dev=0.15, phase=0.15):
    """
    添加粉红噪声到信号。

    参数:
        signal (torch.Tensor): 输入信号 (假设形状为 [..., 时间步])。
        alpha (float): 控制频谱中噪声的衰减（典型值为1.0表示粉红噪声）。
        std_dev (float): 控制噪声的标准差。
        phase (float): 噪声相位控制因子，用于调整噪声幅值方向。

    返回:
        torch.Tensor: 添加粉红噪声后的信号。
    """
    # 获取信号的最后一维大小（时间步数）
    num_steps = signal.shape[-1]
    
    # 生成随机高斯噪声
    gaussian_noise = torch.normal(mean=0, std=std_dev, size=(num_steps,), dtype=torch.float32)
    
    # 计算频率分量的缩放因子
    freqs = torch.fft.rfftfreq(num_steps)
    scaling = torch.pow(freqs + 1e-10, -alpha / 2)  # 避免除以零
    scaling[0] = 0  # 去除直流分量
    
    # 对高斯噪声进行傅里叶变换
    fft_noise = torch.fft.rfft(gaussian_noise)
    
    # 施加粉红噪声频谱缩放
    fft_noise *= scaling
    
    # 进行逆傅里叶变换
    pink_noise = torch.fft.irfft(fft_noise, n=num_steps)
    
    # 调整相位并添加到信号
    pink_noise = phase * pink_noise
    signal += pink_noise
    
    return signal

# 生成高斯噪声
# def add_gaussian_noise(signal, start_idx, end_idx, mean , std_dev , phase ):
#     noise = phase * torch.normal(mean=torch.full((end_idx - start_idx,), mean,dtype=torch.float32), 
#                                  std=torch.full((end_idx - start_idx,), std_dev,dtype=torch.float32))  # 生成高斯噪声
#     signal[start_idx:end_idx] = signal[start_idx:end_idx]+noise
#     return signal

# # 生成先上升后下降的漂移（线性漂移，前一半上升，后一半下降）
# def add_rise_and_fall_drift(signal, start_idx, end_idx, drift_rate):
#     half_len = (end_idx - start_idx) // 2  # 中点
#     rise = torch.linspace(0, drift_rate * half_len, half_len)  # 上升部分
#     fall = torch.linspace(drift_rate * half_len, 0, end_idx - start_idx - half_len)  # 下降部分
#     drift = torch.cat([rise, fall])  # 将两部分合并
#     signal[start_idx:end_idx] = signal[start_idx:end_idx]+drift
#     return signal

# # 2. Poisson Noise (泊松噪声)
# def add_poisson_noise(signal, start_idx, end_idx, lam=0.5):
#     noise = torch.poisson(torch.full((end_idx - start_idx,), lam))  # Generate Poisson noise with lambda
#     signal[start_idx:end_idx] =signal[start_idx:end_idx]+ noise
#     return signal

# # 3. Salt and Pepper Noise (盐和胡椒噪声)
# def add_salt_and_pepper_noise(signal, start_idx, end_idx, salt_prob=0.05):
#     noise = torch.randint(0, 2, (end_idx - start_idx,))  # Generate random 0s and 1s
#     salt_mask = noise == 1  # Salt noise (set to 1)
#     pepper_mask = noise == 0  # Pepper noise (set to 0)

#     # Apply the noise to the signal
#     signal[start_idx:end_idx][salt_mask] = 1  # Salt noise
#     signal[start_idx:end_idx][pepper_mask] = 0  # Pepper noise

#     return signal

# # 4. Zebra Noise (斑马纹噪声)
# def add_zebra_noise(signal, start_idx, end_idx, phase=1,cycle=5):
#     stripes = torch.sin(2 * torch.pi * cycle * torch.linspace(0, 1, end_idx - start_idx))  # Generate periodic pattern
#     signal[start_idx:end_idx] = signal[start_idx:end_idx]+phase * stripes  # Add the zebra noise with amplitude phase
#     return signal
