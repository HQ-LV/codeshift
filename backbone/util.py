import torch
import torch.fft as fft

def batch_time_shift_signal(signal,  time_shift,sampling_frequency=125,padding_length=64, padding_type='value'):
    """
    Perform time shift on batch signals using frequency domain phase shift with optional padding.
    
    Args:
        signal (torch.Tensor): Input signal of shape (b, seq_len).
        sampling_frequency (float): Sampling frequency of the signal (Hz).
        time_shift (torch.Tensor): Batch time shifts of shape (b, 1).
        padding_length (int, optional): Number of padding points on each side. Default is 512.
        padding_type (str, optional): Padding type - 'zero', 'value', 'custom'. Default is 'value'.
        
    Returns:
        torch.Tensor: Time-shifted signal of shape (b, seq_len).
    """


    signal = signal.squeeze(1)
    time_shift = time_shift.unsqueeze(dim=1)
    
    b, seq_len = signal.shape
    
    # Padding
    signal_max = signal.max(dim=1, keepdim=True)[0]
    signal_min = signal.min(dim=1, keepdim=True)[0]
    
    if padding_type == 'zero':
        left_pad = torch.zeros(b, padding_length, device=signal.device)
        right_pad = torch.zeros(b, padding_length, device=signal.device)
    elif padding_type == 'value':
        left_pad = signal[:, 0:1].expand(-1, padding_length)
        right_pad = signal[:, -1:].expand(-1, padding_length)
    elif padding_type == 'custom':
        left_slope = signal[:, 1] - signal[:, 0]
        right_slope = signal[:, -1] - signal[:, -2]
        
        left_pad = torch.clip(
            signal[:, 0:1] - torch.arange(1, padding_length + 1, device=signal.device) * left_slope[:, None],
            min=signal_min, max=signal_max
        )
        right_pad = torch.clip(
            signal[:, -1:] + torch.arange(1, padding_length + 1, device=signal.device) * right_slope[:, None],
            min=signal_min, max=signal_max
        )
    else:
        left_pad = right_pad = torch.tensor([], device=signal.device)
        padding_length = 0

    # Pad the signal
    padded_signal = torch.cat([left_pad, signal, right_pad], dim=1)
    
    # FFT and frequency domain
    fft_signal = fft.fft(padded_signal)
    n = padded_signal.shape[1]
    frequencies = fft.fftfreq(n, d=1 / sampling_frequency).to(signal.device)  # (n,)
    
    # Apply batch phase shift
    phase_shift = torch.exp(-1j * 2 * torch.pi * frequencies * time_shift)  # (b, n)
    shifted_fft_signal = fft_signal * phase_shift
    
    # Inverse FFT to return to time domain
    shifted_signal = fft.ifft(shifted_fft_signal).real
    
    # Extract valid portion
    valid_start = padding_length
    valid_end = valid_start + seq_len
    shifted_signal_valid = shifted_signal[:, valid_start:valid_end]
    
    return shifted_signal_valid.unsqueeze(1)