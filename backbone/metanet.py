# Define the SeparateConvMetaNet without padding
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaNet(nn.Module):
    def __init__(self, seq_len =768,max_shift=50, feature_dim=16, kernel_size=3,batch_size=64):
        super(MetaNet, self).__init__()
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.max_shift=max_shift

        # Feature extractors for output and label
        self.feature_extractor_out = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size),  # No padding
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size),  # No padding
            nn.ReLU()
        )
        self.feature_extractor_label = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size),  # No padding
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size),  # No padding
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(feature_dim) 
        # Linear transformations for h_out and h_label
        self.linear_out_L = nn.Linear(seq_len-4, 1)
        self.linear_label_L = nn.Linear(seq_len-4, 1)
        self.linear_label = nn.Linear(feature_dim, feature_dim)
        self.linear_out = nn.Linear(feature_dim, feature_dim)
     

        # Regressor to predict t
        self.regressor = nn.Sequential(
            nn.Linear(3*feature_dim , 32),  # Input dimension = feature_dim + cosine similarity
            # nn.Linear(1, 32), 
            nn.ReLU(),
            nn.Linear(32, 1)  # Output shift t
        )
        

    def compute_similarity(self, h_out, h_label): 
        # h_out_norm = F.normalize(h_out, p=2, dim=1)  # Normalize along feature dimension
        # h_label_norm = F.normalize(h_label, p=2, dim=1)  # Normalize along feature dimension
        similarity = torch.sum(h_out * h_label, dim=-1)  # Compute similarity along feature dim
 
        return similarity  

    def forward(self, y_out, y_label):
        b,_,_ =y_out.shape
        # Extract features
        h_out = self.feature_extractor_out(y_out)  # (batch_size, feature_dim, seq_len_out)
        h_label = self.feature_extractor_label(y_label)  # (batch_size, feature_dim, seq_len_label)

        # Compute cosine similarity along the sequence dimension
        similarity = self.compute_similarity(h_out, h_label)  # (batch_size, 1, seq_len)
        similarity = self.layer_norm(similarity)
    
        h_out = self.linear_out_L(h_out).squeeze(-1)
        h_label = self.linear_out_L(h_label).squeeze(-1)
        h_out = self.linear_out(h_out )  # (batch_size, feature_dim)
        h_label = self.linear_label(h_label )  # (batch_size, feature_dim)


        combined_features = torch.cat([similarity , h_out, h_label], dim=1)
        t = self.regressor(combined_features)  # (batch_size, 1) 
        # t = torch.tanh(t)
        t  = t * self.max_shift
        t = t.clamp(min=-self.max_shift, max=self.max_shift).round().long()
        return t.squeeze(-1)  # Output as a 1D tensor
    


class MetaNet_TimeShift2PhaseShift(nn.Module):
    def __init__(self, seq_len=768, max_shift=50, feature_dim=16, kernel_size=15,  smoothing_and_amplify=False):
        super(MetaNet_TimeShift2PhaseShift, self).__init__()
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.max_shift = max_shift
        self.smoothing_and_amplify=smoothing_and_amplify

        # Feature extractors for output and label
        self.feature_extractor_out = nn.Sequential(
            nn.ReplicationPad1d(padding=(kernel_size // 2)),
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size, padding=0),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=(kernel_size // 2)),
 
 
        )
        self.feature_extractor_label = nn.Sequential(
            nn.ReplicationPad1d(padding=(kernel_size // 2)),
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size, padding=0),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=(kernel_size // 2)),
  
 
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(feature_dim * 2, feature_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            nn.ConvTranspose1d(feature_dim, feature_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
        )

        self.layer_norm_out = nn.LayerNorm(normalized_shape=feature_dim)
        self.layer_norm_similarity = nn.LayerNorm(normalized_shape=feature_dim)
        self.linear_out = nn.Linear(seq_len, 1) 
        # Regressor to predict t
    
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim) ,  # Input dimension = feature_dim + cosine similarity
            nn.ReLU(),
            nn.Linear(feature_dim, 1),  # Output shift t
        )

    def compute_similarity(self, tensor1, tensor2,method='cosine'): 

        if method == 'cosine':
            # Normalize along the last dimension for cosine similarity
            tensor1_norm = F.normalize(tensor1, p=2, dim=-1)  # Shape (16, 4, 768)
            tensor2_norm = F.normalize(tensor2, p=2, dim=-1)  # Shape (16, 4, 768)
            # Compute cosine similarity along the last dimension
            similarity = torch.sum(tensor1_norm * tensor2_norm, dim=-1)  # Shape (16, 4)
        elif method == 'euclidean':
            # Compute Euclidean distance along the last dimension
            similarity = -torch.norm(tensor1 - tensor2, dim=-1, p=2)  # Shape (16, 4)
        else:
            raise ValueError("Unsupported similarity method. Use 'cosine' or 'euclidean'.")

        return similarity

    def forward(self, x_out, x_label):
        # if self.smoothing_and_amplify:  
        #     _,x_out=adaptive_smoothing_and_amplify_batch(x_out ) 
        # Extract features
        features_out = self.feature_extractor_out(x_out)#(16,1,768)
        features_label = self.feature_extractor_label(x_label)#(16,1,768)
        
        # Concatenate features # Decode features
        combined_features = torch.cat((features_out, features_label), dim=1)#(16,2,768)
        
        output = self.deconv(combined_features)#(16,1,768)
        output = self.linear_out(output)
        output = self.layer_norm_out(output.squeeze(-1))

        similarity = self.compute_similarity(features_out, features_label)# (16,1,768)
        similarity = self.layer_norm_similarity(similarity)

        combined = torch.cat([similarity , output], dim=1)#(16,2,768)

        
        t = self.regressor(combined)#(16,1) 
        t = t.clamp(min=-self.max_shift, max=self.max_shift) 
        return t.squeeze(-1)  # Output as a 1D tensor







        return output

 
    

# class MetaNetPseudoLabel(nn.Module):
#     def __init__(self, seq_len =768,max_shift=50, feature_dim=16, kernel_size=3,batch_size=64):
#         super(MetaNetPseudoLabel, self).__init__()
#         self.kernel_size = kernel_size
#         self.seq_len = seq_len
#         self.max_shift=max_shift

#         # Feature extractors for output and label
#         self.feature_extractor_out = nn.Sequential(
#             nn.Conv1d(1, feature_dim, kernel_size=kernel_size),  # No padding
#             # nn.ReLU(),
#             nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size),  # No padding
#             nn.ReLU()
#         )
#         self.feature_extractor_label = nn.Sequential(
#             nn.Conv1d(1, feature_dim, kernel_size=kernel_size),  # No padding
#             # nn.ReLU(),
#             nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size),  # No padding
#             nn.ReLU() 
#         )

#         self.deconv = nn.Sequential(
#             nn.ConvTranspose1d(feature_dim*2, feature_dim, kernel_size=3, stride=1, padding=1),
#             # nn.ReLU(),
#             nn.ConvTranspose1d(feature_dim, 1, kernel_size=3, stride=1, padding=1)
#         ) 
        
 
import torch.nn as nn

class MetaNetPseudoLabel(nn.Module):
    def __init__(self, seq_len=768, max_shift=50, feature_dim=16, kernel_size=3, batch_size=64,smoothing_and_amplify=False):
        super(MetaNetPseudoLabel, self).__init__()
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.max_shift = max_shift
        self.smoothing_and_amplify=smoothing_and_amplify

        # Feature extractors for output and label
        self.feature_extractor_out = nn.Sequential(
            nn.ReplicationPad1d(padding=(kernel_size // 2)),
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size, padding=0),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=(kernel_size // 2)),
 
        )
        self.feature_extractor_label = nn.Sequential(
            nn.ReplicationPad1d(padding=(kernel_size // 2)),
            nn.Conv1d(1, feature_dim, kernel_size=kernel_size, padding=0),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=(kernel_size // 2)),
 
        )

        # Ensure the deconvolution outputs the same shape
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(feature_dim * 2, feature_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ConvTranspose1d(feature_dim, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
        )

    def forward(self, x_out, x_label):
        if self.smoothing_and_amplify:  
            _,x_out=adaptive_smoothing_and_amplify_batch(x_out ) 
        # Extract features
        features_out = self.feature_extractor_out(x_out)
        features_label = self.feature_extractor_label(x_label)
        
        # Concatenate features
        combined_features = torch.cat((features_out, features_label), dim=1)
        
        # Decode features
        output = self.deconv(combined_features)
        return output

    def forward(self, y_out, y_label):
        b,_,_ =y_out.shape
        # Extract features
        h_out = self.feature_extractor_out(y_out)  # (batch_size, feature_dim, seq_len_out)
        h_label = self.feature_extractor_label(y_label)  # (batch_size, feature_dim, seq_len_label)

        

        combined_features = torch.cat([ h_out, h_label], dim=1)
        pseudolabel = self.deconv(combined_features)  # (batch_size, 1) 
        # t = torch.tanh(t)
         
        return pseudolabel



class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MetaNetLoss(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MetaNetLoss, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)



class PhaseAmplitudeAlignNet(nn.Module):
    def __init__(self,kernel_size=5,feature_dim=16,
                 adjust_sig1_magnitude = True,
                 feature_enc_sig1 = False ,
                 feature_enc_sig2 = False ):
        super(PhaseAmplitudeAlignNet, self).__init__()

        self.adjust_sig1_magnitude =adjust_sig1_magnitude
        self.feature_enc_sig2=feature_enc_sig2
        self.feature_enc_sig1=feature_enc_sig1
        
        self.feature_extractor_sig1 = nn.Sequential(

        # 定义网络层
        nn.ReplicationPad1d(kernel_size // 2),  # 第一层卷积的边界填充
        nn.Conv1d(1, 16, kernel_size=kernel_size, padding=0),  # 第一层卷积
        nn.ReLU(),
        nn.ReplicationPad1d(kernel_size // 2),  # 第二层卷积的边界填充
        nn.Conv1d(16, 1, kernel_size=kernel_size, padding=0),  # 第二层卷积
        nn.ReLU(),
             
        )
        self.feature_extractor_sig2 = nn.Sequential(
            nn.ReplicationPad1d(kernel_size // 2),  # 第一层卷积的边界填充
        nn.Conv1d(1, 16, kernel_size=kernel_size, padding=0),  # 第一层卷积
        nn.ReLU(),
        nn.ReplicationPad1d(kernel_size // 2),  # 第二层卷积的边界填充
        nn.Conv1d(16, 1, kernel_size=kernel_size, padding=0),  # 第二层卷积
        nn.ReLU(),
        )
        
        # 幅值调整模块
        self.amplitude_adjustor = nn.Linear(16, 1)

        # 可学习的相位映射模块
        self.phase_mapper_real = nn.Linear(1, 1)  # 学习实部权重
        self.phase_mapper_imag = nn.Linear(1, 1)  # 学习虚部权重

        self.magnitude_mapper  = nn.Linear(1, 1)  # 幅值只有绝对值

 

    def forward(self,signal2, signal1, ):
        batch_size, _, length = signal1.shape#(16,1,1024)
        
        # 1. 频域变换
        # signal1=self.feature_extractor_sig1(signal1)
        _,signal2=adaptive_smoothing_and_amplify_batch(signal2 ) # 模型输出 初期不稳定，进行平滑增强


        if self.feature_enc_sig2 == True:
            signal2=self.feature_extractor_sig2(signal2)

        signal1_fft = torch.fft.fft(signal1, dim=-1)  # signal1 的幅值 (16,1,1024)
        signal2_fft = torch.fft.fft(signal2, dim=-1)  # signal2 的相位
        signal1_mag = torch.abs(signal1_fft)  # 幅值 (16,1,1024)
        signal2_phase = torch.angle(signal2_fft).unsqueeze(-1)  # 相位 (16,1,1024,1)
        
       
        if self.adjust_sig1_magnitude:
            adjusted_magnitude = self.magnitude_mapper(signal1_mag.unsqueeze(-1) ).squeeze(-1)#(16,1,1024)
        
        else:adjusted_magnitude = signal1_mag
        
        
        # 3. 可学习的相位映射
        signal2_phase_real = self.phase_mapper_real(signal2_phase).squeeze(-1)#(16,1,1024)
        signal2_phase_imag = self.phase_mapper_imag(signal2_phase).squeeze(-1)#(16,1,1024)
        
        y_real = adjusted_magnitude * signal2_phase_real
        y_imag = adjusted_magnitude * signal2_phase_imag
        
        # 4. 频域重组与反变换
        y_fft = torch.complex(y_real, y_imag)
        signal3 = torch.fft.ifft(y_fft, dim=-1).real
        
        return signal3


class PhaseAmplitudeAlignNetNorm(nn.Module):
    def __init__(self,seq_len=768,kernel_size=5,feature_dim=16,
                 adjust_sig1_magnitude = True,
                 feature_enc_sig2 = False ):
        super(PhaseAmplitudeAlignNetNorm, self).__init__()

        self.adjust_sig1_magnitude =adjust_sig1_magnitude
        self.feature_enc_sig2=feature_enc_sig2
        
        self.feature_extractor_sig1 = nn.Sequential(

        # 定义网络层
        nn.ReplicationPad1d(kernel_size // 2),  # 第一层卷积的边界填充
        nn.Conv1d(1, 16, kernel_size=kernel_size, padding=0),  # 第一层卷积
        nn.ReLU(),
        nn.ReplicationPad1d(kernel_size // 2),  # 第二层卷积的边界填充
        nn.Conv1d(16, 1, kernel_size=kernel_size, padding=0),  # 第二层卷积
        nn.ReLU(),
             
        )
        self.feature_extractor_sig2 = nn.Sequential(
            nn.ReplicationPad1d(kernel_size // 2),  # 第一层卷积的边界填充
        nn.Conv1d(1, 16, kernel_size=kernel_size, padding=0),  # 第一层卷积
        nn.ReLU(),
        nn.ReplicationPad1d(kernel_size // 2),  # 第二层卷积的边界填充
        nn.Conv1d(16, 1, kernel_size=kernel_size, padding=0),  # 第二层卷积
        nn.ReLU(),
        )
        
        # 幅值调整模块
        self.amplitude_adjustor = nn.Linear(16, 1)

        # 可学习的相位映射模块
        self.phase_mapper_real = nn.Linear(seq_len, seq_len)  # 学习实部权重
        self.phase_mapper_imag = nn.Linear(seq_len,seq_len)  # 学习虚部权重

        self.magnitude_mapper  = nn.Linear(seq_len, seq_len)  # 幅值只有绝对值

    def forward(self, signal2, signal1):
        batch_size, _, length = signal1.shape


        _,signal2=adaptive_smoothing_and_amplify_batch(signal2 ) # 模型输出 初期不稳定，进行平滑增强
        
        # **Step 1: Standardize input signals (normalize to zero mean and unit variance)**
        signal1_mean = signal1.mean(dim=-1, keepdim=True)
        signal1_std = signal1.std(dim=-1, keepdim=True)
        signal1_normalized = (signal1 - signal1_mean) / (signal1_std + 1e-8)  # Avoid division by zero

        signal2_mean = signal2.mean(dim=-1, keepdim=True)
        signal2_std = signal2.std(dim=-1, keepdim=True)
        signal2_normalized = (signal2 - signal2_mean) / (signal2_std + 1e-8)

        # **Step 2: Transform to frequency domain**
        signal1_fft = torch.fft.fft(signal1_normalized, dim=-1)
        signal2_fft = torch.fft.fft(signal2_normalized, dim=-1)

        signal1_mag = torch.abs(signal1_fft)
        signal2_phase = torch.angle(signal2_fft) 

        # Adjust magnitude (optional)
        if self.adjust_sig1_magnitude:
            adjusted_magnitude = self.magnitude_mapper(signal1_mag ) 
        else:
            adjusted_magnitude = signal1_mag

        # Learnable phase mapping
        signal2_phase_real = self.phase_mapper_real(signal2_phase) 
        signal2_phase_imag = self.phase_mapper_imag(signal2_phase) 

        y_real = adjusted_magnitude * signal2_phase_real
        y_imag = adjusted_magnitude * signal2_phase_imag

        # Combine and inverse FFT
        y_fft = torch.complex(y_real, y_imag)
        signal3_normalized = torch.fft.ifft(y_fft, dim=-1).real

        # **Step 3: De-standardize signal3 to return to original scale**
        signal3 = signal3_normalized * signal1_std + signal1_mean

        return signal3

def adaptive_smoothing_and_amplify_batch(signals, window_size=15, power=2, epsilon=1e-6):
    """
    批量平滑和动态增强输入信号，高于均值部分增强，低于均值部分减弱。
    Args:
        signals: 输入信号, Tensor of shape (batch_size, 1, length)
        window_size: 平滑窗口大小
        power: 偏离幅值的增强程度，越大增强越明显
        epsilon: 避免除零的小常数
    Returns:
        smoothed_signals: 平滑后的信号, shape (batch_size, 1, length)
        enhanced_signals: 平滑并增强后的信号, shape (batch_size, 1, length)
    """
    batch_size, _, length = signals.shape

    # 平滑信号 (使用 1D 卷积实现滑动平均)
    pad_size = window_size // 2
    padded_signals = F.pad(signals, pad=(pad_size, pad_size), mode='replicate')

    # **2. 平滑信号 (1D 卷积滑动平均)**
    kernel = torch.ones(1, 1, window_size).to(signals.device) / window_size
    smoothed_signals = F.conv1d(padded_signals, kernel, groups=1) 
 
    # 计算信号均值 (沿时间维度)
    mean_values = torch.mean(smoothed_signals, dim=-1, keepdim=True)  # Shape: (batch_size, 1, 1)

    # 计算偏离量 (幅值偏离均值的绝对值)
    deviations = smoothed_signals - mean_values  # 偏离均值的值，带符号 (正偏离或负偏离)

    # 动态增强因子
    max_deviation = torch.max(torch.abs(deviations), dim=-1, keepdim=True)[0]  # 最大偏离值，用于归一化
    enhancement_factors = (torch.abs(deviations) / ((max_deviation + epsilon)/1.5)) ** power  # 动态增强因子 
    # print('enhancement_factors',enhancement_factors.shape,(enhancement_factors[:,:,300:400]).max(),(enhancement_factors[:,:,300:400]).min())
    # print('deviations',deviations.shape,(deviations[:,:,300:400]).max(),(deviations[:,:,300:400]).min())

    # print('enhancement_factors * deviations',(enhancement_factors * deviations)[:,:,300:400].shape,(enhancement_factors * deviations)[:,:,300:400].max(),(enhancement_factors * deviations)[:,:,300:400].min())
    # 高于均值部分增强，低于均值部分减弱
    enhanced_signals = smoothed_signals + enhancement_factors * deviations

    return smoothed_signals, enhanced_signals


if __name__ == '__main__':
    b,c,l=4,1,768
    x = torch.randn((b,c,l))
    y = torch.randn((b,c,l))
    model = MetaNet_TimeShift2PhaseShift()
    t = model(x,y)
    print(t.shape)
    print(t)