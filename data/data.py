

import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from data.tool import add_gaussian_noise_all,add_pink_noise_all
# from tool import add_gaussian_noise_all,add_pink_noise_all

# from util import time2timefreq,add_gaussian_noise,add_rise_and_fall_drift,add_salt_and_pepper_noise,add_zebra_noise,add_poisson_noise
# from data.util import time2timefreq,add_gaussian_noise,add_rise_and_fall_drift,add_salt_and_pepper_noise,add_zebra_noise,add_poisson_noise

def normalize_signal(signal,mean_val,std_val): 
    normalized = (signal - mean_val) / std_val
    return normalized  

# def add_complex_noise(signal,  mean=0, std_dev=0.15,phase_gaussian = 1,phase_zebra=2, drift_rate=0.015,noise_r = 0.5,lam = 0.7,cycle=7 ):
#     # 随机选择一个位置，确保能选择到30%长度的连续区间
#     num_samples = signal.shape[-1]
#     start_idx = torch.randint(0, num_samples - int(noise_r * num_samples), (1,)).item()  # 随机起始索引
#     end_idx = start_idx + int(noise_r * num_samples)  # 计算连续区间的结束索引
#     signal = add_rise_and_fall_drift(signal, start_idx, end_idx, drift_rate)
#     signal = add_gaussian_noise(signal, start_idx, end_idx, mean, std_dev,phase_gaussian)
#     signal = add_zebra_noise(signal, start_idx, end_idx,phase_zebra,cycle)
#     signal = add_poisson_noise(signal, start_idx, end_idx,lam) 
    
#     return signal

class ShiftSignal(data.Dataset):
    def __init__(self, 
        # data_dir=r'/home/notebook/data/group/bcg2ppg_data/preprocessed_data/labeled_data',
        data_dir = r'/data/home/qian_hong/signal/vitaldb/0',
        split='train',
        train_type = 'noisy',#train_type=['noisy','metaset','train_noisy_metaset]
        metaset_len = 100,
        
        normalize=False, 
        seed=42,
        sample_rate=1.0,
        keep_good_subject=0.6,
        shift='all',# from 'center' or 'shifts' or 'all'
        max_shift =50,
        prob_shift = 1.0,
        temporary_visulize_train = False,
        signal_noise = 'gaussian',# 'gaussian' or 'pink'
    ): 
        # Set all input args as attributes
        self.__dict__.update(locals())
        if self.signal_noise == 'gaussian':
            self.noise_func = add_gaussian_noise_all
        elif self.signal_noise == 'pink':
            self.noise_func = add_pink_noise_all

        self.load_signals()

        
 
    
    def load_signals(self): 
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


        data_dir = self.data_dir 
        signals_file_path = op.join( data_dir, self.split, 'signals.npz')
        data = np.load(signals_file_path,allow_pickle=True)


        # if self.keep_good_subject<1.0:
        #     sub = np.load(op.join( data_dir, self.split, f'filtered_subjects_{self.keep_good_subject}.npz'),allow_pickle=True)
        #     good_indices = np.array([i for i, subject in enumerate(data['Subject']) if subject in sub['good_subject']])
        # else:
        #     good_indices = np.arange(data[data.files[0]].shape[0])
        if self.keep_good_subject<1.0:

            raw_keep_good_subject = 0.6
            sub = np.load(op.join( data_dir, self.split, f'filtered_subjects_{raw_keep_good_subject}.npz'),allow_pickle=True)
            subject_len = len(sub['good_subject'])
            # 总共是60%的患者，mae从大到小排列。
            # 60%: [:],所有
            # 40%: [-raw_keep_good_subject_len*(self.keep_good_subject/raw_keep_good_subject):],后40%
            sub_good = sub['good_subject'][-int(subject_len*(self.keep_good_subject/raw_keep_good_subject)):]
            good_indices = np.array([i for i, subject in enumerate(data['Subject']) if subject in sub_good])
        else:
            good_indices = np.arange(data[data.files[0]].shape[0])
 
 
        random_indices = np.random.choice(good_indices, size=int(good_indices.shape[0]*self.sample_rate), replace=False)
        # random_indices.sort()

        if self.split =='train' and self.train_type=='metaset':
                indices = random_indices[:self.metaset_len] # 比如前100条作为metaset
        elif self.split =='train' and self.train_type=='noisy':
            indices = random_indices[self.metaset_len:]
            
        elif self.split =='train' and self.train_type=='train_noisy_metaset':
            indices = random_indices 
        elif self.split in  [ 'val','test']:
            indices = random_indices
        indices.sort()
 
 

        

        



        
        if self.normalize:
            # ppg = torch.from_numpy(normalize_signal(data['ppg'],data['ppg_mean'],data['ppg_std'])).to(torch.float32).unsqueeze(1)
            # bcg_env = torch.from_numpy(normalize_signal(data['bcg_env'],data['bcg_env_mean'],data['bcg_env_std'])).to(torch.float32).unsqueeze(1)  
            ppg,bp,ecg = [
               torch.from_numpy(normalize_signal(data[f'{sig}'],data[f'{sig}_mean'],data[f'{sig}_std'])).to(torch.float32).unsqueeze(1) 
               for sig in ['ppg','bp','ecg' ]
            ]
            
        else:
            # ppg = torch.from_numpy(data['ppg']).to(torch.float32).unsqueeze(1)
            # bcg_env = torch.from_numpy(data['bcg_env']).to(torch.float32).unsqueeze(1) 
            ppg,bp,ecg = [
               torch.from_numpy( (data[f'{sig}'] )).to(torch.float32).unsqueeze(1) 
               for sig in ['ppg','bp','ecg' ]
            ]
        
        sbp,dbp = [
               torch.from_numpy( (data[f'{sig}'] )).to(torch.float32).unsqueeze(1) 
               for sig in ['sbp','dbp' ]
            ]
        
        ppg = ppg[indices]
        bp=bp[indices]
        ecg=ecg[indices]

        ppg_noise = torch.stack([self.noise_func(ss[0].clone(), ) for ss in ppg]).unsqueeze(1) 

 

        self.labels = np.concatenate([sbp[indices],dbp[indices]], axis=1) 

        if self.shift == 'all':
            pass

        else:
 
            shifts = np.load(f'/data/home/qian_hong/signal/vitaldb/0/shift_id/{self.split}_{self.max_shift}_{self.prob_shift}.npz')

            seq_len = shifts['center'][0][1]-shifts['center'][0][0]
            target_center = shifts['center'][indices] 
            if (self.split in [ 'val','test']) or (self.split =='train' and self.train_type=='metaset'): 
                input_shifts = shifts['center'][indices]
                input_s = np.array([0]*indices.shape[0])
            elif self.split =='train' and self.train_type=='noisy':
                input_shifts = shifts['shifts'][indices]
                input_s = shifts['s'][indices]
            
            elif self.split =='train' and self.train_type=='train_noisy_metaset':
                input_shifts = shifts['shifts'][indices]
                input_s = shifts['s'][indices]

                input_shifts[:self.metaset_len] = shifts['center'][indices[:self.metaset_len]]
                input_s[:self.metaset_len] = np.array([0]*self.metaset_len)
 
        self.ppg =  torch.zeros((indices.shape[0], 1, seq_len), dtype=ppg.dtype )
        self.ecg =  torch.zeros((indices.shape[0], 1, seq_len), dtype=ecg.dtype ) 
        self.bp = torch.zeros((indices.shape[0], 1, seq_len), dtype=bp.dtype )
        self.bp_align = torch.zeros((indices.shape[0], 1, seq_len), dtype=bp.dtype )

        self.ppg_noise = torch.zeros((indices.shape[0], 1, seq_len), dtype=ppg.dtype )

        for idx in range(self.ppg.shape[0]):
            # input
            self.ppg[idx]      = ppg[idx][:, input_shifts[idx][0]: input_shifts[idx][1]]
            self.ecg[idx]      = ecg[idx][:, input_shifts[idx][0]: input_shifts[idx][1]] 
            self.bp_align[idx] =  bp[idx][:, input_shifts[idx][0]: input_shifts[idx][1]]
            self.bp[idx] =   bp[idx][:, target_center[idx][0]:target_center[idx][1]]

            self.ppg_noise[idx] = ppg_noise[idx][:, input_shifts[idx][0]: input_shifts[idx][1]]
        
        
        
        print('ppg',self.ppg.shape) 
        print('ecg',self.ecg.shape)
        print('bp',self.bp.shape)
        print('bp_align',self.bp_align.shape)
        print('labels',self.labels.shape)
        print('ppg_noise',self.ppg_noise.shape)
        self.idx_center = target_center
        self.idx_shifts = input_shifts
        self.indices = indices
        self.s = input_s

        

    
    def __len__(self): 
        return self.ppg.shape[0]
 
    
    # def __getitem__(self, idx):
        
    #     return {
    #             # input
    #             'ppg': self.ppg[idx], 
    #             'ecg': self.ecg[idx],
 

    #             # output
    #             'bp': self.bp[idx] ,
    #             'labels': self.labels[idx], 
    #             'idx_center': self.idx_center[idx],
    #             'idx_shifts': self.idx_shifts[idx],
    #             'all_ppg': self.all_ppg[idx],
    #             's':self.s[idx],
    #     }


    def __getitem__(self, idx):
        return self.ppg[idx], self.bp[idx],self.bp_align[idx]
        

class ShiftSignalModified(ShiftSignal):
    def __init__(self,   *args, **kwargs):
        # 调用父类初始化
        super().__init__(*args, **kwargs)

        
        self.pred = torch.full((self.ppg.shape[0], ), -1) # 初始预测为-1
        self.probability = torch.full((self.ppg.shape[0], ), -1) # 初始概率为-1

        self.idx = torch.arange(self.ppg.shape[0])     
        
       
    def get_prob(self,probability,pred):
        self.probability = probability
        self.pred = pred
    def __getitem__(self, idx):
         
        return self.ppg[idx], self.ppg_noise[idx],self.bp[idx],self.bp_align[idx],self.probability[idx],self.idx[idx]
   


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # 添加导入

    # 初始化 HypertensionDataenvfiveReg 实例

    split = 'test'
    train_type = 'train_noisy_metaset'
    hypertension_data_env = ShiftSignal(
        normalize=False,
        data_dir = '/data/home/qian_hong/signal/vitaldb/0',
        keep_good_subject=0.6,
        shift='shifts' ,
        split=split,
        train_type=train_type,
          ) 
    
    data = np.load(op.join(hypertension_data_env.data_dir, hypertension_data_env.split, 'signals.npz'),allow_pickle=True)
    ppg = data['ppg'][hypertension_data_env.indices]
    abp = data['bp'][hypertension_data_env.indices]
    idx_shifts = hypertension_data_env.idx_shifts
    idx_center = hypertension_data_env.idx_center

    print('indices shape',hypertension_data_env.indices.shape)
    print('indices',hypertension_data_env.indices)
    print('s',hypertension_data_env.s)
    print('idx_shifts',idx_shifts.shape)
    print('idx_center',idx_center.shape)
 
    # 使用 DataLoader 加载数据
    data_loader = DataLoader(hypertension_data_env, batch_size=1, shuffle=False)

    # 打印 id=1 的信号的形状
 
    name = f'fig/{split}'
    if split=='train':name+=f'-{train_type}'
    for idx, signals in enumerate(data_loader):
        ppgi,bpi,bp_aligni = signals
        if idx in [50,98,150]:  # id=1
            # print('ppg shape:', signals['ppg'].shape)
            # print('bp shape:', signals['bp'].shape)
            # print('ecg shape:', signals['ecg'].shape)
     

            # 绘制信号
            # plt.figure(figsize=(12, 4))

            # plt.subplot(2, 1, 1)
            # plt.plot(ppgi.numpy().flatten())
            # plt.title('PPG Signal')
            # plt.xlabel('Time')
            # plt.ylabel('Amplitude')

            # plt.subplot(2, 1, 2)
            # plt.plot(bpi.numpy().flatten())
            # plt.title('BP Signal')
            # plt.xlabel('Time')
            # plt.ylabel('Amplitude') 
 

            # plt.tight_layout()
            # plt.savefig(f'{name}-{idx}.png')
            # plt.show()  # 显示绘图
            # plt.close()

     
            plt.figure(figsize=(12, 8 ))
            plt.subplot(2, 1, 1)
            plt.plot(range(0,1250), ppg[idx] , 'b--', label='raw') 
            plt.plot(range(idx_shifts[idx][0],idx_shifts[idx][1]), ppgi[0][0], 'r-', label='Shift') 
            # 添加原始最大损失值的竖线     
            plt.axvline(x=idx_shifts[idx][0], color='blue', linestyle='--', linewidth=0.8,) 

            plt.subplot(2, 1, 2)
            plt.plot(range(0,1250), abp[idx] , 'b--', label='raw') 
            plt.plot(range(idx_center[idx][0],idx_center[idx][1]), bpi[0][0], 'r--', label='Shift') 
            # plt.plot(range(idx_center[idx][0], idx_center[idx][1]),bp_aligni[0][0], 'g--', label='Align') 
            plt.axvline(x=idx_center[idx][0], color='red', linestyle='--', linewidth=0.8,) 
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{name}-{idx}.png')
            print(f'{idx}.png saved')
            plt.show()
            plt.close()
            
        