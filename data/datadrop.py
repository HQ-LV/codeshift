

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

class DropSignal(data.Dataset):
    def __init__(self, 
        # data_dir=r'/home/notebook/data/group/bcg2ppg_data/preprocessed_data/labeled_data',
        data_dir = r'/data/home/qian_hong/signal/vitaldb/0_drop',
        split='train',
        train_type = 'noisy',#train_type=['noisy','metaset','train_noisy_metaset]
        metaset_len = 100,
        shift='drop', # in [drop,align]
        normalize=False, 
        seed=42,
        sample_rate=1.0,
        keep_good_subject=0.6,  
        max_drop_rate = 0.05,
        prob_shift = 0.7, 
        signal_noise = 'gaussian',# 'gaussian' or 'pink'
        input_signal = 'ppg',
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
        split_file = self.split
        if self.split == 'train':
            split_file = f'train_dropr={self.max_drop_rate}_prob={self.prob_shift}'

            
        signals_file_path = op.join( data_dir, split_file, 'signals.npz')
        data = np.load(signals_file_path,allow_pickle=True)


        if self.keep_good_subject<1.0:

            raw_keep_good_subject = 0.6
            sub = np.load(op.join( data_dir, split_file, f'filtered_subjects_{raw_keep_good_subject}.npz'),allow_pickle=True)
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
 

        filename = ['ppg','bp','ecg' ]
        if self.shift == 'align' and self.split == 'train':
            filename = ['ppg','raw_bp','ecg' ]
     
        if self.normalize:
            ppg,bp,ecg = [ torch.from_numpy(normalize_signal(data[f'{sig}'],data[f'{sig}_mean'],data[f'{sig}_std'])).to(torch.float32).unsqueeze(1) 
               for sig in filename ] 
        else:
            ppg,bp,ecg = [ torch.from_numpy( (data[f'{sig}'] )).to(torch.float32).unsqueeze(1) 
               for sig in filename ]
        
        # 如果是train_noisy_metaset,将前metaset_len条bp替换为raw_bp
        if self.split =='train':
            bp_align= torch.from_numpy( (data['raw_bp'] )).to(torch.float32).unsqueeze(1) 
        else:
            bp_align = bp
            


        sbp,dbp = [
               torch.from_numpy( (data[f'{sig}'] )).to(torch.float32).unsqueeze(1) 
               for sig in ['sbp','dbp' ]
            ]
        
        ppg = ppg[indices]
        bp=bp[indices]
        ecg=ecg[indices]
        bp_align = bp_align[indices]
        
        if self.split =='train' and self.train_type in ['train_noisy_metaset','metaset']:
            bp[:self.metaset_len] = bp_align[:self.metaset_len] 

        if self.input_signal == 'ppg':
            noise = torch.stack([self.noise_func(ss[0].clone(), ) for ss in ppg]).unsqueeze(1) 
        elif self.input_signal == 'ecg':
            noise = torch.stack([self.noise_func(ss[0].clone(), ) for ss in ecg]).unsqueeze(1) 

 

        self.labels = np.concatenate([sbp[indices],dbp[indices]], axis=1) 
        self.ppg= ppg
        self.bp = bp
        self.ecg = ecg
        self.noise = noise
        self.bp_align = bp_align
        
  
        print('ppg',self.ppg.shape) 
        print('ecg',self.ecg.shape)
        print('bp',self.bp.shape) 
        print('labels',self.labels.shape)
        print('noise',self.noise.shape) 
        print('bp_align',self.bp_align.shape) 
        self.indices = indices 

        

    
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
        if self.input_signal == 'ppg':
            return self.ppg[idx], self.bp[idx],self.bp_align[idx]
        elif self.input_signal == 'ecg':
            return self.ecg[idx], self.bp[idx],self.bp_align[idx]

class DropSignalModified(DropSignal):
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
        if self.input_signal == 'ppg':
            return self.ppg[idx], self.noise[idx],self.bp[idx],self.bp_align[idx],self.probability[idx],self.idx[idx]
        elif self.input_signal == 'ecg':
            return self.ecg[idx], self.noise[idx],self.bp[idx],self.bp_align[idx],self.probability[idx],self.idx[idx] 
   


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # 添加导入

    # 初始化 HypertensionDataenvfiveReg 实例

    split = 'train'
    train_type = 'metaset'
    shift = 'shift'
    max_drop_rate = 0.05
    prob_shift = 1.0
    hypertension_data_env = DropSignal(
  
 
        split=split,
        train_type=train_type,
        shift=shift,
        max_drop_rate=max_drop_rate,
        prob_shift=prob_shift,
          ) 
    
    split_file = hypertension_data_env.split
    if split=='train':
        split_file =  f'train_dropr={ max_drop_rate}_prob={prob_shift}'
    data = np.load(op.join(hypertension_data_env.data_dir, split_file, 'signals.npz'),allow_pickle=True)
    ppg = data['ppg'][hypertension_data_env.indices]
    abp = data['bp'][hypertension_data_env.indices]

    print('indices shape',hypertension_data_env.indices.shape)
    print('indices',hypertension_data_env.indices)

 
    # 使用 DataLoader 加载数据
    data_loader = DataLoader(hypertension_data_env, batch_size=1, shuffle=False)

    # 打印 id=1 的信号的形状
 
    name = f'fig/{split}'
    if split=='train':name+=f'-{train_type}'
    for idx, (ppgi,bpi,bp_aligni) in enumerate(data_loader):
        if idx in [50,99,110,200]:  # id=1
            # print('ppg shape:', signals['ppg'].shape)
            # print('bp shape:', signals['bp'].shape)
            # print('ecg shape:', signals['ecg'].shape)
     

            # 绘制信号
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(ppgi.numpy().flatten(),)
            plt.title('PPG Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')

            plt.subplot(2, 1, 2)
            plt.plot(bpi.numpy().flatten(),'r--',label='bp')
            plt.plot(bp_aligni.numpy().flatten(),'g--',label='bp_align')
            plt.title('BP Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')

            # plt.subplot(3, 1, 3)
            # plt.plot(ecgi.numpy().flatten())
            # plt.title('ECG Signal')
            # plt.xlabel('Time')
            # plt.ylabel('Amplitude')
 
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{name}-{idx}.png')
            plt.show()  # 显示绘图
            plt.close()
 
            
        if idx>201:break
 