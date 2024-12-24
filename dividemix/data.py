from torch.utils.data import Dataset, DataLoader,TensorDataset 
import torchvision.transforms as transforms
import random
import numpy as np 
import json
import os
import sys
import torch
from torchnet.meter import AUCMeter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from data.data import ShiftSignal,ShiftSignalModified
from data.datadrop import DropSignal,DropSignalModified



# class ShiftSignalModified(ShiftSignal):
#     def __init__(self, mode="default", *args, **kwargs):
#         """
#         初始化 ShiftSignalModified 类。

#         参数:
#             mode (str): 模式，用于控制数据处理方式。
#             *args, **kwargs: 传递给父类 ShiftSignal 的其他参数。
#         """
#         # 调用父类初始化
#         super().__init__(*args, **kwargs)
        
#         # 新增属性
#         self.mode = mode
#     def get_prob(self,probability):
#         self.probability = probability

#     def __getitem__(self, idx):

        
#         if self.mode=='labeled':
#             return self.ppg[idx], self.ppg_noise[idx],self.bp[idx],self.bp_align[idx],self.probability[idx]
#         elif self.mode=='unlabeled':
#             return self.ppg[idx], self.ppg_noise[idx],self.bp[idx],self.bp_align[idx]
#         elif self.mode=='all':
#             return self.ppg[idx], self.bp[idx],self.bp_align[idx],idx
        

#         elif self.mode in ['test','val']:
#             return self.ppg[idx], self.bp[idx],self.bp_align[idx]



class DropSignalModified_dataloader():  
    def __init__(self, 
                 
                 split ='trian',
                 metaset_len = 2000,
                 data_dir = '/data/home/qian_hong/signal/vitaldb/0_drop',
                keep_good_subject=0.4,
                shift='shift' ,
                
      
                prob_shift=0.7,
                batch_size = 64,
                num_workers= 8,

                
                train_type='train_noisy_metaset',
                # DropSignal相关
                max_drop_rate =0.05,

                input_signal ='ppg'
        
        ): 
        self.__dict__.update(locals())

        self.dataset = DropSignalModified(
                split=self.split,
                train_type = train_type ,
                data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_drop_rate=max_drop_rate,prob_shift=prob_shift,
                input_signal=input_signal
            ) 
        
     
    def run(self,mode,pred=[],prob=[]):
 
        if mode == 'warmup':# 2倍batchsize
            ppg,ppg_noise,bp,bp_align, probability ,idx =self.dataset[:]
            tensor_dataset = TensorDataset(ppg,ppg_noise,bp,bp_align, probability ,idx)
            loader = DataLoader(tensor_dataset, batch_size=self.batch_size*2, shuffle=True, num_workers=self.num_workers)
            return loader

        elif  mode=='eval_train':# 1倍batchsize
            ppg,ppg_noise,bp,bp_align, probability ,idx =self.dataset[:]
            tensor_dataset = TensorDataset(ppg,ppg_noise,bp,bp_align, probability ,idx)
            loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return loader
        
        elif  mode == 'train':#分labeled和unlabeled
            pred_idx = pred.nonzero()
            prob = torch.tensor(prob).float()
            self.dataset.get_prob(prob,pred)
            ppg,ppg_noise,bp,bp_align, probability ,idx =self.dataset[pred_idx]
            print('labeled',ppg.shape )
            # for iii,iiii in enumerate([ppg,ppg_noise,bp,bp_align, probability ,idx]):
            #     if type(iiii) == torch.Tensor:
            #         print(iii,type(iiii),iiii.shape) 
            #     elif type(iiii) == list:
            #         print(iii,type(iiii),len(iiii))
            tensor_dataset = TensorDataset(ppg,ppg_noise,bp,bp_align, probability ,idx)
            labeled_loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
            unlabeled_idx = (1-pred).nonzero() 
            ppg,ppg_noise,bp,bp_align, probability ,idx =self.dataset[unlabeled_idx]
            print('unlabeled',ppg.shape) 
            tensor_dataset = TensorDataset(ppg,ppg_noise,bp,bp_align, probability ,idx)
            unlabeled_loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
            return labeled_loader,unlabeled_loader


        elif  mode in ['test','val']:
            ppg,ppg_noise,bp,bp_align, probability ,idx =self.dataset[:]
            tensor_dataset = TensorDataset(ppg,ppg_noise,bp,bp_align, probability ,idx)
            loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return loader
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
         

if __name__ == "__main__":
    prob_shift = 1.0
    dataloader = DropSignalModified_dataloader(split='train',prob_shift=prob_shift )
    

    print('---------------------------')
    mode='train'
    print(mode)
    lens =len(dataloader.dataset)
    probabilities = torch.rand(lens) 
    predictions = torch.randint(0, 2, (lens,))
    labeled_loader,unlabeled_loader = dataloader.run(mode=mode,prob=probabilities,pred=predictions)
    print(len(labeled_loader.dataset))
    print(len(unlabeled_loader.dataset)) 
    j=0
    for ppg,ppg_noise,bp,bp_align, probability ,idx in labeled_loader:
        print(ppg.shape,ppg_noise.shape,bp.shape,bp_align.shape, probability.shape,idx.shape)
        j+=1
        if j>0:
            break

    mode='eval_train'
    print(mode)
    eval_train_loader = dataloader.run(mode=mode)
    print(len(eval_train_loader.dataset))
    j=0
    for ppg,ppg_noise,bp,bp_align, probability ,idx in eval_train_loader:
        print(ppg.shape,ppg_noise.shape,bp.shape,bp_align.shape, probability.shape,idx.shape)
        j+=1
        if j>0:
            break
    mode='warmup'
    print(mode)
    warm_loader = dataloader.run(mode=mode)
    print(len(warm_loader.dataset))
    j=0
    for ppg,ppg_noise,bp,bp_align, probability ,idx in warm_loader:
        print(ppg.shape,ppg_noise.shape,bp.shape,bp_align.shape, probability.shape,idx.shape)
        j+=1
        if j>0:
            break


    test_dataloader = DropSignalModified_dataloader(split='val' )
    print('---------------------------')
    mode='val'
    print(mode)
    test_loader = test_dataloader.run(mode=mode)
    print(len(test_loader.dataset))
    j=0
    for ppg,ppg_noise,bp,bp_align, probability ,idx in test_loader:
        print(ppg.shape,ppg_noise.shape,bp.shape,bp_align.shape, probability.shape,idx.shape)
        j+=1
        if j>0:
            break