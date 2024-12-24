import os
import sys
import torch  
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) 



def set_cudnn(device='cuda'):
    torch.backends.cudnn.enabled = (device == 'cuda')
    torch.backends.cudnn.benchmark = (device == 'cuda')


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def build_dataloader_pretrain(
        expname='pretrain',
        data_dir = '/data/home/qian_hong/signal/vitaldb/0',
        keep_good_subject=0.6,
        #for shift : from 'center' or 'shifts' or 'all' 
        #for drop : from 'drop' or 'align'
        shift='shifts' ,

        metaset_len = 100,
        batch_size = 64,
        num_workers= 8,

        dataset='ShiftSignal',
        # 噪声样本的比例
        prob_shift=0.7,#prob_drop = 0.7

        # ShiftSignal相关
        max_shift =50,
        # DropSignal相关
        max_drop_rate =0.05,

        input_signal='ppg',

        
):  
    if expname=='pretrain':
        train_type = 'metaset'
    elif expname=='baseline':
        train_type='train_noisy_metaset'

    # if 'DropSignal' in dataset:
    #     if dataset =='DropSignal':
    #         from data.datadrop import DropSignal as Signal
    #     elif dataset =='DropSignalModified':
    #         from data.datadrop import DropSignalModified as Signal
        
    #     train = Signal(
    #             split='train',
    #             train_type = train_type ,
    #             data_dir = data_dir , keep_good_subject=keep_good_subject,   metaset_len = metaset_len,  shift=shift,max_drop_rate=max_drop_rate,prob_shift=prob_shift,
    #             input_signal = input_signal
    #         ) 
    #     val= Signal(
    #             split='val',
    #             train_type = '' ,
    #             data_dir = data_dir, keep_good_subject=keep_good_subject,   metaset_len = metaset_len, shift=shift,max_drop_rate=max_drop_rate,prob_shift=prob_shift,
    #             input_signal = input_signal
    #         ) 
    #     test= Signal(
    #             split='test',
    #             train_type = '' ,
    #             data_dir = data_dir, keep_good_subject=keep_good_subject, metaset_len = metaset_len, shift=shift,max_drop_rate=max_drop_rate,prob_shift=prob_shift,
    #             input_signal = input_signal
    #         ) 
        

    if 'ShiftSignal' in dataset:
        if dataset =='ShiftSignal':
            from data.data import ShiftSignal as Signal
        elif dataset =='ShiftSignalModified':
            from data.data import ShiftSignalModified as Signal


        train = Signal(
                split='train',
                train_type = train_type ,
                data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_shift =max_shift,prob_shift=prob_shift,
            ) 
        val= Signal(
                split='val',
                train_type = '' ,
                data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_shift =max_shift,prob_shift=prob_shift,
            ) 
        test= Signal(
                split='test',
                train_type = '' ,
                data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_shift =max_shift,prob_shift=prob_shift,
            ) 
    
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers= num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers= num_workers)
     

    
    return train_loader,val_loader ,test_loader
    
  

    

def build_dataloader(
        data_dir = '/data/home/qian_hong/signal/vitaldb/0',
        keep_good_subject=0.6,
        shift='shifts' ,
        metaset_len = 100,
        max_shift =50,
        prob_shift=0.7,
        batch_size = 64,
        num_workers= 8,
        dataset='DropSignal',
        no_train_noisy =False,#避免加载训练数据集时间太长
        max_drop_rate =0.05,
        input_signal = 'ppg',
):
    if dataset =='ShiftSignal':
        from data.data import ShiftSignal as Signal
    elif dataset =='ShiftSignalModified':
        from data.data import ShiftSignalModified as Signal

    if not no_train_noisy:
        train_noisy = Signal(
                split='train',
                train_type = 'noisy' ,
                data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_shift =max_shift,prob_shift=prob_shift,
              ) 
        train_noisy_loader = DataLoader(train_noisy, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    else:
        train_noisy_loader = None


    train_metaset= Signal(
            split='train',
            train_type = 'metaset' ,
            data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_shift =max_shift, 
          ) 
    val= Signal(
            split='val',
            train_type = '' ,
            data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_shift =max_shift,
          ) 
    test= Signal(
            split='test',
            train_type = '' ,
            data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_shift =max_shift,
          ) 

    # if dataset =='DropSignal':
    #     from data.datadrop import DropSignal as Signal
    # elif dataset =='DropSignalModified':
    #     from data.datadrop import DropSignalModified as Signal

    # if not no_train_noisy:
    #     train_noisy = Signal(
    #             split='train',
    #             train_type = 'noisy' ,
    #             data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len, prob_shift=prob_shift,max_drop_rate=max_drop_rate,
    #             input_signal = input_signal
    #           ) 
    #     train_noisy_loader = DataLoader(train_noisy, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    # else:
    #     train_noisy_loader = None


    # train_metaset= Signal(
    #         split='train',
    #         train_type = 'metaset' ,
    #         data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_drop_rate=max_drop_rate,
    #         input_signal = input_signal
    #       ) 
    # val= Signal(
    #         split='val',
    #         train_type = '' ,
    #         data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_drop_rate=max_drop_rate,
    #         input_signal = input_signal
    #       ) 
    # test= Signal(
    #         split='test',
    #         train_type = '' ,
    #         data_dir = data_dir, keep_good_subject=keep_good_subject, shift=shift, metaset_len = metaset_len,max_drop_rate=max_drop_rate,
    #         input_signal = input_signal
    #       ) 
    
    
    train_metaset_loader = DataLoader(train_metaset, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers= num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers= num_workers)
 

    
    return train_noisy_loader,train_metaset_loader,val_loader,test_loader



def compute_loss(net, data_loader, criterion, device,):
    net.eval() 
    total_loss = 0.

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):

            if len(batch)==3:
                inputs, labels,labels_align = batch
            elif len(batch)==6:
            
                inputs, _,labels,labels_align,w_x,_ = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
             

    return total_loss / (batch_idx + 1) ,labels,outputs,inputs




def plot_compare_sig(labels, recons,logvar=None,input = None,title = 'val',pseudo_labels=None,labels_align=None):
    # 随机选择一个样本的索引
    sample_index = random.randint(0, labels.shape[0] - 1)
    
  
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 1, 1)  # 创建第一个子图
    plt.plot(input[sample_index, 0, :].detach().cpu().numpy(), label='Input' )

    plt.title(f'id-in-batch:{sample_index}')

    plt.subplot(2, 1, 2)  # 创建第二个子图
    plt.plot(labels[sample_index, 0, :].detach().cpu().numpy(),'b--', label='Label', )
    plt.plot(recons[sample_index, 0, :].detach().cpu().numpy(),'r--', label='Recons', )
    if pseudo_labels is not None:
        plt.plot(pseudo_labels[sample_index, 0, :].detach().cpu().numpy(),'g--', label='Pseudo' )
    if labels_align is not None:
        plt.plot(labels_align[sample_index, 0, :].detach().cpu().numpy(),'y--', label='Align' )
    if logvar is not None:
        pass
    plt.legend(loc="lower right")  
    wandb.log({f"True vs Recons ({title})": wandb.Image(plt)}) 
    plt.close()


