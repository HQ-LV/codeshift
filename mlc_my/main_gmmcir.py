import numpy as np
import pickle
import copy
import sys
import argparse
import logging
import os
 
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist 
import sys
import os
import wandb
# 添加项目根目录到模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from mlc_my.mlc import step_hmlc_K
from mlc_my.mlc_utils import clone_parameters,  DummyScheduler 
from data.utils import *
from sklearn.mixture import GaussianMixture
# 添加项目根目录到模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from backbone.swin_transformer_recons_fan import SwinTransformerReconsFan
from backbone.metanet import MetaNetPseudoLabel,PhaseAmplitudeAlignNet,MetaNet_TimeShift2PhaseShift
# from dividemix.data import ShiftSignalModified_dataloader
# from dividemix.data import DropSignalModified_dataloader
from dividemix.dataShift import ShiftSignalModified_dataloader
from backbone.soft_dtw_cuda import SoftDTW


parser = argparse.ArgumentParser(description='MLC Training Framework')
parser.add_argument('--dataset', type=str,  default='ShiftSignalModified')
parser.add_argument('--method', default='hmlc_K_mix', type=str, choices=['hmlc_K_mix', 'hmlc_K'])
parser.add_argument('--seed', type=int, default=1) 
parser.add_argument('--data_seed', type=int, default=1)
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--every', default=100, type=int, help='Eval interval (default: 100 iters)')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')

parser.add_argument('--cls_dim', type=int, default=64, help='Label embedding dim (Default: 64)')
parser.add_argument('--grad_clip', default=0.0, type=float, help='max grad norm (default: 0, no clip)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--main_lr', default=0.001, type=float, help='lr for main net')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for main net lr scheduler')
parser.add_argument('--meta_lr', default=3e-5, type=float, help='lr for meta net')
parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'sgd', 'adadelta'])
parser.add_argument('--opt_eps', default=1e-8, type=float, help='eps for optimizers')
#parser.add_argument('--tau', default=1, type=float, help='tau')
parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay (default: 5e-4)')

 
parser.add_argument('--skip', default=False, action='store_true', help='Skip link for LCN (default: False)')
parser.add_argument('--sparsemax', default=False, action='store_true', help='Use softmax instead of softmax for meta model (default: False)')
parser.add_argument('--tie', default=False, action='store_true', help='Tie label embedding to the output classifier output embedding of metanet (default: False)')


parser.add_argument('--queue_size', default=1, type=int, help='Number of iterations before to compute mean loss_g')

############## LOOK-AHEAD GRADIENT STEPS FOR MLC ##################
parser.add_argument('--gradient_steps', default=1, type=int, help='Number of look-ahead gradient steps for meta-gradient (default: 1)')

# CIFAR
# Positional arguments
parser.add_argument('--data_dir', default='/data/home/qian_hong/signal/vitaldb/0', type=str, help='Root for the datasets.')
# Optimization options
parser.add_argument('--nosgdr', default=False, action='store_true', help='Turn off SGDR.')

# Acceleration
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
 
parser.add_argument('--meta_method', default='timeshift_phaseshift', type=str,choices=['phase_amplitude_align','pseudolabel','timeshift_phaseshift'])
parser.add_argument('--task',default='reconstruction',type=str)
 
parser.add_argument('--num_meta',default=2000,type=int)
parser.add_argument('--max_shift',default=50,type=int)
parser.add_argument('--prob_shift',default=0.7,type =float)
parser.add_argument('--num_workers',default=8,type=int)
parser.add_argument('--device',default='cuda:1',type=str)
parser.add_argument('--ckpt_path',type =str)
parser.add_argument('--ckpt_path_meta',type =str)
parser.add_argument('--keep_good_subject',default=0.6,type=float)
parser.add_argument('--history_loss_eval_train',action='store_true')
parser.add_argument('--last_epoch',default=-1,type=int)

parser.add_argument('--dtwloss_r',type=float,default=0.0)
parser.add_argument('--magnitudeloss_r',type=float,default=0.0)

parser.add_argument('--shift',type=str,default='shifts',choices=['shifts','all','center'])
parser.add_argument('--max_drop_rate',default=0.05,type=float)
parser.add_argument('--input_signal',default='ppg',type=str,choices=['ppg','ecg'])

parser.add_argument('--nogmm',action='store_true')
parser.add_argument('--gmm_method',default='1_loss',type=str,choices=['gmm','1_loss'])

args = parser.parse_args() 

# //////////////// set logging and model outputs /////////////////
 
 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if 'Shift' in args.dataset:  project="Shift"
elif 'Drop' in args.dataset: project="Drop"
if 'vitaldb' in args.data_dir: project+="-VitalDB"
elif 'mimic' in args.data_dir: project+="-MIMIC"

name = f'mlc-gmmcir-{args.prob_shift}-{args.num_meta}-{args.input_signal}'


run = wandb.init(project=project, 
                 config=args,
                 name=name,
                 )  # 初始化wandb
print(f"Current run name: {run.name}")

best_main_path = f"ckpt/{run.id}-main.pth"
best_meta_path = f"ckpt/{run.id}-meta.pth"
set_cudnn(device=args.device)
set_seed(seed=args.seed)
 
if args.meta_method =='pseudolabel':
    meta_net = MetaNetPseudoLabel().to(device=args.device)
elif args.meta_method =='phase_amplitude_align':
    meta_net = PhaseAmplitudeAlignNet().to(device=args.device)
elif args.meta_method =='timeshift_phaseshift':
    meta_net = MetaNet_TimeShift2PhaseShift().to(device=args.device)
main_net = SwinTransformerReconsFan(args.task).to(device=args.device)
if args.ckpt_path:
    print('load pth',args.ckpt_path)
    main_net.load_state_dict(torch.load(args.ckpt_path))
if args.ckpt_path_meta:
    print('load pth',args.ckpt_path_meta)
    meta_net.load_state_dict(torch.load(args.ckpt_path_meta))


def setup_training(main_net, meta_net, exp_id=None):

    # ============== setting up from scratch ===================
    # set up optimizers and schedulers
    # meta net optimizer
    optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr,
                                 weight_decay=0, #args.wdecay, # meta should have wdecay or not??
                                 amsgrad=True, eps=args.opt_eps)
    scheduler = DummyScheduler(optimizer)

    # main net optimizer
    main_params = main_net.parameters() 

    if args.optimizer == 'adam':
        main_opt = torch.optim.Adam(main_params, lr=args.main_lr, weight_decay=args.wdecay, amsgrad=True, eps=args.opt_eps)
    elif args.optimizer == 'sgd':
        main_opt = torch.optim.SGD(main_params, lr=args.main_lr, weight_decay=args.wdecay, momentum=args.momentum)

    
    # follow MW-Net setting
    main_schdlr = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=[ 30,60,90], gamma=args.gamma)
     
    last_epoch = args.last_epoch

    return main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler, last_epoch
     


# //////////////////////// run experiments ////////////////////////


def test(main_net, test_loader,epoch,split='val'): # this could be eval or test
 

    # forward
    main_net.eval()
    test_losss = []

    for idx, batch in enumerate(test_loader):

        if len(batch)==3:
            data, target,target_align = batch
        elif len(batch)==6:
            data,_,target,target_align,_,_ = batch

        data, target,target_align = (data).to(args.device), (target).to(args.device),(target_align).to(args.device)

        # forward
        with torch.no_grad():
            output = main_net(data)
        
        test_losss.append(loss_fn(output,target))

    test_losss =  torch.stack(test_losss).mean(dim=0).item()
    if  epoch%10==0 or epoch==args.epochs-1 or split=='test':
        plot_compare_sig(target,output, input=data,title=split )

    # set back to train
    main_net.train()

    return test_losss
    

####################################################################################################
###  training code 
####################################################################################################
def loss_fn(outputs, labels,  reduction='mean'):
    return F.mse_loss(outputs, labels,reduction=reduction)

def magnitude_loss(outputs, labels):
    # 计算 FFT 幅值
    outputs_fft = torch.fft.fft(outputs)  # 傅里叶变换
    labels_fft = torch.fft.fft(labels)
    outputs_magnitude = torch.abs(outputs_fft)  # 幅值
    labels_magnitude = torch.abs(labels_fft)

    # 频域 MSE 损失
    loss = torch.mean(torch.log(torch.abs((outputs_magnitude - labels_magnitude))) ** 2)
    return loss
sdtw = SoftDTW(use_cuda=True, gamma=0.1)
def meta_loss_fn(recons,pseudo_label, noisy_label=None,dtwloss_r =args.dtwloss_r,magnitudeloss_r =args.magnitudeloss_r,):
    total_loss = 0
    mseloss = F.mse_loss(recons, pseudo_label)
    s=f'mseloss:{mseloss}' 
    total_loss+=mseloss
    if dtwloss_r>0: 
        sdtwloss = sdtw(recons, pseudo_label ).mean()
        s+=f'sdtwloss:{sdtwloss}' 
        total_loss+=sdtwloss*dtwloss_r
     
    if magnitudeloss_r>0:
        magnitudeloss = magnitude_loss(pseudo_label, noisy_label )
        s+=f'magnitudeloss:{magnitudeloss}'
        total_loss+=magnitudeloss*magnitudeloss_r
 
    if dtwloss_r+magnitudeloss_r>0:print(s)

    return  total_loss

def eval_train(model,all_loss,train_loader):    
    """
    •	功能：基于 GMM 模拟样本损失分布，计算清洁概率。
	•	步骤：
        1.	计算每个样本的损失值，归一化处理。
        2.	使用 GMM 将样本划分为两类：清洁样本和噪声样本。
        3.	返回每个样本的清洁概率。"""
    model.eval()
    losses = torch.zeros(len(train_loader.dataset))    
    with torch.no_grad():
        # for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        # inputs_x, inputs_x2, labels_x,labels_x_align, w_x,_
        for batch_idx, (inputs,_, targets,_,_, index) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device) 
            outputs = model(inputs) 
            loss = loss_fn(outputs, targets,reduction='none')
            loss = loss.mean(dim=(1, 2)) 
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]        
    
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    print(losses.shape,'minloss',losses.min(),'maxloss',losses.max())
    # print(losses[ np.argsort(losses)[-15:] ])
    all_loss.append(losses)


    # 保存loss到本地
    if os.path.exists('out')==True: 
        torch.save(losses, f'out/losses{args.prob_shift}.pth')
        print('save losses.pth')

    if args.history_loss_eval_train: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)

    prob = prob[:,gmm.means_.argmin()]  
    # print(losses[ np.argsort(prob)[-15:] ])


    return prob,all_loss

def train_and_test(main_net, meta_net, exp_id=None):

    # silver_loader, gold_loader, valid_loader ,test_loader= build_dataloader(
    #     data_dir = args.data_dir,
    #     keep_good_subject=0.6,
    #     shift='shifts' ,
    #     metaset_len = args.num_meta,
    #     prob_shift = args.prob_shift,
    #     max_shift =args.max_shift,
    #     batch_size = args.batch_size,
    #     num_workers= args.num_workers,) 
    
    if args.nogmm:
        silver_loader, gold_loader, valid_loader ,test_loader= build_dataloader(
            data_dir = args.data_dir,
            keep_good_subject=args.keep_good_subject,
            shift=args.shift ,
            metaset_len = args.num_meta,
            prob_shift = args.prob_shift,
            # max_drop_rate=args.max_drop_rate,
            max_shift=args.max_shift,
            batch_size = args.batch_size,
            num_workers= args.num_workers,
            dataset=args.dataset, 
            input_signal=args.input_signal,
            ) 
    else:
        train_SSMloader = ShiftSignalModified_dataloader(
            split='train',
            batch_size=args.batch_size,
            train_type='noisy' ,
            metaset_len = args.num_meta,
            keep_good_subject=args.keep_good_subject,
            shift=args.shift,
            prob_shift=args.prob_shift,
            # max_drop_rate=args.max_drop_rate,
            max_shift=args.max_shift,
            input_signal=args.input_signal
            )
        _, gold_loader, valid_loader ,test_loader= build_dataloader(
            data_dir = args.data_dir,
            keep_good_subject=args.keep_good_subject,
            shift=args.shift,
            metaset_len = args.num_meta,
            prob_shift = args.prob_shift,
            # max_drop_rate=args.max_drop_rate,
            max_shift=args.max_shift,
            batch_size = args.batch_size,
            num_workers= args.num_workers,
            dataset=args.dataset,
            no_train_noisy=True,
            input_signal=args.input_signal,
            ) 
    
    
    gold_loader_iter = iter(gold_loader)
    main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler, last_epoch = setup_training(main_net, meta_net, exp_id)

    # //////////////////////// switching on training mode ////////////////////////
    meta_net.train()
    main_net.train()

    # set up statistics 
    best_params = None #主模型参数。 
    best_val_metric = float('inf')
        
    args.dw_prev = [0 for param in meta_net.parameters()] # 0 for previous iteration, 保存 LCN 参数的上一次梯度更新值（如果需要渐变优化）。
    args.steps = 0


    all_loss = [[],[]] 
    
    for epoch in tqdm(range(last_epoch+1, args.epochs)):# change to epoch iteration
        
        if args.nogmm:
            silver_loader_train =silver_loader
        else:    
            gmm_method = args.gmm_method
            if epoch%5==0:   
                # 先计算训练集的损失，用GMM划分噪声样本和干净样本
                get_percentile = lambda epoch, maxepoch: 70 - (epoch / maxepoch) * (70 - 10) # 阈值从80到10，也就是认为干净样本是选了30%-90%
                percentile = get_percentile(epoch, args.epochs) 
                eval_train_loader = train_SSMloader.run('eval_train')   
                prob,all_loss[0]=eval_train(main_net,all_loss[0],eval_train_loader)  
                # pred = (prob > np.percentile(prob, args.prob_shift*100))
                
                if gmm_method =='gmm':
                    
                    pred = (prob > np.percentile(prob, percentile))
  
                    print('eval_train gmm prob.shape',prob.shape,'pred.shape',pred.shape,f'{pred.sum()}/{pred.shape[0]}')
                elif gmm_method =='1_loss':
                    prob = 1-all_loss[0][-1].squeeze()
 
                    pred = (prob > np.percentile(prob, percentile))
                    pred = pred.to(torch.int) 
                    print(pred.shape, )

                    print('eval_train 1_loss prob.shape',prob.shape,'pred.shape',pred.shape,f'{pred.sum()}/{pred.shape[0]}')
                # labeled_trainloader, unlabeled_trainloader = train_SSMloader.run('train',pred,prob)
                silver_loader_train, _ = train_SSMloader.run('train',pred ,prob,gmm_method=gmm_method)
        

        train_loss_s =[]
        train_loss_g = []
        main_net.train()
        for i, (data_s,_, target_s,target_s_align,_, _) in enumerate(silver_loader_train):#silver_loader：加载带噪声标签的数据。
            try:
                # data_g,target_g,target_g_align = next(gold_loader_iter)
                batch_g = next(gold_loader_iter)
            except StopIteration:   
                gold_loader_iter = iter(gold_loader)
                batch_g = next(gold_loader_iter)

            if len(batch_g)==3:
                data_g,target_g,target_g_align = batch_g
            elif len(batch_g)==6:
                data_g,_,target_g,target_g_align,_,_ = batch_g

            data_g, target_g,target_g_align = (data_g).to(args.device), (target_g).to(args.device),(target_g_align).to(args.device)
            data_s, target_s_,target_s_align = (data_s).to(args.device), (target_s).to(args.device),(target_s_align).to(args.device)

            # bi-level optimization stage
            eta = main_schdlr.get_lr()[0]#从主模型的学习率调度器中提取当前学习率，传递给训练函数。
            if args.method == 'hmlc_K':#hmlc_K：直接在噪声数据和干净数据上进行校正和优化。
                loss_g, loss_s,logit_g,logit_s,pseudo_target_s= step_hmlc_K(main_net, main_opt, loss_fn,
                                             meta_net, optimizer, meta_loss_fn,
                                             data_s, target_s_, data_g, target_g,
                                             None, None,
                                             eta, args)
            elif args.method == 'hmlc_K_mix':#hmlc_K_mix：将干净数据分成两部分，一部分用于训练，一部分用于元优化（meta-evaluation）。
                # split the clean set to two, one for training and the other for meta-evaluation
                gbs = int(target_g.size(0) / 2)
                if type(data_g) is list:
                    data_c = [x[gbs:] for x in data_g]
                    data_g = [x[:gbs] for x in data_g]
                else:
                    data_c = data_g[gbs:]
                    data_g = data_g[:gbs]
                    
                target_c = target_g[gbs:]
                target_g = target_g[:gbs]
                loss_g, loss_s,logit_g,logit_s,pseudo_target_s = step_hmlc_K(
                                            main_net, main_opt, loss_fn,
                                             meta_net, optimizer, meta_loss_fn,
                                             data_s, target_s_, data_g, target_g,
                                             data_c, target_c,
                                             eta, args)
                

            # 	loss_g：干净样本的损失（用于元优化）。   
            #   loss_s：带噪声样本的损失（用于主模型训练）。
            train_loss_s.append(loss_s.item())
            train_loss_g.append(loss_g.item())
            args.steps += 1
            if i % args.every == 0:
                
                # ''' get entropy of predictions from meta-net '''
                # # entropy用于衡量meta_net修正标签的不确定性
                # logit_s  = main_net(data_s )
                # pseudo_target_s = meta_net(logit_s.detach(), target_s_).detach()
                # entropy =loss_fn(logit_s,pseudo_target_s)  


                main_lr = main_schdlr.get_lr()[0]
                meta_lr = scheduler.get_lr()[0]

            
                print('\tIteration %d\tMain LR: %.8f\tMeta LR: %.8f\tloss_s: %.4f\tloss_g: %.4f ' %( i, main_lr, meta_lr, loss_s , loss_g , ))

        # PER EPOCH PROCESSING

        # lr scheduler
        main_schdlr.step() 

        
        train_loss_s = torch.mean(torch.tensor(train_loss_s))
        train_loss_g = torch.mean(torch.tensor(train_loss_g)) 

               
        #scheduler.step()

        # evaluation on validation set
        val_loss = test(main_net, valid_loader,epoch,'val') 
        # test_loss = test(main_net, test_loader,epoch,'test')
        wandb.log({"epoch": epoch, 
                   "lr":main_lr,
                   "train_loss": train_loss_s, 
                   "train_loss_meta": train_loss_g, 
                   "val_loss":val_loss,
                #    'test_loss':test_loss,
                   })
        if epoch%10==0 or epoch==args.epochs-1:
            if args.meta_method not in ['pseudolabel','phase_amplitude_align','timeshift_phaseshift']:
                pseudo_target_s =None
            plot_compare_sig(target_s_,logit_s, input=data_s,title='train',pseudo_labels=pseudo_target_s,labels_align=target_s_align)
            plot_compare_sig(target_g,logit_g, input=data_g,title='metaset',labels_align=target_g_align)

 
        # print('Epoch %d \train_loss_s: %.4f\train_loss_g: %.4f\tval_loss: %.4f\ttest_loss: %.4f' %( epoch,train_loss_s,train_loss_g, val_loss,test_loss))

        print('Epoch %d \train_loss_s: %.4f\train_loss_g: %.4f\tval_loss: %.4f ' %( epoch,train_loss_s,train_loss_g, val_loss))

        


        if val_loss < best_val_metric:
            best_val_metric = val_loss

            best_params = copy.deepcopy(main_net.state_dict())

            
            torch.save(main_net.state_dict(), best_main_path)
            torch.save(meta_net.state_dict(), best_meta_path)
 
    main_net.load_state_dict(best_params)
    test_loss = test(main_net, test_loader,epoch,'test') # evaluate best params picked from validation
    wandb.log({ 
                   'test_loss':test_loss,
                   })

    print('Final test_loss: %.4f' %( test_loss))
 

    return test_loss

if __name__ == '__main__':
    
    train_and_test(main_net,meta_net)