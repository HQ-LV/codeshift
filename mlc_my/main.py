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

# 添加项目根目录到模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from backbone.swin_transformer_recons_fan import SwinTransformerReconsFan
from backbone.metanet import MetaNetPseudoLabel,PhaseAmplitudeAlignNet



parser = argparse.ArgumentParser(description='MLC Training Framework')
parser.add_argument('--dataset', type=str, choices=['ShiftSignal' ], default='ShiftSignal')
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
 
parser.add_argument('--meta_method', default='pseudolabel', type=str,choices=['phase_amplitude_align','pseudolabel'])
parser.add_argument('--task',default='reconstruction',type=str)
 
parser.add_argument('--num_meta',default=2000,type=int)
parser.add_argument('--max_shift',default=50,type=int)
parser.add_argument('--prob_shift',default=0.4,type =float)
parser.add_argument('--num_workers',default=8,type=int)
parser.add_argument('--device',default='cuda:1',type=str)
parser.add_argument('--ckpt_path',type =str)


args = parser.parse_args()

# //////////////// set logging and model outputs /////////////////
 
 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

run = wandb.init(project="Shift-VitalDB", 
                 config=args,
                 name = f'mlc-{args.prob_shift}-{args.num_meta}')  # 初始化wandb
print(f"Current run name: {run.name}")

best_main_path = f"ckpt/{run.id}-main.pth"
best_meta_path = f"ckpt/{run.id}-meta.pth"
set_cudnn(device=args.device)
set_seed(seed=args.seed)
 
if args.meta_method =='pseudolabel':
    meta_net = MetaNetPseudoLabel().to(device=args.device)
elif args.meta_method =='phase_amplitude_align':
    meta_net = PhaseAmplitudeAlignNet().to(device=args.device)
    
main_net = SwinTransformerReconsFan(args.task).to(device=args.device)
if args.ckpt_path:
    print('load pth',args.ckpt_path)
    main_net.load_state_dict(torch.load(args.ckpt_path))
 

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

    if args.dataset in ['ShiftSignal' ]:
        # follow MW-Net setting
        main_schdlr = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=[ 30,60,90], gamma=0.1)
     
    last_epoch = -1

    return main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler, last_epoch
     


# //////////////////////// run experiments ////////////////////////


def test(main_net, test_loader,epoch,split='val'): # this could be eval or test
 

    # forward
    main_net.eval()
    test_losss = []

    for idx, (data, target,target_align) in enumerate(test_loader):
        data, target,target_align = (data).to(args.device), (target).to(args.device),(target_align).to(args.device)

        # forward
        with torch.no_grad():
            output = main_net(data)
        
        test_losss.append(loss_fn(output,target))

    test_losss =  torch.stack(test_losss).mean(dim=0).item()
    if  epoch%20==0 or split=='test':
        plot_compare_sig(target,output, input=data,title=split )

    # set back to train
    main_net.train()

    return test_losss
    

####################################################################################################
###  training code 
####################################################################################################
def loss_fn(outputs, labels,  reduction='mean'):
    return F.mse_loss(outputs, labels,reduction=reduction)

def train_and_test(main_net, meta_net, exp_id=None):

    silver_loader, gold_loader, valid_loader ,test_loader= build_dataloader(
        data_dir = args.data_dir,
        keep_good_subject=0.6,
        shift='shifts' ,
        metaset_len = args.num_meta,
        prob_shift = args.prob_shift,
        max_shift =args.max_shift,
        batch_size = args.batch_size,
        num_workers= args.num_workers,) 
    
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


    
    for epoch in tqdm(range(last_epoch+1, args.epochs)):# change to epoch iteration
 
        train_loss_s =0
        train_loss_g = 0
        for i, (data_s, target_s,target_s_align) in enumerate(silver_loader):#silver_loader：加载带噪声标签的数据。
            try:
                data_g, target_g,target_g_align = next(gold_loader_iter)
            except StopIteration:
                gold_loader_iter = iter(gold_loader)
                data_g, target_g,target_g_align = next(gold_loader_iter)


            data_g, target_g,target_g_align = (data_g).to(args.device), (target_g).to(args.device),(target_g_align).to(args.device)
            data_s, target_s_,target_s_align = (data_s).to(args.device), (target_s).to(args.device),(target_s_align).to(args.device)

            # bi-level optimization stage
            eta = main_schdlr.get_lr()[0]#从主模型的学习率调度器中提取当前学习率，传递给训练函数。
            if args.method == 'hmlc_K':#hmlc_K：直接在噪声数据和干净数据上进行校正和优化。
                loss_g, loss_s,logit_g,logit_s,pseudo_target_s= step_hmlc_K(main_net, main_opt, loss_fn,
                                             meta_net, optimizer, loss_fn,
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
                                             meta_net, optimizer, loss_fn,
                                             data_s, target_s_, data_g, target_g,
                                             data_c, target_c,
                                             eta, args)
                

            # 	loss_g：干净样本的损失（用于元优化）。   
            #   loss_s：带噪声样本的损失（用于主模型训练）。
            train_loss_s+=loss_s
            train_loss_g+=loss_g
            args.steps += 1
            if i % args.every == 0:
                
                # ''' get entropy of predictions from meta-net '''
                # # entropy用于衡量meta_net修正标签的不确定性
                # logit_s  = main_net(data_s )
                # pseudo_target_s = meta_net(logit_s.detach(), target_s_).detach()
                # entropy =loss_fn(logit_s,pseudo_target_s)  


                main_lr = main_schdlr.get_lr()[0]
                meta_lr = scheduler.get_lr()[0]

            
                print('\tIteration %d\tMain LR: %.8f\tMeta LR: %.8f\tloss_s: %.4f\tloss_g: %.4f ' %( i, main_lr, meta_lr, loss_s.item(), loss_g.item(), ))

        # PER EPOCH PROCESSING

        # lr scheduler
        main_schdlr.step() 

        
        train_loss_s /= len(silver_loader) 
        train_loss_g /= len(gold_loader)

               
        #scheduler.step()

        # evaluation on validation set
        val_loss = test(main_net, valid_loader,epoch,'val') 
        # test_loss = test(main_net, test_loader,epoch,'test')


        wandb.log({"epoch": epoch, 
                   "train_loss": train_loss_s, 
                   "train_loss_meta": train_loss_g, 
                   "val_loss":val_loss,
                #    'test_loss':test_loss,
                   })
        if epoch%20==0:
            if args.meta_method not in ['pseudolabel','phase_amplitude_align']:
                pseudo_target_s =None
            plot_compare_sig(target_s_,logit_s, input=data_s,title='train',pseudo_labels=pseudo_target_s,labels_align=target_s_align)

 

 
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