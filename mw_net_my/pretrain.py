import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from backbone.swin_transformer_recons_fan import SwinTransformerReconsFan 
from data.utils import *


parser = argparse.ArgumentParser(description='Meta_Weight_Net')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--dampening', type=float, default=0.0)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=1e-5)
parser.add_argument('--meta_weight_decay', type=float, default=0.0)

parser.add_argument('--dataset', type=str, default='DropSignal')
parser.add_argument('--data_dir', type=str, default='/data/home/qian_hong/signal/vitaldb/0_drop')
parser.add_argument('--num_meta', type=int, default=2000)
parser.add_argument('--max_shift', type=int, default=50) 
parser.add_argument('--prob_shift',type=float,default=0.7)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--meta_interval', type=int, default=1)
parser.add_argument('--paint_interval', type=int, default=20)

parser.add_argument('--task', type=str, default='reconstruction')
parser.add_argument('--expname',type =str,default='pretrain',choices=['pretrain','baseline'])

parser.add_argument('--lambda_meanvar' ,type = float,default=0.0)
parser.add_argument('--lambda_grad' ,type = float,default=0.0)
parser.add_argument('--lambda_peak' ,type = float,default=0.0)
parser.add_argument('--embed_dim',type=int,default=96)
parser.add_argument('--depths', type=int, nargs='+',default=[2,6])
parser.add_argument('--num_heads', type=int, nargs='+',default=[3,12]) 
parser.add_argument('--mlp_ratio',type=float,default=4.0) 


parser.add_argument('--shift',type=str,default='drop',choices=['drop','align'])
parser.add_argument('--keep_good_subject',type=float,default=0.4)
parser.add_argument('--input_signal',default='ppg',type=str,choices=['ppg','ecg'])
args = parser.parse_args()
print(args)


def meta_weight_net(): 
    if 'Shift' in args.dataset:  project="Shift"
    elif 'Drop' in args.dataset: project="Drop"
    if 'vitaldb' in args.data_dir: project+="-VitalDB"
    elif 'mimic' in args.data_dir: project+="-MIMIC"

    name =f'{args.expname}-{args.prob_shift}-{args.num_meta}'
    if args.lambda_meanvar> 0:name+=f'-miustd{args.lambda_meanvar}'
    if args.lambda_grad> 0:name+=f'-grad{args.lambda_grad}'
    if args.lambda_peak> 0:name+=f'-peak{args.lambda_peak}'


    run = wandb.init(project=project, 
                     config=args,
                     name = name)  # 初始化wandb
    print(f"Current run name: {run.name}")

    best_model_path = f"ckpt/{run.id}.pth"
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
 
    net = SwinTransformerReconsFan(args.task,
                                   embed_dim=args.embed_dim,
                                   depths=args.depths,
                                   num_heads=args.num_heads, 
                                   mlp_ratio=args.mlp_ratio,
                                   ).to(device=args.device)
 

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    lr = args.lr
    
 
    main_dataloader, val_loader ,test_loader= build_dataloader_pretrain(
        expname =args.expname,
        data_dir=args.data_dir,
        keep_good_subject=args.keep_good_subject,
        shift=args.shift,
        metaset_len=args.num_meta,
        prob_shift = args.prob_shift,
        max_shift=args.max_shift,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_signal=args.input_signal
    ) 

    best_val_loss = float('inf')
    

    for epoch in range(args.max_epoch):

        if epoch >= 20 and epoch % 20 == 0:
            lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        # print('Training...')
        train_loss = []
        net.train()
        for iteration, (inputs, labels,labels_align) in enumerate(main_dataloader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = net(inputs)

            if epoch>10:
                lambda_meanvar,lambda_grad,lambda_peak=args.lambda_meanvar,args.lambda_grad,args.lambda_peak
            else:
                lambda_meanvar,lambda_grad,lambda_peak=0.0,0.0,0.0

            loss = loss_fn(outputs, labels,lambda_meanvar=lambda_meanvar,lambda_grad=lambda_grad,lambda_peak=lambda_peak)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
 
        train_loss = torch.mean(torch.tensor(train_loss))
        wandb.log({"epoch": epoch, "train_loss": train_loss, "lr": lr})
        if epoch%20==0:
            plot_compare_sig(labels,outputs,labels_align, input=inputs,title='train')

        # print('Computing Test Result...')
        val_loss,val_labels,val_outputs,val_inputs = compute_loss(
            net=net,
            data_loader=val_loader,
            criterion=loss_fn,
            device=args.device,
        )
        print(f"Epoch: {epoch}, LR: {lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} ")
        wandb.log({"epoch": epoch, "val_loss": val_loss})

        if epoch%20==0:
            plot_compare_sig(val_labels,val_outputs,input=val_inputs,title='val')

        # 保存表现最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch} with val_loss {val_loss:.4f}")

    test_loss,test_labels,test_outputs,test_inputs = compute_loss(
            net=net,
            data_loader=test_loader,
            criterion=loss_fn,
            device=args.device,
        )
    plot_compare_sig(test_labels,test_outputs,input=test_inputs,title='test')

    wandb.log({"epoch": epoch, "test_loss": test_loss})

    print(f"Training complete. Best model saved at {best_model_path}")




def gradient_loss(outputs, labels): 
    # 一阶导数
    grad_outputs = outputs[:,:, 1:] - outputs[:,:, :-1]
    grad_labels = labels[:,:, 1:] - labels[:,:, :-1]
    return F.mse_loss(grad_outputs, grad_labels)
def mean_std_loss(outputs, labels): 
    mean_loss = F.mse_loss(outputs.mean(), labels.mean()) 
    var_loss = F.mse_loss(outputs.std(), labels.std()) 
    return mean_loss , var_loss
def peak_loss(outputs, labels): 
    maxloss = F.mse_loss(outputs.max(dim=2)[0], labels.max(dim=2)[0]) 
    minloss = F.mse_loss(outputs.min(dim=2)[0], labels.min(dim=2)[0]) 
    return maxloss,minloss
# def loss_fn(outputs, labels, reduction = 'mean',lambda_peak=0.):
#     mseloss = F.mse_loss(outputs, labels,reduction=reduction) 
#     return mseloss
def loss_fn(outputs, labels, reduction='mean', lambda_meanvar=0.0,lambda_grad = 0.0,lambda_peak=0.0): 
    mseloss = F.mse_loss(outputs, labels, reduction=reduction)
    total_loss = mseloss
    s = f'mse={mseloss:.4f}'
    if lambda_meanvar > 0:
        mean_loss,std_loss = mean_std_loss(outputs, labels) 
        std_loss/=10.0
        total_loss += lambda_meanvar * (mean_loss+std_loss)
        s+=f' mean={mean_loss:.4f} std={std_loss:.4f}'
    if lambda_grad > 0:
        gradloss = gradient_loss(outputs, labels)
        total_loss += lambda_grad * gradloss
        s+=f' grad={gradloss:.4f}'
    
    if lambda_peak>0:
        maxloss,minloss = peak_loss(outputs, labels)
        total_loss += lambda_peak * (maxloss+minloss)/2
        s+=f' peak={(maxloss+minloss)/2:.4f}'
    if lambda_grad+lambda_meanvar+lambda_peak>0:
        print(s)
    return total_loss

def plot_compare_sig(labels, recons,labels_align=None,logvar=None,input = None,title = 'val'):
    # 随机选择一个样本的索引
    sample_index = random.randint(0, labels.shape[0] - 1)
    
    
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 1, 1)  # 创建第一个子图
    plt.plot(input[sample_index, 0, :].detach().cpu().numpy(), label='Input' )

    plt.title(f'id-in-batch:{sample_index}')

    plt.subplot(2, 1, 2)  # 创建第二个子图
    plt.plot(labels[sample_index, 0, :].detach().cpu().numpy(),'b--', label='True', )
    plt.plot(recons[sample_index, 0, :].detach().cpu().numpy(), 'r--',label='Recons', )
    if labels_align is not None:
        plt.plot(labels_align[sample_index, 0, :].detach().cpu().numpy(),'y--', label='Align', )
    if logvar is not None:
        pass
    plt.legend(loc="lower right")  
    wandb.log({f"True vs Recons ({title})": wandb.Image(plt)}) 
    plt.close()

if __name__ == '__main__':
    meta_weight_net()