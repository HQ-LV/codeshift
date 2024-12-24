import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from mw_net_my.meta import *
from backbone.swin_transformer_recons_fan import SwinTransformerReconsFan
from backbone.metanet import MetaNet,MetaNetLoss,MetaNetPseudoLabel,PhaseAmplitudeAlignNet,PhaseAmplitudeAlignNetNorm
# from mlp import simpleMLP
from data.utils import *
import wandb
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Meta_Weight_Net')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)

parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=1e-5)
parser.add_argument('--meta_weight_decay', type=float, default=0.)

parser.add_argument('--dataset', type=str, default='ShiftSignal')
parser.add_argument('--data_dir',type=str,default='/data/home/qian_hong/signal/vitaldb/0')
parser.add_argument('--num_meta', type=int, default=500) 
parser.add_argument('--max_shift', type=int, default=50)
parser.add_argument('--prob_shift',type=float,default=0.4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=64)


parser.add_argument('--max_epoch', type=int, default=100)

parser.add_argument('--meta_interval', type=int, default=3)
parser.add_argument('--paint_interval', type=int, default=20)


parser.add_argument('--task',type=str,default='reconstruction')
parser.add_argument('--ckpt_path',type=str)

parser.add_argument('--meta_method',type=str,choices=['time_shift','reweight','pseudolabel','phase_amplitude_align'])
parser.add_argument('--meta_norm',action='store_true')
parser.add_argument('--smoothing_and_amplify',action='store_true')

args = parser.parse_args()
print(args)


def meta_weight_net():

    run = wandb.init(project="Shift-VitalDB", config=args,
                     name = f'meta-{args.prob_shift}-{args.num_meta}')  # 初始化wandb
    print(f"Current run name: {run.name}")

    best_model_path = f"ckpt/{run.id}.pth"
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
#     writer = SummaryWriter(log_dir='.\\mwn')

    if args.meta_method =='time_shift':
        meta_net = MetaNet().to(device=args.device)
    elif args.meta_method =='reweight':
        meta_net = MetaNetLoss().to(device=args.device)
    elif args.meta_method =='pseudolabel':
        meta_net = MetaNetPseudoLabel(smoothing_and_amplify=args.smoothing_and_amplify).to(device=args.device)
    elif args.meta_method =='phase_amplitude_align':
        if args.meta_norm:
            meta_net = PhaseAmplitudeAlignNetNorm().to(device=args.device)
        else:
            meta_net = PhaseAmplitudeAlignNet().to(device=args.device)
    net = SwinTransformerReconsFan(args.task).to(device=args.device)
    if args.ckpt_path:
        print('load pth',args.ckpt_path)
        net.load_state_dict(torch.load(args.ckpt_path))
 

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    lr = args.lr

    train_dataloader, meta_dataloader, val_dataloader ,test_dataloader= build_dataloader(
        data_dir = args.data_dir,
        keep_good_subject=0.6,
        shift='shifts' ,
        metaset_len = args.num_meta,
        prob_shift = args.prob_shift,
        max_shift =args.max_shift,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
)

    meta_dataloader_iter = iter(meta_dataloader)
#     with torch.no_grad():
#         for point in range(500):
#             x = torch.tensor(point / 10).unsqueeze(0).to(args.device)
#             fx = meta_net(x)
#             writer.add_scalar('Initial Meta Net', fx, point)

    best_val_loss = float('inf')
    for epoch in range(args.max_epoch):

        if epoch >= 60 and epoch % 20 == 0:
            lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        # print('Training...')
        train_loss = 0
        train_loss_meta =[]
        for iteration, (inputs, labels,labels_align) in enumerate(train_dataloader):
            net.train()
        
            # labels = torch.randint(0, 2, (inputs.shape[0],), device=args.device)
            # labels = torch.randn((inputs.shape[0],2), device=args.device)
            
            inputs, labels,labels_align = inputs.to(args.device), labels.to(args.device),labels_align.to(args.device)
            # if iteration==9:
            # print(iteration,inputs.shape,labels.shape)
            
            if (iteration + 1) % args.meta_interval == 0:
                pseudo_net = SwinTransformerReconsFan(args.task).to(args.device)
                # pseudo_net = simpleMLP().to(args.device)
                
                pseudo_net.load_state_dict(net.state_dict())
                pseudo_net.train()

                if args.meta_method =='time_shift':
                    pseudo_outputs = pseudo_net(inputs) 
                    if torch.isnan(pseudo_outputs).any():
                        print("pseudo_outputs contains NaN")
                    pseudo_shift = meta_net(pseudo_outputs.detach(),labels,)
                    if epoch%20==0 and iteration%50==0:
                        print(pseudo_shift[:15])
                    pseudo_loss = loss_fn(pseudo_outputs,labels,pseudo_shift)
                elif args.meta_method =='reweight':
                    pseudo_outputs = pseudo_net(inputs)
                    pseudo_loss_vector = loss_fn(pseudo_outputs, labels, reduction='none')
                    pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
                    pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)# meta_net输入伪loss，输出权重
                    pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)

                elif args.meta_method in ['pseudolabel','phase_amplitude_align']:
                    pseudo_outputs = pseudo_net(inputs) 
                    pseudo_labels = meta_net(pseudo_outputs.detach(),labels) 
                    pseudo_loss = loss_fn(pseudo_outputs,pseudo_labels) 

                if epoch in [5,15] and (iteration + 1) % (args.meta_interval*100) == 0:
                    print('save',f'pseudo_data_{epoch}_{iteration}.pth')
                    # 将这三个值保存到本地
                    torch.save({
                        'pseudo_outputs': pseudo_outputs,
                        'labels': labels,
                        'pseudo_labels': pseudo_labels
                    }, f'out/pseudo_data_{epoch}_{iteration}.pth')

                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True )
                
                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)
 

                del pseudo_grads

                # 元数据集，pseudo_net预测，计算元数据集的损失，损失用来更新meta-weight-net
                try:
                    meta_inputs, meta_labels,meta_labels_align = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    meta_inputs, meta_labels,meta_labels_align = next(meta_dataloader_iter)

                meta_inputs, meta_labels,meta_labels_align = meta_inputs.to(args.device), meta_labels.to(args.device),meta_labels_align.to(args.device)
                # meta_labels = torch.randn((inputs.shape[0],2), device=args.device)
            
                meta_outputs = pseudo_net(meta_inputs)
                meta_loss = loss_fn(meta_outputs, meta_labels)

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

                train_loss_meta.append(meta_loss.item())
            
            
                
            # pseudo_outputs = pseudo_net(inputs)
            # pseudo_shift = meta_net(pseudo_outputs.data,labels)
            # pseudo_loss = loss_fn(pseudo_outputs,labels,pseudo_shift)
            # pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

            outputs = net(inputs)
            if args.meta_method =='time_shift':
                with torch.no_grad():
                    shift = meta_net(outputs,labels)
                loss = loss_fn(outputs, labels ,shift) 
  
            elif args.meta_method =='reweight':
                loss_vector = loss_fn(outputs, labels, reduction='none')
                loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
                with torch.no_grad(): 
                    weight = meta_net(loss_vector_reshape)
                loss = torch.mean(weight * loss_vector_reshape)
            elif args.meta_method in ['pseudolabel','phase_amplitude_align']:
           
                pseudo_labels = meta_net(outputs ,labels) 
                loss = loss_fn(outputs,pseudo_labels) 
                    

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(meta_dataloader) 

        train_loss_meta = torch.mean(torch.tensor(train_loss_meta))
   


        wandb.log({"epoch": epoch, "train_loss": train_loss, "lr": lr,"train_loss_meta":train_loss_meta})
        if epoch%20==0:
            if args.meta_method not in ['pseudolabel','phase_amplitude_align']:
                pseudo_labels =None 
            plot_compare_sig(labels,outputs, input=inputs,title='train',pseudo_labels=pseudo_labels,labels_align=labels_align)


        # print('Computing Test Result...')
        val_loss,val_labels,val_outputs,val_inputs = compute_loss(
            net=net,
            data_loader=val_dataloader,
            criterion=loss_fn,
            device=args.device,
        )
        print(f"Epoch: {epoch}, LR: {lr:.6f}, Train Loss: {train_loss:.4f}, train_loss_meta:{train_loss_meta:.4f} Val Loss: {val_loss:.4f} ")
        wandb.log({"epoch": epoch, "val_loss": val_loss})

        if epoch%20==0 or epoch ==args.max_epoch-1:
            plot_compare_sig(val_labels,val_outputs,input=val_inputs,title='val')

        # 保存表现最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch} with val_loss {val_loss:.4f}")
    
    # print('Computing Test Result...')
    test_loss,test_labels,test_outputs,test_inputs = compute_loss(
        net=net,
        data_loader=test_dataloader,
        criterion=loss_fn,
        device=args.device,
    )
    print(f"test Loss: {test_loss:.4f} ")
    wandb.log({ "test_loss": test_loss})

  
    plot_compare_sig(test_labels,test_outputs,input=test_inputs,title='test')


def loss_fn(outputs, labels,  reduction='mean'):
    return F.mse_loss(outputs, labels,reduction=reduction)
    
     

def loss_fn_shift(outputs, labels, shift=None,reduction='mean'):
    # 如果没有偏移量，直接计算整个序列的MSE
    if shift is None:
        return F.mse_loss(outputs, labels,reduction=reduction)
    
    batch_size = outputs.shape[0]
    total_loss = 0
   
    # 对每个批次样本分别处理
    for i in range(batch_size):
        curr_shift =  shift[i]   # 获取当前样本的偏移量
        
        if curr_shift > 0:
            # 正偏移：output需要右移
            aligned_output = outputs[i, :, curr_shift:].contiguous()
            aligned_label = labels[i, :, :-curr_shift].contiguous()
        elif curr_shift < 0:
            # 负偏移：output需要左移
            curr_shift = abs(curr_shift)
            aligned_output = outputs[i, :, :-curr_shift].contiguous()
            aligned_label = labels[i, :, curr_shift:].contiguous()
        else:
            # 无偏移
            aligned_output = outputs[i]
            aligned_label = labels[i]
            
        # 计算当前样本的MSE损失
        sample_loss = F.mse_loss(aligned_output, aligned_label,reduction=reduction)
        total_loss += sample_loss
    
    # 返回平均损失
    return total_loss / batch_size


if __name__ == '__main__':
    meta_weight_net()