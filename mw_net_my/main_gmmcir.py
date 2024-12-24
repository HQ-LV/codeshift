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
from backbone.metanet import MetaNet,MetaNetLoss,MetaNetPseudoLabel,PhaseAmplitudeAlignNet,PhaseAmplitudeAlignNetNorm,MetaNet_TimeShift2PhaseShift
# from mlp import simpleMLP
from data.utils import *
import wandb
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# from dividemix.data import DropSignalModified_dataloader
from dividemix.dataShift import ShiftSignalModified_dataloader
from backbone.util import batch_time_shift_signal

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

parser.add_argument('--dataset', type=str, default='ShiftSignalModified')
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

parser.add_argument('--meta_method',type=str,choices=['time_shift','reweight','pseudolabel','phase_amplitude_align','timeshift_phaseshift'])
parser.add_argument('--meta_norm',action='store_true')
parser.add_argument('--smoothing_and_amplify',action='store_true')
parser.add_argument('--p_threshold',type=float,default=0.7)
parser.add_argument('--keep_good_subject',type=float,default=0.6)
parser.add_argument('--history_loss_eval_train',action='store_true')
 
parser.add_argument('--shift',type=str,default='shifts',choices=['shifts','all','center'])
parser.add_argument('--max_drop_rate',default=0.05,type=float)


args = parser.parse_args()
print(args)



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
    print(losses.shape,'minloss',losses.min(),'maxloss',losses.max())
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)


    # 保存loss到本地
    # torch.save(losses, f'out/losses{args.prob_shift}.pth')
    # print('save losses.pth')

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


    return prob,all_loss


def meta_weight_net():

    if 'Shift' in args.dataset:  project="Shift"
    elif 'Drop' in args.dataset: project="Drop"
    if 'vitaldb' in args.data_dir: project+="-VitalDB"
    elif 'mimic' in args.data_dir: project+="-MIMIC"

    run = wandb.init(project=project, 
                     config=args,
                     name = f'meta-gmmcir-{args.prob_shift}-{args.num_meta}')  # 初始化wandb
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
    elif args.meta_method =='timeshift_phaseshift':
        meta_net = MetaNet_TimeShift2PhaseShift().to(device=args.device)
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
        )
    _, meta_dataloader, val_dataloader ,test_dataloader= build_dataloader(
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
        no_train_noisy=True,
)

    meta_dataloader_iter = iter(meta_dataloader)
#     with torch.no_grad():
#         for point in range(500):
#             x = torch.tensor(point / 10).unsqueeze(0).to(args.device)
#             fx = meta_net(x)
#             writer.add_scalar('Initial Meta Net', fx, point)

    best_val_loss = float('inf')




    all_loss = [[],[]] 

    for epoch in range(args.max_epoch):

        if epoch >= 40 and epoch % 20 == 0:
            lr = lr / 5
        for group in optimizer.param_groups:
            group['lr'] = lr

        # print('Training...')
        train_loss_unlabeled = []
        train_loss_meta =[]

        if epoch%5==0:   
            # 先计算训练集的损失，用GMM划分噪声样本和干净样本
            get_percentile = lambda e, maxe : 70 - (e / maxe) * (70 - 10) # 阈值从80到10，也就是认为干净样本是选了30%-90%

            eval_train_loader = train_SSMloader.run('eval_train')   
            prob,all_loss[0]=eval_train(net,all_loss[0],eval_train_loader)  
            # pred = (prob > np.percentile(prob, args.prob_shift*100)) 
            
            percentile = get_percentile(epoch, args.max_epoch) 
            pred = (prob > np.percentile(prob, percentile))

            labeled_trainloader, unlabeled_trainloader = train_SSMloader.run('train',pred,prob)
        
        #ppg,ppg_noise,bp,bp_align, probability ,idx
        # for iteration, (inputs, labels,labels_align) in enumerate(train_dataloader):
         
        # for iteration, (inputs, _,labels,labels_align,w_x,_) in enumerate(unlabeled_trainloader):

        
        for iteration, (inputs, _,labels,labels_align,w_x,_) in enumerate(labeled_trainloader):
            net.train()
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
                elif args.meta_method =='timeshift_phaseshift':
                    pseudo_outputs = pseudo_net(inputs) 
                    pseudo_timeshift = meta_net(pseudo_outputs.detach(),labels)
                    pseudo_label= batch_time_shift_signal(labels, pseudo_timeshift)
                    pseudo_loss = loss_fn(pseudo_outputs,pseudo_label)

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
                    # inputs, _,labels,labels_align,w_x,_
                    meta_inputs, _,meta_labels,meta_labels_align,_,_ = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    meta_inputs, _,meta_labels,meta_labels_align,_,_ = next(meta_dataloader_iter)
                meta_inputs, meta_labels,meta_labels_align = meta_inputs.to(args.device), meta_labels.to(args.device),meta_labels_align.to(args.device)
                # meta_labels = torch.randn((inputs.shape[0],2), device=args.device)
            
                meta_outputs = pseudo_net(meta_inputs)
                meta_loss = loss_fn(meta_outputs, meta_labels)

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

                train_loss_meta.append(meta_loss.item())
            
             
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

            train_loss_unlabeled.append(loss.item())

 
        train_loss_unlabeled = torch.mean(torch.tensor(train_loss_unlabeled))
        train_loss_meta = torch.mean(torch.tensor(train_loss_meta))
        wandb.log({"epoch": epoch, "train_loss_unlabeled": train_loss_unlabeled, "lr": lr,"train_loss_meta":train_loss_meta})
        if epoch%20==0 or epoch ==args.max_epoch-1:
            if args.meta_method not in ['pseudolabel','phase_amplitude_align','timeshift_phaseshift']:
                pseudo_labels =None 
            plot_compare_sig(labels,outputs, input=inputs,title='train-unlabeled',pseudo_labels=pseudo_labels,labels_align=labels_align)

            plot_compare_sig(meta_labels,meta_outputs, input=meta_inputs,title='metaset',labels_align=meta_labels_align)


        # train_loss_labeled = []
        # for batch_idx, (inputs, _,labels,labels_align,w_x,_) in enumerate(labeled_trainloader):
        #     net.train()
        #     w_x = w_x.to(args.device)
        #     # w_x = w_x.view(-1,1).unsqueeze(-1) 
        #     inputs, labels,labels_align = inputs.to(args.device), labels.to(args.device),labels_align.to(args.device)
        #     outputs = net(inputs)
        #     # loss = loss_fn(outputs, labels,reduction='none')
        #     # loss = loss.mean(dim=(1, 2)) 
        #     # loss = torch.mean(loss*w_x)
        #     loss = loss_fn(outputs, labels ) 

        #     if epoch>0:
        #         # print(loss)
        #         # 打印参数和梯度的函数
        #         nan_yes = False
        #         if torch.isnan(inputs).any():
        #             print("inputs contains NaN")
        #             nan_yes = True
        #         if torch.isnan(outputs).any():
        #             print("outputs contains NaN")
        #             nan_yes = True
        #         if torch.isnan(labels).any():
        #             print("labels contains NaN")
        #             nan_yes = True
        #         if torch.isnan(w_x).any():
        #             print("w_x contains NaN") 
        #             nan_yes = True
        #         if torch.isnan(loss).any():
        #             print("loss contains NaN")
        #             nan_yes = True
        #         if nan_yes:
        #             for nnnp,(name, param) in enumerate(net.named_parameters()):
        #                 if nnnp==2:
        #                     print(f"参数: {name}")
        #                     print(f"值: {param.data}")  # 打印参数的值
        #                     if param.grad is not None:
        #                         print(f"梯度: {param.grad}")  # 打印参数的梯度
        #                     else:
        #                         print("没有梯度")
        #                     if torch.isnan(param).any():
        #                         print("param contains NaN")
        #                         sys.exit()
        #                     print("="*40)
                    

  

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     train_loss_labeled.append(loss.item())

        # train_loss_labeled = torch.mean(torch.tensor(train_loss_labeled))
        # wandb.log({"epoch": epoch, "train_loss_labeled": train_loss_labeled})
        # if epoch%20==0 or epoch ==args.max_epoch-1:
        #     if args.meta_method not in ['pseudolabel','phase_amplitude_align']:
        #         pseudo_labels =None 
        #     plot_compare_sig(labels,outputs, input=inputs,title='train-labeled',labels_align=labels_align)


 
        # print('Computing Test Result...')
        val_loss,val_labels,val_outputs,val_inputs = compute_loss(
            net=net,
            data_loader=val_dataloader,
            criterion=loss_fn,
            device=args.device,
        )
        # print(f"Epoch: {epoch}, LR: {lr:.6f}, || Train Loss Unlabeled: {train_loss_unlabeled:.4f},Loss Labeled: {train_loss_labeled:.4f}, loss_meta:{train_loss_meta:.4f} ||  Val Loss: {val_loss:.4f} ")
        print(f"Epoch: {epoch}, LR: {lr:.6f}, || Train Loss Unlabeled: {train_loss_unlabeled:.4f}, loss_meta:{train_loss_meta:.4f} ||  Val Loss: {val_loss:.4f} ")
        
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


def loss_fn(outputs, labels,  reduction='mean' ):
 
    return F.mse_loss(outputs, labels,reduction=reduction) 
    

    
     



if __name__ == '__main__':
    meta_weight_net()