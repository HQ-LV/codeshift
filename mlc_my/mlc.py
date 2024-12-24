import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.util import batch_time_shift_signal

def _concat(xs):#将多个张量展开为一维后拼接成一个大张量。
    return torch.cat([x.view(-1) for x in xs])

@torch.no_grad()#根据不同优化器类型选择参数更新策略。
def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')

# be aware that the opt state dict returns references, hence take care not to
# modify them
#实现带有动量和权重衰减的 SGD 参数更新。
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov'] #支持 Nesterov 动量模式。

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay # s=1
        s = 1

        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. -dampening) # s=1.-dampening

            if nesterov:
                dparam = dparam + momentum * moment # s= 1+momentum*(1.-dampening)
                s = 1 + momentum*(1.-dampening)
            else:
                dparam = moment # s=1.-dampening
                s = 1.-dampening

        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam  * eta)

        if return_s:
            ss.append(s*eta)

    if return_s:
        return ans, ss
    else:
        return ans


# ============== mlc step procedure debug with features (gradient-stopped) from main model ===========
#
# METANET uses the last K-1 steps from main model and imagine one additional step ahead
# to compose a pool of actual K steps from the main model
#
#
def step_hmlc_K(main_net, main_opt, hard_loss_f,
                meta_net, meta_opt, soft_loss_f,
                data_s, target_s, data_g, target_g,
                data_c, target_c, 
                eta, args,meta_loss2=False):

    # compute gw for updating meta_net
    # •	对干净数据计算主模型的 hard损失 loss_g。
	# •	通过 torch.autograd.grad 计算主模型参数的梯度 gw。
    # compute d_w L_{D}(w)
    logit_g = main_net(data_g)
    loss_g = hard_loss_f(logit_g, target_g)
    gw = torch.autograd.grad(loss_g, main_net.parameters())
    
    # given current meta net, get corrected label
    # •	使用标签校正网络（MetaNet）生成带噪声数据的校正标签 pseudo_target_s。
	# •	基于校正标签计算主模型的软标签损失 loss_s。
    logit_s = main_net(data_s)
    if args.meta_method == 'pseudolabel':
        pseudo_target_s = meta_net(logit_s.detach(), target_s) 
        loss_s = soft_loss_f(logit_s, pseudo_target_s,target_s)
    elif args.meta_method == 'timeshift_phaseshift':
        pseudo_timeshift = meta_net(logit_s.detach(), target_s)
        pseudo_target_s = batch_time_shift_signal(target_s ,pseudo_timeshift )
        loss_s = soft_loss_f(logit_s, pseudo_target_s)
    

    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)

        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2 ) / (bs1+bs2)

    # •	对 噪声数据矫正后的 loss_s 求梯度，得到主模型参数的梯度 f_param_grads。
    # compute d_w L_{D'}(w)
    f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)    
    # •	使用 update_params 函数，更新主模型参数  w \to w‘ 。
    f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True)
    # 2. set w as w'
    f_param = []
    for i, param in enumerate(main_net.parameters()):
        f_param.append(param.data.clone())
        param.data = f_params_new[i].data # use data only as f_params_new has graph
    
    # training loss Hessian approximation
    Hw = 1 # assume to be identity 

    # 3. compute d_w' L_{D}(w')
    # •	重新计算 w’ 下的梯度 gw_prime。
    logit_g = main_net(data_g)
    loss_g  = hard_loss_f(logit_g, target_g)
    gw_prime = torch.autograd.grad(loss_g, main_net.parameters())

    # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2
    # •	使用近似 Hessian 矩阵 Hw = I，计算梯度校正因子 gamma。
    tmp1 = [(1-Hw*dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
    gw_norm2 = (_concat(gw).norm())**2
    tmp2 = [gw[i]/gw_norm2 for i in range(len(gw))]
    gamma = torch.dot(_concat(tmp1), _concat(tmp2))

    # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
    Lgw_prime = [ dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]     
    #使用代理梯度 proxy_g 反向传播，更新 MetaNet 的参数
    proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))

    # back prop on alphas
    meta_opt.zero_grad()
    proxy_g.backward()
    
    # accumulate discounted iterative gradient 
    for i, param in enumerate(meta_net.parameters()):
        if param.grad is not None:
            param.grad.add_(gamma * args.dw_prev[i])
            args.dw_prev[i] = param.grad.clone()

    if (args.steps+1) % (args.gradient_steps)==0: # T steps proceeded by main_net
        meta_opt.step()
        args.dw_prev = [0 for param in meta_net.parameters()] # 0 to reset 

    # modify to w, and then do actual update main_net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
        param.grad = f_param_grads[i].data
    main_opt.step()
    
    return loss_g, loss_s,logit_g,logit_s,pseudo_target_s

