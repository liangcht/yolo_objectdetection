import torch       
def get_optim_params(model_parameters, initial_lr):
    decay, no_decay, lr2 = [], [], []

    for name, param in model_parameters:
        if not param.requires_grad:
            continue
        if "last_conv" in name and name.endswith(".bias"):
            lr2.append(param)
            print(name, torch.typename(param))
        elif "scale" in name:
            decay.append(param)
            print(name, torch.typename(param))

        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
            print(name, torch.typename(param))
        else:
            decay.append(param)
            print(name, torch.typename(param))

    return [{'params': no_decay, 'weight_decay': 0., 'initial_lr': initial_lr, 'lr_mult': 1.},
            {'params': decay, 'initial_lr': initial_lr, 'lr_mult': 1.},
            {'params': lr2, 'weight_decay': 0., 'initial_lr': initial_lr * 2., 'lr_mult': 2. , 'lr': initial_lr * 2.}]


def get_lrs(solver):
    lr_policy = solver.get('lr_policy', 'fixed')
    
    if not lr_policy or lr_policy == 'fixed':
        return [float(solver['base_lr'])]
    
    if lr_policy == "multifixed":
        return [float(lr) for lr in solver["stagelr"]]
    
    raise NotImplementedError("Learning policy: {} not implemented".format(lr_policy))


def get_steps(solver):
    lr_policy = solver.get('lr_policy', 'fixed')

    if not lr_policy or lr_policy == 'fixed':
        return None
  
    if lr_policy == "multifixed":
        return [int(ii) for ii in solver["stageiter"]]

    raise NotImplementedError("Learning policy: {} not implemented".format(lr_policy))
    



