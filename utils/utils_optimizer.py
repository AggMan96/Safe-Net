import torch.optim as optim
from torch.optim import lr_scheduler


### Optimizer
def make_optimizer(model, opt):
    i = model.loc_model

    ignored_params = []
    ignored_params += list(map(id, i.transformer.parameters())) # vit backbone
    locnet_params = list(map(id, i.loc_net.parameters())) # loc net

    fam_params = filter(lambda p: id(p) in locnet_params, model.parameters()) # parameters of feature alignment module(fam)
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters()) # parameters of vit backbone
    extra_params = filter(lambda p: id(p) not in ignored_params+locnet_params, model.parameters()) # parameters of other modules (e.g. classifier)
    
    optimizer_ft = optim.SGD([
        {'params': fam_params, 'lr': 0.01*opt.lr},
        {'params': base_params, 'lr': 0.3 * opt.lr},
        {'params': extra_params, 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=opt.steps, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=4, verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

    return optimizer_ft, exp_lr_scheduler