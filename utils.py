import torch
import torch.nn as nn
from torch import Tensor
from models.densedepth import DenseDepth
from models.depth_anything_v2.dpt import DepthAnythingV2

def DepthNorm(depth:Tensor, maxDepth:float=1000.0) -> Tensor:
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val:float, n:int=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_or_load_model(depthmodel:str, enc_pretrain:bool, lr:float, ckpt:str|None=None, device:torch.device=torch.device('cuda:0')) -> tuple[nn.Module, torch.optim.Adam, torch.optim.lr_scheduler.StepLR, int]:

    if ckpt is not None:
        checkpoint = torch.load(ckpt)

    if 'densedepth' in depthmodel.lower().strip():
        model = DenseDepth(encoder_pretrained=enc_pretrain)
    elif 'depthanythingv2' in depthmodel.lower().strip():
        model = DepthAnythingV2(encoder='vitb',features=128,out_channels=[96,192,384,768], max_depth=100)
        if enc_pretrain:
            vitb_ckpt = R'models\depth_anything_v2\depth_anything_v2_vitb.pth'
            model.load_state_dict({k: v for k, v in torch.load(vitb_ckpt, map_location='cpu').items() if 'pretrained' in k}, strict=False)

    if ckpt is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if ckpt is not None:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1 if ckpt is not None else 0

    return model, optimizer, scheduler, start_epoch
