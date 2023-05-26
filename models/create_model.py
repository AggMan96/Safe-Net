import torch.nn as nn
from .model import Safe_Net_model

class Safe_Net(nn.Module):
    def __init__(self, class_num, block=4, return_f=False, imgsize=256):
        super(Safe_Net, self).__init__()
        self.loc_model = Safe_Net_model(num_classes=class_num, block=block, return_f=return_f, imgsize=imgsize)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            y1 = self.loc_model(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.loc_model(x2)

        return y1, y2
        

def create_model(opt):
    model_path = "./models/pretrain_backbone/vit_small_p16_224-15ec54c9.pth"

    # create Safe-Net
    model = Safe_Net(class_num=opt.nclasses, block=opt.block, return_f=opt.triplet_loss, imgsize=opt.h)
    # load pretrain param
    model.loc_model.transformer.load_param(model_path)

    return model
