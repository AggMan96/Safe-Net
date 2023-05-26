import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_pytorch import vit_small_patch16_224_backbone

### Initialization Function
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


### Classifier Module
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)

        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x,f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


### Localization Net
class Loc_Net(nn.Module):
    def __init__(self):
        super(Loc_Net, self).__init__()

        self.fc_loc = nn.Sequential(
            nn.Linear(in_features=768, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=3*2),
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x):
        A_theta = self.fc_loc(x)
        A_theta = A_theta.view(-1, 2, 3)

        return A_theta


### Self-Adaptive Feature Extraction Network (Safe-Net)
class Safe_Net_model(nn.Module):
    def __init__(self, num_classes, block = 4 ,return_f=False, imgsize=256):
        super(Safe_Net_model, self).__init__()
        
        self.block = block
        self.return_f = return_f
        self.num_classes = num_classes

        ## backbone -> Vit-S
        transformer_name = "vit_small_patch16_224_backbone"
        print('using Transformer_type: {} as a backbone'.format(transformer_name))
        self.transformer = vit_small_patch16_224_backbone(img_size = (imgsize, imgsize), stride_size = [16, 16], drop_path_rate = 0.1,
                                                            drop_rate = 0.0, attn_drop_rate = 0.0)

        ## classifier
        self.global_classifier = ClassBlock(768, num_classes, 0.5, return_f = return_f)
        for i in range(self.block):
            name = 'part_classifier' + str(i+1)
            setattr(self, name, ClassBlock(768, num_classes, 0.5, return_f = self.return_f))
        
        ## localization net
        self.loc_net = Loc_Net()

    def forward(self, x):
        
        features, all_features = self.transformer(x) # vit backbone to extract features
        
        # batchsize=8 block=3 imagesize=256*256 patchsize=16*16 patchnum=16*16=256 feat_dim=768
        global_feature = features[:,0]
        patch_features = features[:,1:]  # shape:[8, 256, 768]
        
        # classifier for global feature
        tranformer_feature = self.global_classifier(global_feature)
        if self.block==1:
            return tranformer_feature        

        # FAM
        aligned_feature_map = self.feat_alignment(global_feature, patch_features)  
        # FPM
        part_features = self.feat_partition(aligned_feature_map, pooling='avg')     

        # classifier for part features
        y = self.part_classifier(self.block, part_features, cls_name='part_classifier')   # shape:list([8,701]*3)

        if self.training:
            y = y + [tranformer_feature]
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])    # i[0] ->  classification result
                    features.append(i[1])   # i[1] -> feature
                return cls, features
        else:
            tranformer_feature = tranformer_feature.view(tranformer_feature.size(0),-1,1)
            y = torch.cat([y,tranformer_feature],dim=2)

        return y

    def feat_alignment(self, global_feature, patch_features):
        ''' Feature Alignment Module (FAM)'''

        ## reshape patch features to patch feature map
        B, N, C = patch_features.size(0), patch_features.size(1), patch_features.size(2)
        H = W = int(N**0.5)
        patch_features = patch_features.view(B, H, W, C)
        patch_features = patch_features.permute(0, 3, 1, 2)
        
        ## calculate A_theta for affine transformation 
        A_theta = self.loc_net(global_feature)

        ## generate sampling grid
        grid = F.affine_grid(A_theta, patch_features.size(), align_corners=True)
        grid = grid.float()

        ## sampling
        aligned_feature_map = F.grid_sample(patch_features, grid, mode='nearest', padding_mode="border", align_corners=True)        

        return aligned_feature_map

    def feat_partition(self, aligned_feature_map, pooling='max'):
        ''' Feature Partition Module (FPM)'''

        part_features = self.partition(aligned_feature_map, pool=pooling)
        part_features = part_features.squeeze(-1)

        return part_features

    def partition(self, x, pool='avg', no_overlap=True):
        result = []

        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 

        ## calculate the center coordinate of the input feature map
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        
        ## obtain the boundaries for feature partition
        boundary = self.get_boundary(x)   # [batchsize, block]

        ### divide the input feature map into #block part features 
        for i in range(self.block):
            block_result = []
            for b in range(x.size(0)):
                x_curr = x[b, :, (c_h-boundary[b][i]):(c_h+boundary[b][i]), (c_w-boundary[b][i]):(c_w+boundary[b][i])]
                if no_overlap and i >= 1:
                    x_pre = x[b, :, (c_h-boundary[b][i-1]):(c_h+boundary[b][i-1]), (c_w-boundary[b][i-1]):(c_w+boundary[b][i-1])] 
                    x_pad = F.pad(x_pre, (boundary[b][i]-boundary[b][i-1], boundary[b][i]-boundary[b][i-1], boundary[b][i]-boundary[b][i-1], boundary[b][i]-boundary[b][i-1]), "constant", 0)
                    x_curr = x_curr - x_pad     # [2048, h, w]
                
                avgpool = pooling(x_curr)       # [2048, 1, 1]
                block_result.append(avgpool)     # [8, 2048, 1, 1]
            x_block = torch.stack(block_result, dim=0)  # [8, 2048, 1, 1]
            result.append(x_block)

        return torch.cat(result, dim=2)

    def get_boundary(self, x):
        
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
 
        pooling = torch.nn.AdaptiveAvgPool2d((1,1)) # GAP
        ring_block = int(H/2)    # number of square-ring blocks with ring width of 1
        no_overlap = True

        n_div = int(math.log(self.block, 2))    # Number of divisions: if block=4, n_div=2.

        ## aggregated feature map
        agg_feat, _ = torch.max(input=x, dim=1, keepdim=False)

        ### slice the aggregated feature map into #ring_block square-ring feature maps and get saliency values by GAP
        value_result = []
        for i in range(ring_block):
            i = i + 1
            agg_curr = agg_feat[:, (c_h-i):(c_h+i), (c_w-i):(c_w+i)]
            if no_overlap and i > 1:
                agg_pre = agg_feat[:, (c_h-(i-1)):(c_h+(i-1)), (c_w-(i-1)):(c_w+(i-1))] 
                agg_pad = F.pad(agg_pre, (1,1,1,1), "constant", 0)
                agg_curr = agg_curr - agg_pad

            value = pooling(agg_curr) # GAP
            value_result.append(value)

        ## saliency value
        saliency_values = torch.cat(value_result, dim=1).squeeze(-1) # [batchsize, block] -> [8, 8]
        
        ## obtain boundaries based on saliency values by recursion 
        boundaries = []
        for n in range(saliency_values.size(0)):
            boundary = self.get_boundary_recur(saliency_values[n], 0, len(saliency_values[n])-1, n_div) + [len(saliency_values[n])]
            boundaries.append(boundary)
 
        assert len(boundaries) == x.size(0)
        assert len(boundaries[0]) == self.block

        return boundaries
    
    def get_boundary_recur(self, sa_value, center, margin, n):
        '''
            Input:
                sa_value:   Saliency values for division
                center  :   Center index
                margin  :   Margin index
                n       :   Number of divisions

            Output:
                boundary
        '''
        
        boundaries = []

        ## end recursion
        if n == 0:
            return []
        
        ## get center value and margin value from saliency values
        center_value = sa_value[center]
        margin_value = sa_value[margin]
        comp_value = sa_value[center: margin+1] # used for calculating the boundaries

        ## calculate the boundary index
        boundary = torch.argmin(abs(abs(comp_value - center_value) - abs(comp_value - margin_value)))
        boundary = int(center + boundary)

        ## avoid boundary crossing and ensure subsequent divisibility
        if boundary < center + 2**(n-1):
            boundary = center + 2**(n-1)
        if boundary > margin - 2**(n-1) + 1:
            boundary = margin - 2**(n-1) + 1

        ## recursive calculating boundaries
        boundaries = self.get_boundary_recur(sa_value, center, boundary-1, n-1) # region A -> [center: center of region A; boundary-1: margin of region A]
        boundaries = boundaries + [boundary]
        boundaries = boundaries + self.get_boundary_recur(sa_value, boundary, margin, n-1) # add region B -> [boundary: center of region B; margin: margin of region B]

        return boundaries

    def part_classifier(self, block, x, cls_name='classifier'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            # return torch.cat(y,dim=1)
            return torch.stack(y, dim=2)
        return y

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
