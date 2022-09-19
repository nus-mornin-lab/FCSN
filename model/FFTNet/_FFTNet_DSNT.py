"""
* This is for testing the training and testing speed of our model.
* This scipt should not be used for other reasons.
"""

import torch
import numpy as np

from toolz         import *
from toolz.curried import *

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet, vgg16, resnet34, resnet50

try:
    from .ViT import VisionTransformer, CONFIGS as vitConfigs
    from .backbone.drn import drn_d_54 as DResNet54
    from .backbone.drn import drn_c_26 as DResNet26
    from .backbone.drn import drn_a_50 as DResNet50
    from .backbone.drn import drn_d_105 as DResNet105
    from .coordConv import AddCoords
except:
    from ViT import VisionTransformer, CONFIGS as vitConfigs
    from backbone.drn import drn_d_54 as DResNet54
    from backbone.drn import drn_c_26 as DResNet26
    from backbone.drn import drn_a_50 as DResNet50
    from backbone.drn import drn_d_105 as DResNet105
    from coordConv import AddCoords

from scipy.stats import multivariate_normal

    
class NETS(nn.Module):
    '''    
    Note: FFTs used in this branch has norm='ortho'
    '''
    def __init__(self, encoder_name, input_size, learnSize,
                 coord = False,
                 pretrained = False,
                 dev='cuda',
                 cov=0.003):
       
        super(NETS, self).__init__()
       
        blocks = {
            ""
            "AlexNet"    : lambda : nn.Sequential(*alexnet(pretrained=pretrained).features),
            "AlexNet_S"  : lambda : nn.Sequential(*get_children(alexnet(pretrained=pretrained).features)[:-1]),
            "VGGNet"     : lambda : nn.Sequential(*get_children(vgg16(pretrained=pretrained).features)),
            "VGGNet_S"   : lambda : nn.Sequential(*get_children(vgg16(pretrained=pretrained).features)[:-10]),
            "VGGNet_VS"  : lambda : nn.Sequential(*get_children(vgg16(pretrained=pretrained).features)[:-15]),
            "DResNet26"  : lambda : nn.Sequential(*list(DResNet26(pretrained=pretrained).children())[:-2]),
            "DResNet50"  : lambda : nn.Sequential(*list(DResNet50(pretrained=pretrained).children())[:-2]),
            "DResNet54"  : lambda : nn.Sequential(*list(DResNet54(pretrained=pretrained).children())[:-2]),
            "DResNet105" : lambda : nn.Sequential(*list(DResNet105(pretrained=pretrained).children())[:-2]),            
            "ResNet"     : lambda : nn.Sequential(*list(resnet50(pretrained=pretrained).children())[:-2]),
            "Vit"        : lambda : VisionTransformer(vitConfigs['ViT-L_16'], img_size = input_size)}
        
        self.encoder_name = encoder_name
        self.input_size   = input_size
        self.learnSize    = learnSize
        self.coord        = coord
        self.addCoords    = AddCoords()
        self.blocks       = blocks[self.encoder_name]()
        self.cov          = cov
        
        if self.coord:
            if self.encoder_name in ["DResNet105", "DResNet54"]:
                tmp = self.blocks[0][0]
            else :
                tmp = self.blocks[0]
                
            self.blocks = nn.Sequential(nn.Conv2d(5, tmp.out_channels, kernel_size=tmp.kernel_size,
                                        stride=tmp.stride, padding=tmp.padding, dilation=tmp.dilation,
                                        groups=tmp.groups, padding_mode=tmp.padding_mode),
                                        *self.blocks[1:])
        
        c,m,n = self.get_shape()
        
        self.upsample_kernelSize = 12
        self.hm_outSize = (m-1)*self.upsample_kernelSize//2+self.upsample_kernelSize
        self.upsample = torch.nn.ConvTranspose2d(c,learnSize,self.upsample_kernelSize,
                                                  stride=self.upsample_kernelSize//2)
        self.DSNT_Scale = torch.ones(learnSize)
        self.DSNT_Scale[1:learnSize//2+1] = 2*(torch.pi)/torch.arange(1,learnSize//2+1)
        self.DSNT_Scale[learnSize//2+1: ] = 2*(torch.pi)/(learnSize//2+1-torch.arange(1,learnSize//2+1))
        self.DSNT_Scale[0] = 4
        self.KLDiv = torch.nn.KLDivLoss(reduction='batchmean')

        y,x = torch.meshgrid(torch.linspace(-1,1,self.hm_outSize),
                             torch.linspace(-1,1,self.hm_outSize))
        self.pos = torch.dstack((x,y))
        self.x = torch.linspace(-1,1,self.hm_outSize)
        self.dev = dev
        self.pos = self.pos.to(self.dev)
        self.DSNT_Scale = self.DSNT_Scale.to(self.dev)
        self.x = self.x.to(self.dev)
        
    def get_shape(self):
        
        with torch.no_grad():
            if self.coord:
                X = torch.Tensor(1,5,self.input_size,self.input_size)
            else:
                X = torch.Tensor(1,3,self.input_size,self.input_size)
            
            if self.encoder_name == "Vit":
                hiddenSize = first(self.blocks(X))
                return [hiddenSize.shape[-1], 1, 1]
            else:
                Y = nn.Sequential(*self.blocks)(X)
                return Y.shape[1:]                     

    def forward(self, x):

        if self.coord:
            x = self.addCoords(x)
        
        if self.encoder_name == "Vit":
            
            z_last = first(self.blocks(x)).unsqueeze(-1).unsqueeze(-1)            
            features = None
            
        else:
            z_cur = x
            features = [z_cur]
            for block in self.blocks:
                z_next = block(z_cur)
                features.append(z_next)
                z_cur = z_next

            z = last(features)

        y = self.upsample(z) # heatmap predictions, with size Batch*learnSize*input_Size*input_Size
        b,c,h,w = y.shape
        hm = torch.nn.functional.softmax(y.reshape(b,c,h*w),dim=2).reshape(b,c,h,w)
        p = self.DSNT(hm)
        logit = p*self.DSNT_Scale.tile((2,1))
       
        return {"logit" : logit, "hm" : hm, 'centres' : p}
    
    def getLoss(self, outputs, y, lamb=1):

        hm, logit, p = outputs['hm'], outputs['logit'], outputs['centres'].detach()

        # First calculate loss
        # We only learn low freqeuncies of 'y' (label).
        W, O = self.learnSize // 2, self.learnSize % 2
        y_low = torch.cat((y[:,:,0:W+O], y[:,:,-W:]), dim = -1)
        
        weights = get_weight(y_low)
    
        loss1 = l1_weighted(y_low, logit, weights) + l2_weighted(y_low, logit, weights)
        
        # Second calculate JS Divergence regularization
        target_dist = self.get_targetDist(p, num_points=self.hm_outSize, cov=self.cov)
        b,c,h,w = hm.shape
        m = (target_dist+hm+1e-8)/2
        JS_Div = self.KLDiv(torch.log(hm+1e-8).reshape(b*c,h,w),m.reshape(b*c,h,w))/2+\
                 self.KLDiv(torch.log(target_dist+1e-8).reshape(b*c,h,w),m.reshape(b*c,h,w))/2
        
        return loss1 + lamb*JS_Div
    
    def DSNT(self, hm):
        x_coord = torch.matmul(hm,self.x).sum(dim=-1)
        y_coord = torch.matmul(self.x,hm).sum(dim=-1)
        p = torch.cat((x_coord.unsqueeze(1),y_coord.unsqueeze(1)),dim=1)
        return p

    def getLogit(self, hm):
        p = self.DSNT(hm)
        logit = p*self.DSNT_Scale.tile((2,1))
        return logit

    def get_targetDist(self, means, num_points=256, cov=0.003):
        means = means.clone()
        means = means.permute(0,2,1)
        b,l,_ = means.shape
        means = means.reshape(b,l,1,1,2)    
        dist = self.pos-means
        tmp = -0.5*dist**2/cov
        tmp = tmp.sum(dim=-1)
        rv = torch.exp(tmp)
        rv_sums = rv.reshape(-1, num_points**2).sum(dim=-1).reshape(b,l,1,1)
        rv = rv/(rv_sums+1e-8)
        return rv


def get_weight(y_low):
    # y_low is the label (type: torch tensor) containing low frequencies of FFT coefs.
    # y_low has shape: Batch*2*learnSize
    m = y_low.shape[-1]
    max_values = y_low.view(-1,m).max(dim=0)[0]
    min_values = y_low.view(-1,m).min(dim=0)[0]

    # We clamp the weight at 10 to prevent extremely large weight values
    weight = torch.minimum(1+1/(max_values-min_values+1e-5), torch.tensor(10))

    return weight

def l1_weighted(y, y_pred, weight):
    return ((y-y_pred).abs() * weight).mean()

def l2_weighted(y, y_pred, weight):
    return ((y-y_pred)**2 * weight).mean()
            
def get_children(model):
        
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

if __name__ == "__main__" :
    
    input_size   = [2, 3, 256, 256]
    encoder_name = "DResNet"
    codeSize     = 71
    learnSize    = 21 

    x = torch.randn(*input_size).cuda()
    y = torch.randn(input_size[0], 2, codeSize).cuda()

    net = NETS(encoder_name, 256, learnSize).cuda()

    output = net(x)
    
    hm = output["hm"]
    print(hm.shape)
    # print(net.getLoss(logit, y))
    

