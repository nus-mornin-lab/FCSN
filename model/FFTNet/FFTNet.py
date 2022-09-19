import torch
import numpy as np

from toolz         import *
from toolz.curried import *

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet, vgg16, resnet34, resnet50

try:
    from .ViT import VisionTransformer, CONFIGS as vitConfigs
    from .backbone.drn import drn_c_26 as DResNet
    from .coordConv import AddCoords
except:
    from ViT import VisionTransformer, CONFIGS as vitConfigs
    from backbone.drn import drn_c_26 as DResNet
    from coordConv import AddCoords
    
class NETS(nn.Module):
   
    def __init__(self, encoder_name, input_size, learnSize, coord = True, pretrained = False):
       
        super(NETS, self).__init__()
       
        blocks = {
            ""
            "AlexNet"    : lambda : nn.Sequential(*alexnet(pretrained=pretrained).features),
            "AlexNet_S"  : lambda : nn.Sequential(*get_children(alexnet(pretrained=pretrained).features)[:-1]),
            "VGGNet"     : lambda : nn.Sequential(vgg16(pretrained=pretrained).features),
            "VGGNet_S"   : lambda : nn.Sequential(*get_children(vgg16(pretrained=pretrained).features)[:-10]),
            "VGGNet_VS"  : lambda : nn.Sequential(*get_children(vgg16(pretrained=pretrained).features)[:-15]),
            "DResNet"  : lambda : nn.Sequential(*list(DResNet(pretrained=False).children())[:-2]),
            "ResNet"     : lambda : nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-2]),
            "Vit"        : lambda : VisionTransformer(vitConfigs['ViT-L_16'], img_size = input_size)}
        
        self.encoder_name = encoder_name
        self.input_size   = input_size
        self.learnSize    = learnSize
        self.coord = coord
        self.blocks  = blocks[self.encoder_name]()
        self.blocks = nn.Sequential(nn.Conv2d(5, 16, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
                                    *self.blocks[1:])
        self.z_shape = self.get_shape()
        self.addCoords = AddCoords()
        self.ZConv = nn.Sequential(nn.Conv2d(self.z_shape[0], 10 * self.learnSize, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(10 * self.learnSize, 10 * self.learnSize, 1))

        ZConvSize = 10 * self.learnSize * self.z_shape[1] * self.z_shape[2]

        self.fft = nn.Sequential(nn.Linear(ZConvSize, 20 * self.learnSize),
                                 nn.ReLU(),
                                 nn.Linear(20 * self.learnSize, 20 * self.learnSize),
                                 nn.ReLU(),
                                 nn.Linear(20 * self.learnSize, 2 * self.learnSize)
                                 )
        
    def get_shape(self):
        
        with torch.no_grad():

            X = torch.Tensor(1,5,self.input_size,self.input_size)
            
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

        zConv = torch.flatten(self.ZConv(z), start_dim=1)
        logit = self.fft(zConv).view(-1, 2, self.learnSize)
       
        return {'logit' : logit}
    
    def getLoss(self, outputs, y):
        
        logit = outputs["logit"]

        # We only learn low freqeuncies of 'y' (label).
        W, O = self.learnSize // 2, self.learnSize % 2
        y_low = torch.cat((y[:,:,0:W+O], y[:,:,-W:]), dim = -1)
        
        weights = get_weight(y_low).to("cuda")
    
        loss = l1_weighted(y_low, logit, weights) + l2_weighted(y_low, logit, weights)
        
        return loss
    
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
    encoder_name = ["AlexNet","VGGNet","SqueezeNet","ResNet", "Vit"][0]
    codeSize     = 200
    learnSize    = 41 # hongfei told me odd number is preferred for a technical reason

    x = torch.randn(*input_size).cuda()    
    y = torch.randn(input_size[0], 2, codeSize).cuda()

    net = NETS(encoder_name, 256, learnSize).cuda()

    logit, features = net(x)
    
    print(net.getLoss(logit, y))
    
    
