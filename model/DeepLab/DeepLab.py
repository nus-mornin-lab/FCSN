"""
collection of deepLab family networks.

currently supports 
    1. deepLabV3  : https://github.com/VainF/DeepLabV3Plus-Pytorch/
    1. deepLabV3+ : https://github.com/VainF/DeepLabV3Plus-Pytorch/
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try :
    from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, IntermediateLayerGetter
    from .backbone import resnet
except:
    from _deeplab import DeepLabHead, DeepLabHeadV3Plus, IntermediateLayerGetter
    from backbone import resnet

import sys
sys.path.append('..')

try:
    from lovasz_losses import lovasz_softmax as lovasz
except:
    from ..lovasz_losses import lovasz_softmax as lovasz

class NETS(nn.Module): 
    
    def __init__(self,
                 modelName    = ["standard", "plus"][1],         
                 encoderName  = ['resnet18', 'resnet50', 'resnet101'][-1],
                 lossType = ["dice", "lovasz"][0],
                 num_classes  = 1,
                 outputStride = [8,16][0], 
                 pretrained   = False) :
        super().__init__()
                
        if outputStride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]
        
        self.encoder = getEncoder(modelName, encoderName, replace_stride_with_dilation, pretrained)        
        self.decoder = getDecoder(modelName, num_classes, aspp_dilate)
        
        self.lossType = lossType
        self.diceLoss = DiceLoss()
        
    def forward(self, x):
        
        input_shape = x.shape[-2:]
        features = self.encoder(x)
        x = self.decoder(features)
        logit = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return {"logit" : logit, "embeded" : features["out"]}
    
    def getLoss(self, outputs, target):

        if self.lossType == "dice":
            loss = self.diceLoss(outputs["logit"], target.type(torch.long))
        elif self.lossType == "lovasz":
            loss = lovasz(F.sigmoid(outputs["logit"]), target.type(torch.long), classes=[1])

        return loss


    
def getEncoder(modelName, encoderName, replace_stride_with_dilation, pretrained):
    
    if modelName=='standard':
        return_layers = {'layer4': 'out'}
    if modelName=='plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}        
        
    _encoder = resnet.__dict__[encoderName](pretrained = pretrained,
                                           replace_stride_with_dilation=replace_stride_with_dilation)
    
    encoder = IntermediateLayerGetter(_encoder, return_layers)
        
    return encoder

def getDecoder(modelName, num_classes, aspp_dilate):

    inplanes = 2048
    low_level_planes = 256

    if modelName=='standard':
        classifier = DeepLabHead(inplanes, num_classes+1, aspp_dilate)    
    if modelName=='plus':
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes+1, aspp_dilate)
        
    return classifier

class DiceLoss(nn.Module):
    
    """
    https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """        
    
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self, logits, true):
        
        num_classes = logits.shape[1]
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        
        return (1 - dice_loss)

if __name__ == "__main__" :
            
    modelName     = ["standard", "plus"][1]
    encoderName   = ['resnet18', 'resnet50', 'resnet101'][1]
    num_classes   = 1
    output_stride = [8,16][0]
                    
    x = torch.randn(2, 3, 256, 256).to("cuda")
    y = torch.randint(0,2, (2, 256, 256)).to("cuda")
    
    net = NETS(modelName = modelName,
               encoderName = encoderName,
               num_classes = num_classes,
               lossType="lovasz",
               outputStride = output_stride).to("cuda")
        
    out = net(x)
    print(out['logit'].shape)
    
    print(net.getLoss(logit, y))
    
    
    