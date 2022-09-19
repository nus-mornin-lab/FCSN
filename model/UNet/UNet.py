"""
collection of unet family networks.

currently supports 
    1. UNet    : https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
    2. VNet    : https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
"""
import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append('..')

try:
    from lovasz_losses import lovasz_softmax as lovasz
except:
    from ..lovasz_losses import lovasz_softmax as lovasz


class NETS(nn.Module):
    
    def __init__(self, modelName = ["standard", "plus"][0], lossType = ["dice", "lovasz"][0], classN = 1, startC = 64, inC = 3):
        super().__init__()
        
        if modelName == "standard" : self.net = UNet(classN, startC, inC)            
        if modelName == "plus"     : self.net = NestedUNet(classN, startC, inC)

        self.lossType = lossType
        self.diceLoss = DiceLoss()
        # if loss == "dice":
        #     self.loss = DiceLoss()
        # elif loss == "lovasz":
        #     self.loss = 
                                    
    def forward(self, x):
        
        logit, z = self.net(x)
        
        return {"logit" : logit, "embeded" : z}
    
    def getLoss(self, outputs, target):

        if self.lossType == "dice":
            loss = self.diceLoss(outputs["logit"], target.type(torch.long))
        elif self.lossType == "lovasz":
            loss = lovasz(F.sigmoid(outputs["logit"]), target.type(torch.long), classes=[1])
        
        return loss

class UNet(nn.Module):
    def __init__(self, classN, startC, inC = 3):
        super().__init__()

        filterN = [startC*2**(i) for i in range(5)]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = convBlock(inC, filterN[0], filterN[0])
        self.conv1_0 = convBlock(filterN[0], filterN[1], filterN[1])
        self.conv2_0 = convBlock(filterN[1], filterN[2], filterN[2])
        self.conv3_0 = convBlock(filterN[2], filterN[3], filterN[3])
        self.conv4_0 = convBlock(filterN[3], filterN[4], filterN[4])

        self.conv3_1 = convBlock(filterN[3]+filterN[4], filterN[3], filterN[3])
        self.conv2_2 = convBlock(filterN[2]+filterN[3], filterN[2], filterN[2])
        self.conv1_3 = convBlock(filterN[1]+filterN[2], filterN[1], filterN[1])
        self.conv0_4 = convBlock(filterN[0]+filterN[1], filterN[0], filterN[0])

        self.final = nn.Conv2d(filterN[0], classN+1, kernel_size=1)

    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        
        return output, x4_0


class NestedUNet(nn.Module):
    def __init__(self, classN, startC, inC = 3):
        super().__init__()

        filterN = [startC*2**(i) for i in range(5)]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = convBlock(inC, filterN[0], filterN[0])
        self.conv1_0 = convBlock(filterN[0], filterN[1], filterN[1])
        self.conv2_0 = convBlock(filterN[1], filterN[2], filterN[2])
        self.conv3_0 = convBlock(filterN[2], filterN[3], filterN[3])
        self.conv4_0 = convBlock(filterN[3], filterN[4], filterN[4])

        self.conv0_1 = convBlock(filterN[0]+filterN[1], filterN[0], filterN[0])
        self.conv1_1 = convBlock(filterN[1]+filterN[2], filterN[1], filterN[1])
        self.conv2_1 = convBlock(filterN[2]+filterN[3], filterN[2], filterN[2])
        self.conv3_1 = convBlock(filterN[3]+filterN[4], filterN[3], filterN[3])

        self.conv0_2 = convBlock(filterN[0]*2+filterN[1], filterN[0], filterN[0])
        self.conv1_2 = convBlock(filterN[1]*2+filterN[2], filterN[1], filterN[1])
        self.conv2_2 = convBlock(filterN[2]*2+filterN[3], filterN[2], filterN[2])

        self.conv0_3 = convBlock(filterN[0]*3+filterN[1], filterN[0], filterN[0])
        self.conv1_3 = convBlock(filterN[1]*3+filterN[2], filterN[1], filterN[1])

        self.conv0_4 = convBlock(filterN[0]*4+filterN[1], filterN[0], filterN[0])

        self.final = nn.Conv2d(filterN[0], classN+1, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x) 
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        
        return output, x4_0
    
    
class convBlock(nn.Module):
    
    def __init__(self, inC, middleC, outC):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(inC, middleC, 3, padding = 1),
                                   nn.BatchNorm2d(middleC),
                                   nn.ReLU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(middleC, outC, 3, padding = 1),
                                   nn.BatchNorm2d(outC),
                                   nn.ReLU())
        
    def forward(self, x):
        return self.conv2(self.conv1(x))    
                
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
        
    modelName  = ["standard", "plus"][0]
    lossType = ["dice", "lovasz"][1]
    
    x = torch.randn(2, 3, 256, 256).to("cuda")
    y = torch.randint(0,2, (2, 256, 256)).to("cuda")
    
    net = NETS(modelName = modelName, lossType = lossType).to("cuda")
    
    logit = net(x)
    
    loss = net.getLoss(logit, y)
    print(loss)
    
    
    
    