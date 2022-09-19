import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blocks import *
except:
    from blocks import *
    
class NETS(nn.Module):
    def __init__(self,
                 classN      = 1,
                 startC      = 32,
                 inC         = 3,
                 dropout     = 0.5):
        
        super(NETS, self).__init__()
                
        self.diceLoss = DiceLoss()
        
        self.input_block = ResidualBlock(inC)
        self.conv_input = nn.Conv2d(inC, startC, 1)
        self.down_sample1 = DownSamplingBlock(startC, startC*2)
        self.down_sample2 = DownSamplingBlock(startC*2, startC*4)
        self.bottom = BottomAggBlock(startC*4, startC*2, startC*2, dropout=dropout)
        self.up_sample1 = UpSamplingAggBlock(startC*4, startC*2, startC*2, startC*2)
        self.up_sample2 = UpSamplingAggBlock(startC*2, startC, startC, startC)
        self.output_block = ResidualBlock(startC)
        self.dropout = nn.Dropout(dropout)
        self.conv_output = nn.Conv2d(startC, classN+1, 1)

    def forward(self, x) -> torch.tensor:
        
        x = F.interpolate(x, size=(x.shape[2]//2, x.shape[3]//2))
        
        x = x.contiguous()
        x = self.input_block(x)
        x1 = self.conv_input(x)
        x2 = self.down_sample1(x1)
        x = self.down_sample2(x2)
        x = self.bottom(x)
        x = self.up_sample1(x)
        x = x + x2
        x = self.up_sample2(x)
        x = x + x1
        x = self.output_block(x)
        x = self.dropout(x)        
        x = self.conv_output(x)
        
        logit = F.interpolate(x, size=(x.shape[2]*2, x.shape[3]*2))        
        
        # x = torch.sigmoid(x)        
        return {"logit" : logit}
    
    def getLoss(self, outputs, target):

        loss = self.diceLoss(outputs["logit"], target.type(torch.long))
        
        return loss    
    
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
            
    import os
    from torch.optim import Adam
    
    # pick a gpu that has the largest space
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
                
    x = torch.randn(1, 3, 256, 256).to("cuda")
    y = torch.randint(0,2, (1, 256, 256)).to("cuda")
        
    net = NETS().to("cuda")
        
    logit = net(x)
    
    optimizer = Adam(net.parameters(), lr=0.1, weight_decay = 0)

    loss = net.getLoss(logit, y)
    print(loss)

    loss.backward()    
    optimizer.step()
    
    
