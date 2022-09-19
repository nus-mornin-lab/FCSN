import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    """
    https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    
    def forward(self, logits, true):
        
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
    return (1 - dice_loss)
