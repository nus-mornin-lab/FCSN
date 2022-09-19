"""
this is visualisation class that is modified from the previous visualisation class to provide
    1. multiple image viusalisation
    3. able to change to color
    
I do all these to have a better visualsation for our result section.
"""

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch
import numpy as np

from skimage.draw import polygon

from torchvision.utils import make_grid
from torchvision.utils import draw_segmentation_masks as pasteMask

import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity as _normalise

class visualisation():

    def __init__(self, learnSize = 31, codeSize = 71, inputSize = 256):
        
        self.learnSize = learnSize
        self.codeSize  = codeSize
        self.m, self.n = [inputSize] * 2
        
    @curry
    def normalise(self, img):                        
        return torch.tensor(_normalise(np.array(img), out_range = (0, 255))).type(torch.uint8)
    
    def draw_polygon(self, contour, color=1):

        m, n = self.m, self.n

        contour[0] = contour[0] * m + m // 2
        contour[1] = contour[1] * n + n // 2

        rr, cc = polygon(contour[0], contour[1])
        
        # treat unrealistic predictions
        rr = np.where(rr > m - 1, m - 1, rr); rr = np.where(rr < 0, 0, rr)
        cc = np.where(cc > n - 1, n - 1, cc); cc = np.where(cc < 0, 0, cc)
        
        # In PyTorch, img has size channel*M*N
        # We assume img has range in [0,1]
        mask = torch.zeros(m, n).type(torch.bool)
        mask[rr, cc] = True

        return mask

    def pad_FFTs(self, con_FFT):
        
        if con_FFT.shape[-1] != self.codeSize:
            con_FFT_padded = torch.zeros((2, self.codeSize))
            W, O = self.learnSize // 2, self.learnSize % 2
            con_FFT_padded[:, :W + O] = con_FFT[:, :W + O]
            con_FFT_padded[:, -W:] = con_FFT[:, -W:]
            return con_FFT_padded
        else:
            return con_FFT

    def get_contour(self, coefs):
        
        coefs = self.pad_FFTs(coefs)
        
        Z = torch.fft.ifft(coefs[0, :] + 1j * coefs[1, :], norm='ortho')
        
        contour = torch.stack([Z.real, Z.imag])
        
        return contour

    def toGrid(self, imgs, masks, colors = "blue"):
        
        grid = [pasteMask(self.normalise(img), masks=mask, alpha=0.4, colors = colors) for img, mask in zip(imgs, masks)]
        
        return torch.stack(grid)    
                
    def showEval(self, imgss, maskss, maskColor = "blue"):
        
        n = len(imgss[0])
        
        grids = []
        for i in range(n):
            
            imgs  = [imgs[i]  for imgs in imgss]
            masks = [masks[i] for masks in maskss]
            
            grid = self.toGrid(imgs, masks, colors = maskColor)
            grids.append(grid)
        
        grids = torch.stack(grids)        
        
        sampleN, maskN, C, H, W = grids.shape 
        #grids = grids.permute(1,0,2,3,4)
        grids = grids.reshape(maskN * sampleN, C, H, W)
        
        grid = make_grid(grids, nrow = len(maskss))
        
        return grid
    
        
        
            
            
            
            
            
        
        