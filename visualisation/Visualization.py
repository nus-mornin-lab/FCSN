import torch
import torchvision.transforms.functional as F
import numpy as np
from torchvision.utils import make_grid, save_image
from skimage.draw import polygon
from torchvision.utils import draw_segmentation_masks as pasteMask
import matplotlib.pyplot as plt

class visualisation():

    def __init__(self, learnSize=21, codeSize=71, inputSize=256):
        
        self.learnSize = learnSize
        self.codeSize = codeSize
        self.m, self.n = inputSize, inputSize

    def normalise(self, imgs):
        # We normalise imags to range [0,1]
        if imgs.min() > -1e-5:
            # Images in ISIC are already in the range [0,1]
            return imgs
        else:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            std_inv = 1 / (std + 1e-7)
            mean_inv = -mean * std_inv
            imgs_norm = F.normalize(imgs, mean_inv, std_inv)
            return imgs_norm
        
    def draw_polygon(self, contour, color=1):

        m, n = self.m, self.n

        contour[0] = contour[0] * m + m // 2
        contour[1] = contour[1] * n + n // 2

        rr, cc = polygon(contour[0], contour[1])
        rr = np.where(rr > m - 1, m - 1, rr)
        rr = np.where(rr < 0, 0, rr)
        cc = np.where(cc > n - 1, n - 1, cc)
        cc = np.where(cc < 0, 0, cc)

        # In PyTorch, img has size channel*M*N
        # We assume img has range in [0,1]
        mask = torch.zeros(m, n)
        mask[rr, cc] = 1
        mask = mask > 0.5

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

    def get_contour(self, con_FFT):
        con_FFT = self.pad_FFTs(con_FFT)
        Z = torch.fft.ifft(con_FFT[0, :] + 1j * con_FFT[1, :],norm='ortho')
        x = Z.real.unsqueeze(0)
        y = Z.imag.unsqueeze(0)
        con = torch.cat([x, y])
        return con

    def show(self, imgs):
        """
        This show function is copied from:
        https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py
        """
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    def toGrids(self, img, masks, colors = "red"):
        # masks is a list containing all masks (e.g. predictions, ground truth, etc)

        img = self.normalise(img)
        img = (255*img).type(torch.uint8)
        imgsWithMasks = [pasteMask(img, masks=mask, alpha=0.5, colors = colors) for mask in masks]

        return [img, *imgsWithMasks]

    def save_results(self, save_path, img, predicts, returnOutput = False):
        
        # predicts should be a list containing all predicts
        # supported prediction: mask with same width and height of image, or
        # fft coefs

        if not isinstance(predicts, list):
            predicts = [predicts]

        if img.shape[-2] == predicts[0].shape[-2]:
            masks = [mask>0.5 for mask in predicts]
        else:
            contours = [self.get_contour(conFFT) for conFFT in predicts]
            masks = [self.draw_polygon(contour) for contour in contours]

        imgs = self.toGrids(img, masks)
        imgs_uint8 = [img.type(torch.float32) / 255 if img.dtype is torch.uint8 else img for img in imgs]
        whole_img = make_grid(imgs_uint8, nrow=len(imgs))        
        
        if returnOutput :
            save_image(whole_img, fp=f"{save_path}.png")
        else:
            return whole_img