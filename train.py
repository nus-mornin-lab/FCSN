from matplotlib import pyplot as plt

import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
from easydict import EasyDict
from glob import glob

from toolz import *
from toolz.curried import *
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

from data.dataLoader import MakeDataLoader

from model.FFTNet import FFTNet
from model.FFTNet import FFTNet_DSNT
from model.UNet import UNet
from model.DeepLab import DeepLab 
from model.NonLocal import NonLocal
from model.TRANS import TRANS 

from utils import GET_OPTIMIZER
from torchmetrics.functional import dice_score 

from visualisation.Visualization import visualisation

from scipy.spatial.distance import directed_hausdorff

def parse():

    parser = ArgumentParser()

    parser.add_argument("--dataPath", type=str, default="./data/datasets")
    parser.add_argument("--logPath", type=str, default="./log")
    parser.add_argument("--ckptPath", type=str, default="./ckpt")
    parser.add_argument("--visPath", type=str, default="./vis")
    
    # training related
    parser.add_argument("--task", type=str, default="ISIC", help = "PET | ISIC | PROSTATE | FETAL | RIM_CUP | RIM_DISC")
    parser.add_argument("--augType", type=str, default="aug0", help = "aug0 | aug1")
    parser.add_argument("--model", type=str, default="TRANS", help = "FFTNet | FFTDSNTNet | UNet | DeepLab | NonLocal | TRANS " )
    parser.add_argument("--lossType", type=str, default="dice", help = "dice | lovasz")
    parser.add_argument("--fold", type=str, default = "0", help = " 0 | 1 | 2 | 3 | 4")
        
    # hyprer parameter
    parser.add_argument("--inputSize", type=int, default=256)    
    parser.add_argument("--batchN", type=int, default=8)
    parser.add_argument("--epochN", type=int, default=350)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default= 3e-4)
    parser.add_argument("--weight_decay", type=float, default= 0)
    
    #hardware
    parser.add_argument("--cpuN", type=int, default=4)
    parser.add_argument("--gpuN", type=str, default="1", help = "0|1|2|3")
    
    #model specific args
    parser.add_argument("--learnSize", type=int, default=21)
    parser.add_argument("--encoderName", type=str, default="ResNet")
    parser.add_argument("--modelName", type=str, default="standard")
    
    config = first(parser.parse_known_args())
    
    # pick a gpu that has the largest space
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpuN)
    
    if "FFT" not in config.model:
        config.epochN = 200
    
    if config.model == "FFTNet":
        
        config.learnSize   = config.learnSize 
        config.codeSize    = 71
        net = FFTNet.NETS(config.encoderName, config.inputSize, config.learnSize)

    if config.model == "FFTDSNTNet":
        
        config.learnSize   = config.learnSize 
        config.codeSize    = 71
        config.cov         = 0.005
        net = FFTNet_DSNT.NETS(config.encoderName, config.inputSize, config.learnSize, cov = config.cov)        
        
    if config.model == "UNet":
        net = UNet.NETS(modelName = config.modelName, lossType = config.lossType)
        
    if config.model == "DeepLab":
        net = DeepLab.NETS(modelName = config.modelName, encoderName = "resnet50", lossType = config.lossType, outputStride = 8)
        
    if config.model == "NonLocal":
        net = NonLocal.NETS()
        
    if config.model == "TRANS":
        net = TRANS.NETS()                
                
    return config, net

def trainStep(x, y, net, optimizer):

    optimizer.zero_grad()
    
    outputs = net(x)
    
    loss = net.getLoss(outputs, y)
    
    loss.backward()    
    optimizer.step()
    
    return loss, outputs['logit']

@torch.no_grad()
def validStep(x, y, net):
    
    outputs = net(x)
    
    loss = net.getLoss(outputs, y)
    
    return loss, outputs['logit']

def getMetric(pred, mask):
    
    dice_score = lambda pred, mask : torch.sum(pred[mask==1])*2.0 / (torch.sum(pred) + torch.sum(mask))     
    
    dice = np.array(dice_score(pred.view(-1), mask.view(-1)))

    return {'dice' : dice}
    
def saveResults(cases, net, savePath, vis):
    """
    cases: a dictionary with key (name of the dataloader) and value (dataloader), e.g. {'train':trainDataloader,'valid':validDataloader}
    """
    
    for case in cases:
        savePathCase = os.path.join(savePath, case)
        os.makedirs(savePathCase, exist_ok=True)
        for i, (x, y, fNames) in enumerate(cases[case]):
            _, logit = validStep(x.type(torch.float32).cuda(), y.type(torch.float32).cuda(), net)
            for img, conFFT, predFFT, fName in zip(x, y, logit, fNames):
                # inside vis we used torch.tensor -> numpy.arry, so make sure tensors are NOT on GPU
                img, conFFT, predFFT = img.detach().cpu(), conFFT.detach().cpu(), predFFT.detach().cpu()
                vis.save_results(os.path.join(savePathCase, fName), \
                                 img, [conFFT, predFFT])

def train(trainDataLoader, validDataLoader, net, optimizer, config) :
    
    if config.model == "FFTDSNTNet" :
        net.initialize_Scale(validDataLoader)        
        print(net.DSNT_Scale)
    
    if "FFT" in config.model :
        vis = visualisation(config.learnSize, config.codeSize, config.inputSize)
    
    bestValidDice, bestValidHaus = 0, np.inf
    bestDiceEpoch, bestHausEpoch = 0, 0
    net.train()    
    for i,e in enumerate(range(config.epochN)):
        
        # train 
        ######################################################
        lossTrain = 0
        net.train()
        for (x, y, _) in tqdm(trainDataLoader):
            
            loss, logit = trainStep(x.type(torch.float32).cuda(),
                                    y.type(torch.float32).cuda(),
                                    net,
                                    optimizer)
            
            lossTrain += loss.item()
            
        
        ######################################################

        savePath = os.path.join(f"{config.visPath}/{config.task}/{config.model}", 'epoch_' + str(e))
        os.makedirs(os.path.join(savePath,'valid'), exist_ok=True)

        # valid
        ######################################################
        lossValid = 0
        diceValid = 0
        #HausdorffValid = 0
        net.eval()        
        for (x, y, fName) in validDataLoader:
            
            loss, logit = validStep(x.type(torch.float32).cuda(),
                                    y.type(torch.float32).cuda(),
                                    net)
            
            logit  = logit.detach().cpu().squeeze()
            target = y.detach().cpu().squeeze()
            
            if "FFT" in config.model :
                
                pred = vis.draw_polygon(vis.get_contour(vis.pad_FFTs(logit))).type(torch.long)
                mask = vis.draw_polygon(vis.get_contour(target)).type(torch.long)
                
            else :
                
                pred = logit.argmax(0).type(torch.long)
                mask = target.type(torch.long)
                                        
            metrics = getMetric(pred, mask)
            
            lossValid += loss.item()
            diceValid += metrics['dice']
            
        lossTrain     = round(lossTrain / len(trainDataLoader), 3)
        currValidLoss = round(lossValid / len(validDataLoader), 3)
        currValidDice = round(diceValid / len(validDataLoader), 3)
        
        print(f"######## EPOCH : ({e}/{config.epochN}) / TRAIN LOSS : {lossTrain} ########\n")
        
        if currValidDice < bestValidDice:            
            print(f"(aborting) {currValidDice} : lower than the previous best DICE score {bestValidDice} ({bestDiceEpoch})")
            
        else:            
            print(f"(saving) {currValidDice} : higher than the previous best DICE score {bestValidDice} ({bestDiceEpoch})")            
            
            bestValidDice = currValidDice
            bestDiceEpoch = e
            
            if "FFT" in config.model : path = f"{config.ckptPath}/{config.task}/{config.model}/{config.encoderName}_{config.learnSize}/{config.fold}"
            else                     : path = f"{config.ckptPath}/{config.task}/{config.model}/{config.modelName}/{config.fold}"
                
            os.makedirs(f"{path}", exist_ok=True)
            torch.save(net.state_dict(), f"{path}/ckpt.pt")
            
if __name__ == "__main__" :
    
    config, net = parse()
    print(config)
    
    # optmizer
    ###################################################    
    optimizer = GET_OPTIMIZER(net.parameters(), config.optimizer, config.lr, config.weight_decay)
    
    # dataloaders
    ###################################################
    trainDataLoader = MakeDataLoader(f"{config.dataPath}/{config.task}/processed",
                                      inputSize = config.inputSize,
                                      task      = config.task,
                                      mode      = "train",
                                      fold      = int(config.fold),
                                      maskType  = ("fourier", config.codeSize) if "FFT" in config.model else "original",
                                      batchN    = config.batchN, 
                                      cpuN      = config.cpuN,
                                      augType   = config.augType)
    
    validDataLoader = MakeDataLoader(f"{config.dataPath}/{config.task}/processed",
                                      inputSize = config.inputSize,
                                      task      = config.task,
                                      mode      = "valid",
                                      fold      = int(config.fold), 
                                      maskType  = ("fourier", config.codeSize) if "FFT" in config.model else "original",
                                      batchN    = 1, 
                                      cpuN      = config.cpuN,
                                      augType   = config.augType)
    # train & valid
    ###################################################
    train(trainDataLoader, validDataLoader,
          net.cuda(),
          optimizer, 
          config)
