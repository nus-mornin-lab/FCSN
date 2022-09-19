import torch
from toolz import *
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop

def GET_OPTIMIZER(params, opt_name, learning_rate, weight_decay):
    
    return \
        {"SGD"      : lambda : SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay),
         "Adadelta" : lambda : Adadelta(params, lr=learning_rate, weight_decay=weight_decay),
         "Adagrad"  : lambda : Adagrad(params, lr=learning_rate, weight_decay=weight_decay),
         "Adam"     : lambda : Adam(params, lr=learning_rate, weight_decay = weight_decay),
         "RMSprop"  : lambda : RMSprop(params, lr=learning_rate, momentum=0.9)}[opt_name]()

if __name__ == "__main__" :
    pass
