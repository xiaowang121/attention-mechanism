import torch
from numba import jit
import numpy as np

@jit
def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold

    GT = GT == torch.max(GT)

    corr = torch.sum(SR==GT)

    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc


@jit
def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    TP = ((SR==1).int()+(GT==1).int())==2
    FP = ((SR==1).int()+(GT==0).int())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

@jit
def get_JS(SR,GT,threshold=0.5):

    SR = SR > threshold
    GT = GT == torch.max(GT)
    sum_int = SR.int() + GT.int()

    Inter = (SR * GT).sum()
    Union = torch.sum((sum_int)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS
@jit
def get_DC(SR,GT,threshold=0.5):

    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = (SR * GT).sum()
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

