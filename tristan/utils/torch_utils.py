import torch
from torch.autograd import Variable

def to_var(x, is_cuda=torch.cuda.is_available(), **kwargs):
    if is_cuda:
        x = x.cuda()
    return Variable(x, **kwargs)
