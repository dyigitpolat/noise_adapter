import torch
import torch.nn as nn

from torch.autograd import Function

class StaircaseFunction(Function):
    @staticmethod
    def forward(ctx, x, Tq):
        #Tq = torch.tensor(Tq)
        #ctx.save_for_backward(x, Tq)
        return torch.floor(x * Tq) / Tq

    @staticmethod
    def backward(ctx, grad_output):
        #x, Tq = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None
    
class QuantizationDecorator:
    def __init__(self, q):
        self.q = q

    def input_transform(self, x):
        return nn.Identity()(x)
    
    def output_transform(self, x):
        return StaircaseFunction.apply(x, self.q)