import torch
import torch.nn as nn

class ShiftDecorator:
    def __init__(self, shift):
        self.shift = shift
    
    def input_transform(self, x):
        return torch.sub(x, self.shift)
    
    def output_transform(self, x):
        return nn.Identity()(x)