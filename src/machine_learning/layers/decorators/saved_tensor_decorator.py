import torch.nn as nn

class SavedTensorDecorator(nn.Module):
    def __init__(self):
        self.latest_input = None
        self.latest_output = None
    
    def input_transform(self, x):
        if(len(x.shape) > 1):
            self.latest_input = x
            
        return nn.Identity()(x)
    
    def output_transform(self, x):
        if(len(x.shape) > 1):
            self.latest_output = x

        return nn.Identity()(x)