import torch.nn as nn

class DecoratedActivation(nn.Module):
    def __init__(self, base_activation, decorator):
        super(DecoratedActivation, self).__init__()
        self.base_activation = base_activation
        self.decorator = decorator
    
    def forward(self, x):
        out = self.decorator.input_transform(x)
        out = self.base_activation(out)
        out = self.decorator.output_transform(out)
        return out

class TransformedActivation(nn.Module):
    def __init__(self, base_activation, decorators):
        super(TransformedActivation, self).__init__()
        self.base_activation = base_activation
        self.decorators = decorators
        self._update_activation()
    
    def decorate(self, decorator):
        self.decorators.append(decorator)
        self._update_activation()

    def pop_decorator(self):
        popped_decorator = self.decorators.pop()
        self._update_activation()
        return popped_decorator

    def forward(self, x):
        return self.act(x)
    
    def _update_activation(self):
        self.act = self.base_activation
        for decorator in self.decorators:
            self.act = DecoratedActivation(self.act, decorator)