import torch.nn as nn

class BasicClassificationLoss:
    def __call__(self, model, x, y):
        return nn.CrossEntropyLoss()(model(x), y)
    