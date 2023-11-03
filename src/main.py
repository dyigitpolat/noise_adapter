from machine_learning.data_loading.data_loader_factory import DataLoaderFactory
from machine_learning.data_loading.data_providers.cifar_10 import CIFAR10_DataProvider
from machine_learning.training.basic_trainer import BasicTrainer

from machine_learning.loss_functions.basic_classification_loss import BasicClassificationLoss
from machine_learning.layers.decoration import TransformedActivation
from machine_learning.layers.decorators.quantization_decorator import QuantizationDecorator
from machine_learning.layers.decorators.shift_decorator import ShiftDecorator

import torch
import torch.nn as nn

def main():
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    print("CUDA device name: {}".format(torch.cuda.get_device_name(device)))

    cifar10_vgg19 = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg19_bn', pretrained=True)

    print(cifar10_vgg19)

    data_loader_factory = DataLoaderFactory(CIFAR10_DataProvider('datasets'))
    test_loader = data_loader_factory.create_test_loader(128)
    training_loader = data_loader_factory.create_training_loader(2048)
    validation_loader = data_loader_factory.create_validation_loader(128)
    
    q = 8
    for idx, module in enumerate(cifar10_vgg19.features):
        if isinstance(module, nn.ReLU):
            cifar10_vgg19.features[idx] = TransformedActivation(
                module, [
                    QuantizationDecorator(q),
                    ShiftDecorator(-0.5/q)])
    
    for idx, module in enumerate(cifar10_vgg19.classifier):
        if isinstance(module, nn.ReLU):
            cifar10_vgg19.classifier[idx] = TransformedActivation(
                module, [
                    QuantizationDecorator(q),
                    ShiftDecorator(-0.5/q)])
        

    trainer = BasicTrainer(
        cifar10_vgg19, device, training_loader, validation_loader, test_loader, 
        BasicClassificationLoss())
    
    trainer.train_n_epochs(0.0001, 10)
    print("Test accuracy: {}".format(trainer.test()))
    pass

if __name__ == '__main__':
    main()