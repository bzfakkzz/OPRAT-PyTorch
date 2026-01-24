from .vgg16 import TorchVGG16
from .alexnet import TorchAlexNetCIFAR10
from .densenet import TorchDenseNet
from .lenet5fashion import TorchLeNet5Fashion
from .lenet5mnist import TorchLeNet5
from .resnet50 import TorchResNet50
from .xception import TorchXception

__all__=[
    'TorchVGG16',
    'TorchAlexNetCIFAR10',
    'TorchDenseNet',
    'TorchLeNet5Fashion',
    'TorchLeNet5',
    'TorchResNet50',
    'TorchXception'
]
