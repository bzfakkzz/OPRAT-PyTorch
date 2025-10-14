from .vgg16 import TorchVGG16, MindSporeVGG16
from .lstm0 import TorchLSTMModel0,MindSporeLSTMModel0
from .alexnet import TorchAlexNetCIFAR10,MindSporeAlexNetCIFAR10
from .densenet import TorchDenseNet,MSDenseNet
from .inceptionblock import TorchInceptionBlock,MindSporeInceptionBlock
from .lenet5fashion import TorchLeNet5Fashion,MindSporeLeNet5Fashion
from .lenet5mnist import TorchLeNet5,MindSporeLeNet5
from .lstm2 import TorchLSTMModel2,MindSporeLSTMModel2
from .mobilenet import TorchMobileNet,MindSporeMobileNet
from .resnet50 import TorchResNet50,MindSporeResNet50
from .vgg19 import TorchVGG19,MindSporeVGG19
from .xception import TorchXception,MindSporeXception
from .convert_weights import convert_weights

__all__=[
    'convert_weights',
    'TorchVGG16',
    'MindSporeVGG16',
    'TorchLSTMModel0',
    'MindSporeLSTMModel0',
    'TorchAlexNetCIFAR10',
    'MindSporeAlexNetCIFAR10',
    'TorchDenseNet',
    'MSDenseNet',
    'TorchInceptionBlock',
    'MindSporeInceptionBlock',
    'TorchLeNet5Fashion',
    'MindSporeLeNet5Fashion',
    'TorchLeNet5',
    'MindSporeLeNet5',
    'TorchMobileNet',
    'MindSporeMobileNet',
    'TorchResNet50',
    'MindSporeResNet50',
    'TorchVGG19',
    'MindSporeVGG19',
    'TorchXception',
    'MindSporeXception'
]