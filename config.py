from models.vgg16 import TorchVGG16,
from models.vgg16 import TorchVGG16
from models.lstm0 import TorchLSTMModel0
from models.alexnet import TorchAlexNetCIFAR10
from models.densenet import TorchDenseNet
from models.inceptionblock import TorchInceptionBlock
from models.lenet5fashion import TorchLeNet5Fashion
from models.lenet5mnist import TorchLeNet5
from models.lstm2 import TorchLSTMModel2
from models.mobilenet import TorchMobileNet
from models.resnet50 import TorchResNet50
from models.vgg19 import TorchVGG19
from models.xception import TorchXception

# 全局类别数
NUM_CLASSES = 100

MODEL_CLASS_MAP = {
    "TorchVGG16": TorchVGG16,
    "TorchLSTMModel0": TorchLSTMModel0,
    "TorchAlexNetCIFAR10": TorchAlexNetCIFAR10,
    "TorchDenseNet": TorchDenseNet,
    "TorchInceptionBlock": TorchInceptionBlock,
    "TorchLeNet5Fashion": TorchLeNet5Fashion,
    "TorchLeNet5": TorchLeNet5,
    "TorchLSTMModel2": TorchLSTMModel2,
    "TorchMobileNet": TorchMobileNet,
    "TorchResNet50": TorchResNet50,
    "TorchVGG19": TorchVGG19,
    "TorchXception": TorchXception,
}

# (Channels, Height, Width)
MODEL_SHAPE_MAP = {
    "VGG16": (3, 224, 224),
    "LSTMModel0": (1, 10, 50),
    "AlexNetCIFAR10": (3, 32, 32),
    "DenseNet": (3, 224, 224),
    "InceptionBlock": (64, 224, 224),
    "LeNet5Fashion": (1, 28, 28),
    "LeNet5": (1, 28, 28),
    "LSTMModel2": (32, 20, 10),
    "MobileNet": (3, 224, 224),
    "ResNet50": (3, 224, 224),
    "VGG19": (3, 224, 224),
    "Xception": (3, 299, 299),
}

model_map = {**MODEL_CLASS_MAP, **MODEL_SHAPE_MAP}

# 分组配置 ('ModelName', 'ShapeKey')
group_0 = [
    ('TorchAlexNetCIFAR10', 'AlexNetCIFAR10'),
    ('TorchDenseNet', 'DenseNet'),
    ('TorchLeNet5Fashion', 'LeNet5Fashion'),
    ('TorchVGG16', 'VGG16')
]

group_1 = [
    ('TorchLeNet5', 'LeNet5'),
    ('TorchResNet50', 'ResNet50'),
    ('TorchXception', 'Xception')
]

# TorchLSTMModel0 (LSTMModel2在ART中会自动改成4维，未解决)
# TorchMobileNet (DeepFool不收敛)
# TorchVGG19 (DeepFool超时)

