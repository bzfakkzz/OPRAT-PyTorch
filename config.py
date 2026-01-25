from models.vgg16 import TorchVGG16
from models.alexnet import TorchAlexNetCIFAR10
from models.densenet import TorchDenseNet
from models.lenet5fashion import TorchLeNet5Fashion
from models.lenet5mnist import TorchLeNet5
from models.resnet50 import TorchResNet50
from models.xception import TorchXception

# 全局类别数
NUM_CLASSES = 100

MODEL_CLASS_MAP = {
    "TorchVGG16": TorchVGG16,
    "TorchAlexNetCIFAR10": TorchAlexNetCIFAR10,
    "TorchDenseNet": TorchDenseNet,
    "TorchLeNet5Fashion": TorchLeNet5Fashion,
    "TorchLeNet5": TorchLeNet5,
    "TorchResNet50": TorchResNet50,
    "TorchXception": TorchXception,
}

# (Channels, Height, Width)
MODEL_SHAPE_MAP = {
    "VGG16": (3, 224, 224),
    "AlexNetCIFAR10": (3, 32, 32),
    "DenseNet": (3, 224, 224),
    "LeNet5Fashion": (1, 28, 28),
    "LeNet5": (1, 28, 28),
    "ResNet50": (3, 224, 224),
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
