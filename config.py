from models.vgg16 import TorchVGG16, MindSporeVGG16
from models.lstm0 import TorchLSTMModel0,MindSporeLSTMModel0
from models.alexnet import TorchAlexNetCIFAR10,MindSporeAlexNetCIFAR10
from models.densenet import TorchDenseNet,MSDenseNet
from models.inceptionblock import TorchInceptionBlock,MindSporeInceptionBlock
from models.lenet5fashion import TorchLeNet5Fashion,MindSporeLeNet5Fashion
from models.lenet5mnist import TorchLeNet5,MindSporeLeNet5
from models.lstm2 import TorchLSTMModel2,MindSporeLSTMModel2
from models.mobilenet import TorchMobileNet,MindSporeMobileNet
from models.resnet50 import TorchResNet50,MindSporeResNet50
from models.vgg19 import TorchVGG19,MindSporeVGG19
from models.xception import TorchXception,MindSporeXception

model_map={
    "TorchVGG16":TorchVGG16,
    "MindSporeVGG16":MindSporeVGG16,
    "VGG16":(3,224,224),

    'TorchLSTMModel0':TorchLSTMModel0,
    'MindSporeLSTMModel0':MindSporeLSTMModel0,
    'LSTMModel0':(1,10,50),

    'TorchAlexNetCIFAR10':TorchAlexNetCIFAR10,
    'MindSporeAlexNetCIFAR10':MindSporeAlexNetCIFAR10,
    'AlexNetCIFAR10':(3,32,32),

    'TorchDenseNet':TorchDenseNet,
    'MSDenseNet':MSDenseNet,
    'DenseNet':(3,224,224),

    'TorchInceptionBlock':TorchInceptionBlock,
    'MindSporeInceptionBlock':MindSporeInceptionBlock,
    'InceptionBlock':(64, 224, 224),

    'TorchLeNet5Fashion':TorchLeNet5Fashion,
    'MindSporeLeNet5Fashion':MindSporeLeNet5Fashion,
    'LeNet5Fashion':(1, 28, 28),

    'TorchLeNet5':TorchLeNet5,
    'MindSporeLeNet5':MindSporeLeNet5,
    'LeNet5': (1, 28, 28),

    'TorchLSTMModel2':TorchLSTMModel2,
    'MindSporeLSTMModel2':MindSporeLSTMModel2,
    'LSTMModel2':(32, 20, 10),

    'TorchMobileNet':TorchMobileNet,
    'MindSporeMobileNet':MindSporeMobileNet,
    'MobileNet': (3, 224, 224),

    'TorchResNet50':TorchResNet50,
    'MindSporeResNet50':MindSporeResNet50,
    'ResNet50': (3, 224, 224),

    'TorchVGG19':TorchVGG19,
    'MindSporeVGG19':MindSporeVGG19,
    'VGG19':(3, 224, 224),

    'TorchXception':TorchXception,
    'MindSporeXception':MindSporeXception,
    'Xception':(3, 299, 299)
}

'''
TORCH_MODEL=['TorchDenseNet']
MS_MODEL=['MSDenseNet']
INPUTSHAPE=['DenseNet']
'''


TORCH_MODEL=['TorchVGG16','TorchLSTMModel0','TorchAlexNetCIFAR10','TorchDenseNet','TorchLeNet5Fashion', 'TorchLeNet5',
             'TorchLSTMModel2','TorchMobileNet','TorchResNet50','TorchVGG19','TorchXception']
MS_MODEL=['MindSporeVGG16','MindSporeLSTMModel0','MindSporeAlexNetCIFAR10','MSDenseNet','MindSporeLeNet5Fashion','MindSporeLeNet5',
          'MindSporeLSTMModel2','MindSporeMobileNet','MindSporeResNet50','MindSporeVGG19','MindSporeXception']
INPUTSHAPE=['VGG16','LSTMModel0','AlexNetCIFAR10','DenseNet','LeNet5Fashion','LeNet5',
'LSTMModel2','MobileNet','ResNet50','VGG19','Xception']


group_0 = [
    ('TorchVGG16', 'VGG16'),
    #('TorchLSTMModel0', 'LSTMModel0'),
    ('TorchAlexNetCIFAR10', 'AlexNetCIFAR10'),
    ('TorchDenseNet', 'DenseNet')
]

group_1 = [
    ('TorchLeNet5Fashion', 'LeNet5Fashion'),
    ('TorchLeNet5', 'LeNet5'),
    #('TorchLSTMModel2', 'LSTMModel2'),
    ('TorchMobileNet', 'MobileNet')
]

group_2 = [
    ('TorchResNet50', 'ResNet50'),
    ('TorchVGG19', 'VGG19'),
    ('TorchXception', 'Xception')
]


NUM_CLASSES=100

#LSTMModel2 在ART中会自动改成4维，未解决