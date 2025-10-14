from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, DeepFool, \
    UniversalPerturbation, HopSkipJump
from art.estimators.classification import PyTorchClassifier
import torch
import numpy as np

def get_attack(attack_name, classifier, **kwargs):
    if attack_name == 'FGM':
        return FastGradientMethod(estimator=classifier, **kwargs)
    elif attack_name == 'PGD':
        return ProjectedGradientDescent(estimator=classifier, **kwargs)
    elif attack_name == 'CW':
        return CarliniL2Method(classifier=classifier, **kwargs)
    elif attack_name == 'DeepFool':
        return DeepFool(classifier=classifier, **kwargs)
    elif attack_name == 'Universal':
        return UniversalPerturbation(classifier=classifier, **kwargs)
    elif attack_name == 'HopSkipJump':
        return HopSkipJump(classifier=classifier, **kwargs)   #可增加新的攻击方式
    else:
        raise ValueError(f"Unsupported attack: {attack_name}")

def create_pytorch_classifier(model, input_shape, nb_classes):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0, 1)
    )

def generate_adversarial_samples(attack_name, classifier, x,y=None, **attack_params):
    attack = get_attack(attack_name, classifier, **attack_params)
    return attack.generate(x=x,y=y)