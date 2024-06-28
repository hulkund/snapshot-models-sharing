'''
    Model implementation.
    We'll be using a "simple" ResNet-18 for image classification here.

    2022 Benjamin Kellenberger
'''

import torch.nn as nn
from torchvision.models import resnet
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments


def CustomResNet18(num_classes):
    # Load pretrained model params
    model = resnet.resnet50(pretrained=True)
    # Replace the original classifier with a new Linear layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Ensure all params get updated during finetuning
    for param in model.parameters():
        param.requires_grad = True
    return model






