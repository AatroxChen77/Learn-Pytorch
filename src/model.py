import torch.nn as nn
import torchvision.models as models

def build_model(num_classes: int = 10, pretrained: bool = False):
    net = models.resnet18(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net
