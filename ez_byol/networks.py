import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision

# MLP used in BYOL projection and prediction heads,
# default options are as per the paper
def MLP(input_sz,  projection_sz=256, hidden_sz=4096):
    return nn.Sequential(
        nn.Linear(input_sz, hidden_sz),
        nn.BatchNorm1d(hidden_sz),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_sz, projection_sz)
    )

def Encoder(name='resnet50', pretrained=False):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
    if name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
    if name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
    # strip head layer
    return nn.Sequential(*list(model.children())[:-1])

