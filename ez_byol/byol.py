import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import copy
from torchvision import transforms as T
from rich import print
from rich.progress import track

from .networks import MLP, Encoder
from .utils import set_grad, get_simclr_augments, BYOLLoss

class BYOLOnlineNetwork(nn.Module):
    def __init__(self, encoder='resnet50', projection_dim = 256):
        super().__init__()
        self.encoder = Encoder(name=encoder)
        self.projection_head = MLP(2048, projection_dim)
        self.prediction_head = MLP(projection_dim, projection_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.projection_head(x)
        x = self.prediction_head(x)

        return x

class BYOLNetwork(nn.Module):
    def __init__(self, encoder_type = 'resnet50', momentum=0.99):
        self.beta = momentum
        self.online = BYOLOnlineNetwork(encoder=encoder_type)
        self.teacher = self._get_teacher(self.online)
        # TODO: dont hardcode image resize size, current assume CIFAR-10
        self.augs1, self.augs2 = get_simclr_augments(20), get_simclr_augments(20)
        
    def _get_teacher(self, online):
        teacher_encoder = copy.deepcopy(online.encoder)
        teacher_projector = copy.deepcopy(online.projector)
        set_grad(teacher_encoder, False)
        set_grad(teacher_projector, False)
        return nn.Sequential(teacher_encoder, nn.Flatten(), teacher_projector)

    # EMA update of teacher parameters
    def update_teacher(self):
        for online_param, teacher_param in zip(self.online.parameters(), self.teacher.parameters()):
            teacher_param.data = teacher_param.data * self.beta + online_param.data * (1-self.beta)

    def forward(self, x):
        x1 = self.augs1(x)
        x2 = self.augs2(x)
        online_pred1 = self.online(x1)
        online_pred2 = self.online(x2)

        with torch.no_grad():
            teacher_pred1 = self.teacher(x1)
            teacher_pred2 = self.teacher(x2)
        
        # Return symmetric BYOL loss
        return (BYOLLoss(online_pred1, teacher_pred2) + BYOLLoss(online_pred2, teacher_pred1)).mean()
        
    # Yields the encoder
    def get_encoder(self):
        return self.online.encoder