import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
import torch.nn as nn

# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.backbone = models.resnet50()
        # MY pre-trained model
        # self.backbone.load_state_dict(torch.load('./resnet50_improved_ver2.pt'))
        # TA pre-trained model
        # self.backbone.load_state_dict(torch.load('../hw1_data/p2_data/pretrain_model_SL.pt'))

        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 65)
        )

    def forward(self, x):

        embed = self.backbone(x)
        out = self.classifier(embed)
        return out