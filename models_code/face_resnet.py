import torch
import torch.nn as nn
from torchvision import models


class FaceResNet(nn.Module):
    def __init__(self):
        super(FaceResNet, self).__init__()
        try:
            resnet = torch.load('trained-models/ResNet50.pt')
        except FileNotFoundError:
            resnet = models.resnet50(pretrained=True)

        self.resnet50 = resnet
        # Replace the final classification layer with an identity mapping
        self.resnet50.fc = nn.Identity()

        # Projection head to map encoded features to a 512-dimensional vector
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.resnet50(x)
        x = self.projection_head(x)
        return x

    def encode(self, x):
        return self(x)

