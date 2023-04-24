import torch
import torchvision.models as models
from torch import nn


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        try:
            encoder = torch.load('trained-models/ResNet50.pt')
        except FileNotFoundError:
            encoder = models.resnet50(pretrained=True)

        self.encoder = encoder
        # Replace the final classification layer with an identity mapping
        self.encoder.fc = nn.Identity()

        # Projection head to map encoded features to a 512-dimensional vector
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        x = x.reshape(-1, 512, 1, 1)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return self.projection_head(x)

    def decode(self, x):
        x = x.reshape(-1, 512, 1, 1)
        return self.decoder(x)
