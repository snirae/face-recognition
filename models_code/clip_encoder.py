import torch
from torch import nn
import open_clip


class CLIPEncoder(nn.Module):
    def __init__(self):
        super(CLIPEncoder, self).__init__()
        try:
            model = torch.load('trained-models/CLIP-visual.pt')
            preprocess = None
        except FileNotFoundError:
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu',
                                                                         pretrained='laion400m_e32')
            model = model.visual

        self.model = model
        self.preprocess = preprocess

    def forward(self, x):
        return self.model(x)

    def encode(self, x):
        return self.model(x)
