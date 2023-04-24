import torch.nn as nn
import torch
import torchvision.models as models


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        distance = torch.abs(output1 - output2)
        output = self.fc2(distance)
        return output

    def encode(self, x):
        return self.forward_once(x)