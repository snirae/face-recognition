import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityNet(nn.Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()
        self.input_size = 512 * 2
        self.hidden_size = 1024

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x