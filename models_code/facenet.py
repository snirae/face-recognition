import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


###################################################################

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = ConvBlock(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4_2 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x3 = self.conv3_1(x)
        x3 = self.conv3_2(x3)
        x4 = self.conv4_1(x)
        x4 = self.conv4_2(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


###################################################################

class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionBlock(192, 256)
        self.inception2 = InceptionBlock(256 * 4, 480)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = InceptionBlock(480 * 4, 512)
        self.inception4 = InceptionBlock(512 * 4, 512)
        self.inception5 = InceptionBlock(512 * 4, 512)
        self.inception6 = InceptionBlock(512 * 4, 528)
        self.inception7 = InceptionBlock(528 * 4, 528)
        self.inception8 = InceptionBlock(528 * 4, 832)
        self.inception9 = InceptionBlock(832 * 4, 832)
        self.inception10 = InceptionBlock(832 * 4, 1024)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024 * 4, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool3(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.inception10(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode(self, x):
        return self(x)
