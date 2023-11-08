import path
import sys
path = path.Path().parent.abspath()
sys.path.append(path)

from torch import nn
from utils.alignment import *

class VGG(nn.Module):
    def __init__(self, block_config, num_classes):
        super(VGG, self).__init__()
        self._load_layers(block_config)
        self.layers.append(VGGFc(512, 512, flatten=True))
        self.layers.append(VGGFc(512, 512))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def _load_layers(self, config):
        self.layers = nn.ModuleList()
        in_channel = 3
        for block in config:
            for index, out_channel in enumerate(block):
                self.layers.append(VGGConv(in_channel, out_channel, pooling=(index == len(block) - 1)))
                in_channel = out_channel


class VGGConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, pooling=False):
        super(VGGConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.pooling = pooling
        if pooling:
            self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.pooling:
            x = self.pool(x)
        return x
    
    def align(self, indices):
        align_neuron(self.conv, indices)
        align_normalize(self.bn, indices)
    
class VGGFc(nn.Module):
    def __init__(self, in_channel, out_channel, flatten=False):
        super(VGGFc, self).__init__()

        self.flatten = flatten
        self.fc = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.flatten:
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
    def align(self, indices):
        align_neuron(self.fc, indices)
        align_normalize(self.bn, indices)

class VGG16(VGG):
    block_config = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]

    def __init__(self, num_classes=10):
        super(VGG16, self).__init__(self.block_config, num_classes)