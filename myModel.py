import torch
import torch.nn as nn

import os.path
from collections import OrderedDict
import numpy as np
from copy import deepcopy


## Define AlexNet model
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def get_model(model):
    return deepcopy(model.state_dict())


class AlexNet(nn.Module):
    def __init__(self, taskcla):
        super(AlexNet, self).__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])

        self.taskcla = taskcla
        self.fc3 = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048, n, bias=False))

    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1'] = x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2'] = x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3'] = x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        self.act['fc1'] = x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2'] = x
        feature_x = x
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        # feature_x = x
        y = []
        for t, i in self.taskcla:
            y.append(self.fc3[t](x))

        return y, feature_x


