# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural networks used for training and evaluation.
"""

import torch
import torch.nn as nn
from itertools import chain


class ConvNet(nn.Module):
    # Based on Conv_Cifar10_32x64x64 from
    # https://github.com/google/learned_optimization/blob/main/learned_optimization/tasks/fixed/conv.py#L124

    def __init__(self, in_channels=3, ks=3, hid=32, activ=nn.ReLU, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hid, kernel_size=ks, stride=2),
            activ(),
            nn.Conv2d(in_channels=hid, out_channels=hid * 2, kernel_size=ks, stride=1, padding='same'),
            activ(),
            nn.Conv2d(in_channels=hid * 2, out_channels=hid * 2, kernel_size=ks, stride=1, padding='same'),
            activ(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hid * 2, num_classes)
        )

    def forward(self, x):
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, in_channels=1, im_size=28, hid=20, activ=nn.ReLU, num_classes=10):
        super().__init__()
        self.hid = (hid,) if not isinstance(hid, (tuple, list)) else hid
        self.fc = nn.Sequential(
            *chain.from_iterable(
                [
                    [nn.Linear(in_channels * im_size**2 if i == 0 else self.hid[i - 1], h), activ()]
                    for i, h in enumerate(self.hid)
                ]
            ),
            nn.Linear(self.hid[-1], num_classes),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(len(x), -1)
        return self.fc(x)
