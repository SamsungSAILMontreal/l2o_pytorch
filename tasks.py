# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training and evaluation tasks.
"""

import torch
from torchvision import datasets, transforms
from functools import partial


TASKS = [

        {
            'net_args': {'hid': 20, 'im_size': 28},
            'batch_size': 128,
            'net_cls': 'models.vision.MLP',
            'dataset': 'fashionmnist', 'name': 'mlp20_fashionmnist'
        },

        {
            'net_args': {'in_channels': 3},
            'batch_size': 128,
            'net_cls': 'models.vision.ConvNet',
            'dataset': 'cifar10',
            'name': 'convnet_32x64x64_cifar10'
            # Based on Conv_Cifar10_32x64x64 from
            # https://github.com/google/learned_optimization/blob/main/learned_optimization/tasks/fixed/conv.py#L124
        },

        {
            'net_args': {'hidden_dim': 192,
                         'mlp_dim': 192 * 4,
                         'num_layers': 12,
                         'num_heads': 3,
                         'patch_size': 2,
                         'num_classes': 10,
                         'image_size': 32,
                         'weights': None,
                         'progress': False},
            'batch_size': 512,
            'net_cls': 'torchvision.models.vision_transformer._vision_transformer',
            'dataset': 'cifar10',
            'name': 'vit_tiny_cifar10'
            # Based on ViT-Tiny from the paper "Training data-efficient image transformers
            # & distillation through attention" (https://arxiv.org/abs/2012.12877)
        }
    ]

TEST_SEEDS = [101, 102, 103, 104, 105]


mnist_normalize = ((0.1307,), (0.3081,))
cifar_normalize = ((0.49139968, 0.48215827, 0.44653124),
                   (0.24703233, 0.24348505, 0.26158768))


def get_loader(dataset, data_dir='./data', batch_size=None, train=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*(mnist_normalize if dataset.lower().find('mnist') >= 0 else cifar_normalize)),
        ]
    )
    loader = torch.utils.data.DataLoader(
        eval('datasets.%s(data_dir, train=train, download=True, transform=transform)' % dataset),
        pin_memory=torch.cuda.is_available(),
        num_workers=4,
        batch_size=(128 if batch_size is None else batch_size) if train else 1000,
        shuffle=train,
    )
    return loader


trainloader_mapping = {
    'fashionmnist': partial(get_loader, dataset='FashionMNIST', train=True),
    'cifar10': partial(get_loader, dataset='CIFAR10', train=True),
}

testloader_mapping = {
    'fashionmnist': partial(get_loader, dataset='FashionMNIST', train=False),
    'cifar10': partial(get_loader, dataset='CIFAR10', train=False),
}
