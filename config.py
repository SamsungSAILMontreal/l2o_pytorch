# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Config.
"""

import subprocess
import psutil
import os
import platform
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.models

process = psutil.Process(os.getpid())  # for ram calculation


def init_config(parser, steps=1000, inner_steps=None):
    """
    Initializes the config that is shared for both training and evaluation.
    Additional arguments can be added to the parser.
    :param parser: ArgumentParser
    :param steps: default value of the number of outer steps
    :param inner_steps: default value of the number of inner steps
    :return: args
    """

    parser.add_argument('-t', '--train_tasks', type=str, default='0')  # can potentially train on a combination tasks
    parser.add_argument('-s', '--steps', type=int, default=steps, help='number of outer steps')
    parser.add_argument('-i', '--inner_steps', type=int,
                        default=parser.parse_known_args()[0].steps if inner_steps is None else inner_steps,
                        help='number of inner/unroll steps')
    parser.add_argument('-H', '--hid', type=int, default=32, help='hidden units in the learned optimizer')
    parser.add_argument('-l', '--layers', type=int, default=2, help='number of layers in the learned optimizer')
    parser.add_argument('-M', '--momentum', type=int, default=5,
                        help='momentum features in the learned optimizer:'
                             '0 means no momentum, '
                             '>0 means multiscale momentum features from the paper'
                             '"Understanding and correcting pathologies in the training of learned optimizers"'
                             'https://arxiv.org/abs/1810.10180')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--seed', type=int, default=0, help='random seed defining initialization and data sampling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_preprocess', action='store_true',
                        help='do not preprocess features as in the paper'
                             '"Learning to learn by gradient descent by gradient descent '
                             '(https://arxiv.org/abs/1606.04474)"')

    print('\nEnvironment:')
    env = {}
    try:
        # print git commit to ease code reproducibility
        # copied from https://github.com/facebookresearch/ppuda
        env['git commit'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(e, flush=True)
        env['git commit'] = 'no git'
    env['hostname'] = platform.node()
    env['torch'] = torch.__version__
    env['torchvision'] = torchvision.__version__
    if env['torch'][0] in ['0', '1'] and not env['torch'].startswith('1.9') and not env['torch'].startswith('1.1'):
        print('WARNING: pytorch >= 1.9 is strongly recommended for this repo!')

    env['cuda available'] = torch.cuda.is_available()
    env['cudnn enabled'] = cudnn.enabled
    env['cuda version'] = torch.version.cuda
    env['start time'] = time.strftime('%Y%m%d-%H%M%S')
    for x, y in env.items():
        print('{:20s}: {}'.format(x[:20], y))

    args = parser.parse_args()
    args.train_tasks = list(map(int, args.train_tasks.split(',')))

    def print_args(args_, name):
        print('\n%s:' % name)
        args_ = vars(args_)
        for a in sorted(args_.keys()):
            print('{:20s}: {}'.format(a[:20], args_[a]))
        print('\n', flush=True)

    print_args(args, 'Script Arguments')

    return args
