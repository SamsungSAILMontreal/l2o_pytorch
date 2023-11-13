# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Evaluate a trained MLP-based optimizer on a given task (see tasks.py for the list of tasks):

    python l2o_eval.py --ckpt path_to_checkpoint -t task_index

"""

import numpy as np
import time
import argparse
import random
from datetime import datetime
import torch
import torch.nn.functional as F
import torchvision.models
import models.vision
from functools import partial
from tasks import trainloader_mapping, testloader_mapping, TEST_TASKS, TEST_SEEDS
from config import init_config, process
from models.metaopt import MetaOpt


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_model(model, device, test_loader):
    """
    Tests the model on a test set.
    :param model: nn.Module based neural net
    :param device: cpu or cuda
    :param test_loader: DataLoader
    :return: test accuracy and loss
    """
    model.eval().to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction='sum'
            ).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)
    return test_acc, test_loss


def eval_opt(opt_cls, test_cfg, device, seed, print_interval=100, max_iters=1000,
             metaopt_cfg=None, metaopt_state=None):
    """
    Evaluates an SGD based optimizer or a learned optimizer on a given task.
    :param opt_cls: optimizer class such as torch.optim.SGD/Adam/AdamW or MetaOpt
    :param test_cfg: task config from tasks.py
    :param device: cpu or cuda
    :param seed: test seed from tasks.py
    :param print_interval: print interval of loss/acc
    :param max_iters: max number of training iterations
    :param metaopt_cfg: config of the learned optimizer (from the checkpoint)
    :param metaopt_state: state of the learned optimizer (from the checkpoint)
    :return: test accuracy
    """
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    net = eval(test_cfg['net_cls'])(**test_cfg['net_args']).to(device).train()
    print('Training %s net with %d parameters' % (test_cfg['name'], sum([p.numel() for p in net.parameters()])))

    if metaopt_state is not None:
        opt = opt_cls(**metaopt_cfg, parameters=net.parameters()).to(device).eval()
        opt.load_state_dict(metaopt_state)
    else:
        opt = opt_cls(net.parameters())

    t = time.time()
    train_loader = trainloader_mapping[test_cfg['dataset']](batch_size=test_cfg['batch_size'])
    test_loader = testloader_mapping[test_cfg['dataset']]()
    epochs = int(np.ceil(max_iters / len(train_loader)))
    step = 0
    for epoch in range(epochs):
        for _, (x, y) in enumerate(train_loader):
            output = net(x.to(device))
            y = y.to(device)
            loss = F.cross_entropy(output, y)
            loss.backward()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc = pred.eq(y.view_as(pred)).sum() / len(y)

            opt.step()
            opt.zero_grad()

            if (step + 1) % print_interval == 0 or step == max_iters - 1:
                test_acc_, test_loss_ = test_model(net, device, test_loader)
                print('Training {} net: seed={}, step={:05d}/{:05d}, train loss={:.3f}, train acc={:.3f}, '
                      'test loss={:.3f}, test acc={:.3f}, '
                      'speed: {:.2f} s/b, mem ram/gpu: {:.2f}/{:.2f}G'.format(test_cfg['name'],
                                                                              seed,
                                                                              step + 1,
                                                                              max_iters,
                                                                              loss.item(),
                                                                              acc.item(),
                                                                              test_loss_,
                                                                              test_acc_,
                                                                              (time.time() - t) / (step + 1),
                                                                              process.memory_info().rss / 10 ** 9,
                                                                              -1 if device == 'cpu' else (
                                                                                      torch.cuda.memory_reserved(0) /
                                                                                      10 ** 9)))
            step += 1
            if step >= max_iters:
                break
        if step >= max_iters:
            break

    test_acc_, test_loss_ = test_model(net, device, testloader_mapping[test_cfg['dataset']]())
    print('seed: {}, test loss: {:.3f}, test accuracy: {:.3f}\n'.format(seed,
                                                                        test_loss_,
                                                                        test_acc_))

    return test_acc_


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='l2o evaluation')
    parser.add_argument('-c', '--ckpt', type=str, default=None, help='path to the trained l2o checkpoint')
    args = init_config(parser, steps=1000, inner_steps=None)  # during eval, steps should equal inner_steps

    seed_everything(args.seed)

    if args.ckpt in [None, 'none']:

        opt_args = {}
        if args.opt == 'adam':
            opt_fn = torch.optim.Adam
        elif args.opt == 'adamw':
            opt_fn = torch.optim.AdamW
        elif args.opt == 'sgd':
            opt_fn = torch.optim.SGD
            opt_args = {'momentum': 0.9}
        else:
            raise NotImplementedError(f'unknown optimizer {args.opt}')
        metaopt = partial(opt_fn, lr=args.lr, weight_decay=args.wd, **opt_args)
        print(f'Using {args.opt}')
        metaopt_cfg = None
        metaopt_state = None
    else:
        state_dict = torch.load(args.ckpt, map_location=args.device)
        metaopt = MetaOpt
        metaopt_cfg = state_dict['metaopt_cfg']
        metaopt_state = state_dict['model_state_dict']
        print('MetaOpt with config %s' % str(metaopt_cfg),
              'loaded from step %d' % state_dict['step'])

    for task in args.train_tasks:
        cgf = TEST_TASKS[task]
        print('\nEval %s, task:' % str(metaopt), cgf)
        acc = []
        for seed in TEST_SEEDS:
            acc.append(eval_opt(metaopt, cgf, args.device, seed,
                                print_interval=1,
                                max_iters=args.steps,
                                metaopt_cfg=metaopt_cfg,
                                metaopt_state=metaopt_state))
        print('test acc for %d seeds: %.3f +- %.3f' % (len(acc), np.mean(acc), np.std(acc)))

    print('done!', datetime.today())
