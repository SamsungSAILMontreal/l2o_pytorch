# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Training an MLP-based optimizer with default hyperparameters:

    python l2o_train.py

"""

import os
import argparse
import numpy as np
import time
import torch
import torchvision.models
import models.vision
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.metaopt import MetaOpt
from l2o_eval import eval_opt, seed_everything, test_model
from tasks import *
from config import init_config, process


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='l2o training')
    args = init_config(parser, steps=1000, inner_steps=100)

    save_dir = ('results/'
                'l2o_{}_{}_{}_lr{:.6f}_wd{:.6f}_mom{:.2f}_hid{}_layers{}_iters{}_innersteps{}{}_seed{}').format(
        TEST_TASKS[args.train_tasks[0]]['dataset'],
        args.train_tasks[0],
        args.opt,
        args.lr,
        args.wd,
        args.momentum,
        args.hid,
        args.layers,
        args.steps,
        args.inner_steps,
        '' if args.no_preprocess else '_preproc',
        args.seed)
    print('save_dir: %s\n' % save_dir)

    if os.path.exists(os.path.join(save_dir, 'step_%d.pt' % args.steps)):
        raise ValueError('Already trained', os.path.join(save_dir, 'step_%d.pt' % args.steps))

    seed_everything(args.seed)

    metaopt_cfg = dict(hid=[args.hid] * args.layers,
                       momentum=args.momentum,
                       preprocess=not args.no_preprocess)
    print('metaopt_cfg', metaopt_cfg)
    metaopt = MetaOpt(**metaopt_cfg).to(args.device).train()
    print(metaopt)

    if args.opt == 'adam':
        opt_fn = Adam
    elif args.opt == 'adamw':
        opt_fn = AdamW
    else:
        raise NotImplementedError(f'unknown optimizer {args.opt}')

    optimizer = opt_fn(metaopt.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps)

    model = None
    momentum = None
    train_cfg = None
    train_loaders = {}
    outer_steps_count = 0
    inner_steps_count = 0
    st = time.time()
    time_count = 0
    test_acc = []
    best_acc = 0
    print('\nTraining MetaOpt with %d params...' % sum([p.numel() for p in metaopt.parameters()]))
    while outer_steps_count < args.steps:

        metaopt.train()

        if train_cfg is None:
            seed_everything(outer_steps_count)
            train_cfg = TEST_TASKS[np.random.choice(args.train_tasks)]

        if train_cfg['dataset'] not in train_loaders or train_loaders[train_cfg['dataset']] is None:
            seed_everything(outer_steps_count)  # to make sure dataloaders are different each time
            train_loaders[train_cfg['dataset']] = iter(trainloader_mapping[train_cfg['dataset']](
                batch_size=train_cfg['batch_size']))

        try:
            data, target = next(train_loaders[train_cfg['dataset']])
        except StopIteration:
            train_loaders[train_cfg['dataset']] = iter(trainloader_mapping[train_cfg['dataset']](
                batch_size=train_cfg['batch_size']))
            data, target = next(train_loaders[train_cfg['dataset']])

        data, target = data.to(args.device), target.to(args.device)

        if model is None:
            model = eval(train_cfg['net_cls'])(**train_cfg['net_args']).to(args.device).train()
            momentum = None
            inner_steps_count = 0

        loss_inner = F.cross_entropy(model(data), target)
        loss_inner.backward(retain_graph=False)

        param_upd, momentum = metaopt(model.parameters(), momentum=momentum)  # upd momentum state and get param deltas
        metaopt.set_model_params(param_upd, model=model, keep_grad=True)

        # use same data for now, but can be a different batch/loader
        loss_outer = F.cross_entropy(model(data), target)
        loss_outer.backward(retain_graph=True)  # retain graph because we backprop multiple times through metaopt

        # detach to avoid backpropagating through the params in the next steps
        # backprop through the hidden states (if it exists) is still happening though
        for p in model.parameters():
            p.detach_()
            p.requires_grad = True  # same as in set_model_params()

        outer_upd = (inner_steps_count + 1) % args.inner_steps == 0
        if outer_upd:

            optimizer.step()  # make a gradient step based on a sequence of inner_steps predictions
            optimizer.zero_grad()

            # test meta-optimized network to make sure it is improving
            test_acc_, test_loss_ = test_model(model, args.device, testloader_mapping[train_cfg['dataset']]())

            scheduler.step()
            model = None  # to reset the model/initial weights
            train_cfg = None  # to let choose potentially different training tasks

            print('Training MetaOpt: '
                  'outer step={:05d}/{:05d}, '
                  'inner step={:05d}/{:05d}, lr={:.3e}, '
                  'loss inner/outer={:.3f}/{:.3f}, test loss={:.3f}, test acc={:.3f}, '
                  'speed: {:.2f} sec/outer step, '
                  'mem ram/gpu: {:.2f}/{:.2f}G'.format(outer_steps_count + 1,
                                                       args.steps,
                                                       inner_steps_count + 1,
                                                       args.inner_steps,
                                                       scheduler.get_last_lr()[0],
                                                       loss_inner.item(),
                                                       loss_outer.item(),
                                                       test_loss_,
                                                       test_acc_,
                                                       (time.time() - st) / (time_count + 1),
                                                       process.memory_info().rss / 10 ** 9,
                                                       -1 if args.device == 'cpu' else (
                                                               torch.cuda.memory_reserved(0) /
                                                               10 ** 9)), flush=True)

        if ((outer_steps_count + 1) % args.inner_steps == 0 or (outer_steps_count + 1) == args.steps) and outer_upd:
            checkpoint = {
                'model_state_dict': metaopt.state_dict(),
                'step': outer_steps_count + 1,
                'config': args,
                'metaopt_cfg': metaopt_cfg
            }

            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                except Exception as e:
                    print('error in creating the directory %s' % save_dir, e)

            file_path = os.path.join(save_dir, f'step_{outer_steps_count + 1}.pt')
            try:
                torch.save(checkpoint, file_path)
                print('saving the checkpoint done to %s' % file_path)
            except Exception as e:
                print('error in saving the checkpoint to %s' % file_path, e)

            print('\nEval MetaOpt, task:', TEST_TASKS[args.train_tasks[0]])
            acc = []
            for seed in TEST_SEEDS:
                acc.append(eval_opt(MetaOpt, TEST_TASKS[args.train_tasks[0]], args.device, seed,
                                    print_interval=100,
                                    metaopt_cfg=metaopt_cfg,
                                    metaopt_state=metaopt.state_dict()))
            print('test acc for %d seeds: %.3f +- %.3f\n' % (len(acc), np.mean(acc), np.std(acc)))
            test_acc.append((outer_steps_count + 1, np.mean(acc)))

            if np.mean(acc) > best_acc:
                file_path = os.path.join(save_dir, 'best.pt')
                try:
                    torch.save(checkpoint, file_path)
                    print('saving the best checkpoint done to %s' % file_path)
                except Exception as e:
                    print('error in saving the checkpoint to %s' % file_path, e)
                best_acc = np.mean(acc)

            st = time.time()  # reset time
            time_count = 0  # reset time count

        inner_steps_count += 1
        if outer_upd:
            outer_steps_count += 1
    print('best metaopt ckpt at outer step %d achieving test acc: %.3f' % (
        test_acc[np.argmax([x[1] for x in test_acc])]))
    print('done!', datetime.today())
