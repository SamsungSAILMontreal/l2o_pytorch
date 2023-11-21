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
from tasks import TASKS, TEST_SEEDS, trainloader_mapping, testloader_mapping
from config import init_config, process


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='l2o training')
    args = init_config(parser, steps=1000, inner_steps=100)

    first_task = args.train_tasks[0]
    dset = TASKS[first_task]['dataset']
    preproc_str = '' if args.no_preprocess else '_preproc'
    save_dir = (f'results/'
                f'l2o_{dset}_{first_task}_{args.opt}_lr{args.lr:.6f}_wd{args.wd:.6f}_b{args.batch_size}_'
                f'mom{args.momentum:.2f}_hid{args.hid}_layers{args.layers}_'
                f'iters{args.steps}_innersteps{args.inner_steps}{preproc_str}_seed{args.seed}')
    print('save_dir: %s\n' % save_dir)

    last_ckpt = os.path.join(save_dir, 'step_%d.pt' % args.steps)
    if os.path.exists(last_ckpt):
        raise ValueError('Already trained', last_ckpt)

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
    cfg = None
    train_loaders = {}
    train_iters = {}
    test_loaders = {}
    outer_steps_count = 0
    inner_steps_count = 0
    st = time.time()
    time_count = 0
    test_acc = []
    best_acc = 0
    print('\nTraining MetaOpt with %d params...' % sum([p.numel() for p in metaopt.parameters()]))
    while outer_steps_count < args.steps:

        metaopt.train()

        if cfg is None:
            # new task
            seed_everything(outer_steps_count)
            cfg = TASKS[np.random.choice(args.train_tasks)]

        dset = cfg['dataset']
        if dset not in train_loaders:
            seed_everything(outer_steps_count)  # to make sure dataloaders are different each time
            train_loaders[dset] = trainloader_mapping[dset](batch_size=args.batch_size)
            test_loaders[dset] = testloader_mapping[dset]()

        if dset not in train_iters:
            train_iters[dset] = iter(train_loaders[dset])  # create a new iterator

        try:
            data, target = next(train_iters[dset])
        except StopIteration:
            train_iters[dset] = iter(train_loaders[dset])
            data, target = next(train_iters[dset])

        data, target = data.to(args.device), target.to(args.device)

        if model is None:
            model = eval(cfg['net_cls'])(**cfg['net_args']).to(args.device).train()
            momentum = None
            inner_steps_count = 0

        loss_inner = F.cross_entropy(model(data), target)
        loss_inner.backward()

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
            test_acc_, test_loss_ = test_model(model, args.device, test_loaders[dset])

            scheduler.step()
            model = None  # to reset the model/initial weights
            cfg = None  # to let choose potentially different training tasks

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

            print('\nEval MetaOpt, task:', TASKS[first_task])
            acc = []
            for seed in TEST_SEEDS:
                acc.append(eval_opt(MetaOpt, TASKS[first_task], args.device, seed,
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
            time_count += 1
    print('best metaopt ckpt at outer step %d achieving test acc: %.3f' % (
        test_acc[np.argmax([x[1] for x in test_acc])]))
    print('done!', datetime.today())
