# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Learned optimizer (MetaOpt) based on MLP.
"""

import torch
import torch.nn as nn
from itertools import chain
from typing import Generator


class MetaOpt(nn.Module):
    """
    MLP-based optimizer.
    """
    def __init__(self,
                 in_features=2,
                 hid=(32, 32),
                 activ=nn.ReLU,
                 momentum=5,
                 preprocess=True,
                 lambda12=0.01,
                 parameters=None,
                 ):
        """
        Constructs an MLP optimizer.
        :param in_features: 2 by default (params and grads)
        :param hid: hidden units in the MLP (int or tuple/list)
        :param activ: activation layer
        :param momentum: number of momentum features (>=0)
        :param preprocess: preprocess param, grad and momentum features based on
        "Learning to learn by gradient descent by gradient descent" https://arxiv.org/abs/1606.04474
        :param lambda12: output scale for predicting two outputs (scale and magnitude) based on
        https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/no_dependency_learned_optimizer.ipynb
        """
        super(MetaOpt, self).__init__()
        self.hid = (hid,) if not isinstance(hid, (tuple, list)) else hid
        self.preprocess = preprocess
        self.lambda12 = lambda12
        if parameters is not None:
            if isinstance(parameters, Generator):
                parameters = list(parameters)
            self.__dict__['params'] = parameters

        if momentum > 0:
            in_features += momentum
            # Momentum scales [0.9, 0.99, 0.999, 0.9999] based on Sec D.1 in https://arxiv.org/pdf/1810.10180.pdf
            mom = torch.tensor([float('0.' + '9' * m) for m in range(1, min(5, momentum + 1))], requires_grad=False)
            if momentum >= 5:
                mom = torch.cat((torch.tensor([0.5]), mom), dim=0)  # [0.5, 0.9, 0.99, 0.999, 0.9999]
            if momentum > 5:
                mom = torch.cat((torch.linspace(0, 0.9, momentum - len(mom) + 2)[1:-1], mom), dim=0)
            self.register_buffer('momentum', mom.view(1, -1))

            if parameters is not None:
                # initialize the momentum state that will be updated during training for each of the scale defined above
                self.register_buffer('state', torch.zeros(sum([p.numel() for p in parameters]),
                                                          max(1, self.momentum.shape[1])), persistent=False)
        else:
            self.momentum = None

        self.fc = nn.Sequential(
            *chain.from_iterable(
                [
                    [nn.Linear((in_features * (2 if preprocess else 1))
                               if i == 0 else self.hid[i - 1], h),
                     activ()]
                    for i, h in enumerate(self.hid)
                ]
            ),
            nn.Linear(self.hid[-1], 2),  # predict two outputs: scale and magnitude
        )

    def zero_grad(self, set_to_none=True):
        if self.params is None:
            return
        for p in self.params:
            p.detach_()  # to detach the computational graph from the previous steps
            if set_to_none:
                p.grad = None
            else:
                p.grad.fill_(0.)
            p.requires_grad = True  # necessary to compute p.grad in the next iter

    def step(self):
        with torch.set_grad_enabled(False):
            predicted_upd, self.state = self.forward(self.params, momentum=self.state)
            self.set_model_params(predicted_upd, keep_grad=False)

    def set_model_params(self, param_upd, model=None, keep_grad=False):
        """
        Sets model parameters to the values in param_upd.
        :param param_upd: parameter updates
        :param model: neural net (nn.Module derivative)
        :param keep_grad: gradients are backpropagated through params
        :return: model with updated parameters
        """

        if model is None:
            assert not keep_grad, 'keep_grad=True is not supported for model=None (meta-testing mode)'
            offset = 0
            for p in self.params:
                n = p.numel()
                p.data = p + param_upd[offset: offset + n].data.view_as(p)
                offset += n
        else:
            # meta-training mode
            assert keep_grad, 'keep_grad=False is not supported in the meta-training mode'
            param_names = set()
            for name, _ in model.named_parameters():
                param_names.add(name.split('.')[-1])

            offset = 0
            for module_name, module in model.named_modules():
                for name, p in module.named_parameters(recurse=False):
                    key = name.split('.')[-1]
                    if key in param_names:
                        # p_name = name if len(module_name) == 0 else '.'.join(
                        #     (module_name, name))  # as model.named_parameters()
                        n = p.numel()
                        tensor = p + param_upd[offset: offset + n].view_as(p)

                        # setting params this way is inspired by https://github.com/facebookresearch/ppuda
                        module.__dict__[key] = tensor  # set the value avoiding the internal logic of PyTorch
                        module._parameters[key] = tensor  # to that model.parameters() returns predicted tensors

                        offset += n
        assert offset == param_upd.numel(), ('not all params are set', offset, param_upd.numel())

    def forward(self, parameters, momentum=None):
        """
        Computes a step based on the current parameters.
        :param parameters: parameters() obtained via nn.Module().parameters()
        :param momentum: momentum state
        :return: parameter deltas
        """
        x = []
        for p in parameters:
            x.append(torch.stack((p.detach().flatten(), p.grad.data.flatten()), dim=-1))
            # if p.detach() is not used, the gradients are preserved for the entire sequence, which creates
            # a discrepancy between training and testing. To avoid such a discrepancy, one can keep the gradients
            # during meta-testing, but that is much more computationally expensive

        x = torch.cat(x, dim=0)  # params and grads
        assert x.dim() == 2, x.dim()

        # Compute momentum features
        if self.momentum is not None:
            if momentum is None:
                momentum = torch.zeros(len(x), max(1, self.momentum.shape[1])).to(x)
            momentum = self.momentum * momentum + x[:, 1:2]  # x[:, 1:2] is the grad
            x = torch.cat((x, momentum), dim=-1)

        if self.preprocess:
            x = preprocess_features(x)

        outs = self.fc(x)

        # slice out the last 2 elements
        scale = outs[:, 0]
        mag = outs[:, 1]
        # Compute a step as follows.
        return (scale * self.lambda12 * torch.exp(mag * self.lambda12)), momentum


def preprocess_features(x, p=10):
    # based on "Learning to learn by gradient descent by gradient descent" https://arxiv.org/abs/1606.04474 (Appendix A)
    # WARNING: the gradient might be unstable/undefined because of the sign function
    assert x.dim() == 2, x.dim()
    n_feat = x.shape[1]
    x_new = torch.zeros(len(x), n_feat * 2).to(x)
    for dim in range(n_feat):
        mask = torch.abs(x[:, dim]) >= torch.exp(-torch.tensor(p)).to(x)
        ind_small, ind_large = torch.nonzero(~mask).flatten(), torch.nonzero(mask).flatten()
        x_new[ind_small, dim * 2] = torch.zeros(len(ind_small)).to(x) - 1
        x_new[ind_small, dim * 2 + 1] = torch.exp(torch.tensor(p)).to(x) * x[ind_small, dim]
        x_new[ind_large, dim * 2] = torch.log(torch.abs(x[ind_large, dim])) / p
        x_new[ind_large, dim * 2 + 1] = torch.sign(x[ind_large, dim])

    return x_new
