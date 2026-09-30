"""The classifier that the original code trains jointly with the generator.

``archive/gan_attack*.py`` feed the generated images to ``ShakePyramidNet(depth=110, alpha=270)`` and add its cross
entropy on the labels of the scrambled images to the generator loss (the classifier is updated by the generator's
optimiser). The paper's Eq. 1 lists only the adversarial loss; :class:`~siagan.SIAGAN` keeps the term of the code
(``classifier="shakepyramidnet"``, ``lambda_cls=1``) and ``classifier=None`` gives the loss of the paper.

Port of ``archive/no_adaptation_network.py`` and ``archive/shakedrop.py`` (ShakeDrop, Yamada et al. 2018, after
github.com/owruby/shake-drop_pytorch). Changes that do not alter the computation: the gate of ShakeDrop is drawn on
the CPU and the random factors on the input's device (the original used ``torch.cuda.FloatTensor``), so the network
runs on CPU, CUDA and Apple-silicon GPUs; the global average pooling is ``adaptive_avg_pool2d`` (= ``avg_pool2d(h, 8)``
for 32x32 inputs).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ShakeDrop", "ShakeBasicBlock", "ShakePyramidNet"]


class _ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=(-1.0, 1.0)):
        if training:
            gate = torch.empty(1).bernoulli_(1.0 - p_drop)  # one gate for the whole mini-batch
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.empty(x.size(0), device=x.device, dtype=x.dtype).uniform_(*alpha_range)
                return alpha.view(-1, 1, 1, 1).expand_as(x) * x
            return x
        return (1.0 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.empty(grad_output.size(0), device=grad_output.device, dtype=grad_output.dtype).uniform_(0, 1)
            return beta.view(-1, 1, 1, 1).expand_as(grad_output) * grad_output, None, None, None
        return grad_output, None, None, None


class ShakeDrop(nn.Module):
    """ShakeDrop: in training, with probability ``p_drop`` the branch is scaled by ``alpha ~ U(-1, 1)`` in the forward
    pass and its gradient by ``beta ~ U(0, 1)``; at test time the branch is scaled by ``1 - p_drop``."""

    def __init__(self, p_drop: float = 0.5, alpha_range=(-1.0, 1.0)):
        super().__init__()
        self.p_drop, self.alpha_range = float(p_drop), tuple(alpha_range)

    def forward(self, x):
        return _ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


class ShakeBasicBlock(nn.Module):
    """Pre-activation basic block with ShakeDrop; the shortcut is the identity (2x2 average pooling when the block
    downsamples) with zero-padded channels."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, p_shakedrop: float = 1.0):
        super().__init__()
        self.downsampled = stride == 2
        self.branch = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.shortcut = nn.AvgPool2d(2) if self.downsampled else None
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        h = self.shake_drop(self.branch(x))
        h0 = self.shortcut(x) if self.downsampled else x
        pad = h.new_zeros(h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3))
        return h + torch.cat([h0, pad], dim=1)


class ShakePyramidNet(nn.Module):
    """ShakeDrop PyramidNet for 32x32 images (``depth = 6 n + 2``; the code uses depth 110, alpha 270)."""

    def __init__(self, depth: int = 110, alpha: int = 270, num_classes: int = 10):
        super().__init__()
        if depth < 8 or (depth - 2) % 6:
            raise ValueError("depth must be 6 n + 2 with n >= 1 (e.g. 8, 20, 110)")
        n_units = (depth - 2) // 6
        in_chs = [16] + [16 + math.ceil((alpha / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        self.in_chs, self.u_idx = in_chs, 0
        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]

        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        self.layer1 = self._make_layer(n_units, 1)
        self.layer2 = self._make_layer(n_units, 2)
        self.layer3 = self._make_layer(n_units, 2)
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], num_classes)

        for m in self.modules():  # initialisation of the original code
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, n_units: int, stride: int) -> nn.Sequential:
        layers = []
        for _ in range(n_units):
            layers.append(ShakeBasicBlock(self.in_chs[self.u_idx], self.in_chs[self.u_idx + 1], stride,
                                          self.ps_shakedrop[self.u_idx]))
            self.u_idx, stride = self.u_idx + 1, 1
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.bn_in(self.c_in(x))
        h = self.layer3(self.layer2(self.layer1(h)))
        h = F.relu(self.bn_out(h))
        return self.fc_out(F.adaptive_avg_pool2d(h, 1).flatten(1))
