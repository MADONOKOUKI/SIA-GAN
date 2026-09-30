"""SIA-GAN networks (Sec. IV, Fig. 3-5, Tables 4-6).

* :class:`AdaptationNetwork` (Table 4, Fig. 4) - one spectrally normalised ``B x B`` convolution (stride ``B``,
  ``16 B^2`` outputs), BatchNorm and LeakyReLU(0.1) **per block**, then a pixel-shuffle layer to ``16 x 32 x 32``.
* :class:`FeatureDecoder` (Table 5, Fig. 5) - SN-convolution + BN + LeakyReLU, three residual blocks, SN-convolution
  + BN + LeakyReLU and a final SN-convolution with Tanh.
* :class:`Generator` = adaptation network + feature decoder (Fig. 3).
* :class:`Discriminator` (Table 6) - six spectrally normalised convolutions, self-attention (SAGAN), a convolution to
  1024 channels, self-attention and a linear layer on the flattened ``1024 x 4 x 4`` map.

Where the paper and the original code (``archive/generator.py``, ``archive/dcgan_spec_acgan.py``) differ, this
follows the code:

* the last decoder convolution has no BatchNorm (Table 5 lists one before the Tanh);
* the 3x3 discriminator convolution to 1024 channels has no activation (Table 6 lists a LeakyReLU);
* the discriminator's spectral normalisation divides the weights *in place* before every forward pass by the estimate
  of one power iteration that always starts from the same random vector (:class:`InplaceSpectralNorm`); the
  generator uses ``torch.nn.utils.spectral_norm``;
* the AC-GAN class heads of the code's discriminator are computed but never used by the losses and are left out.

The 64 per-block sub-networks of the original generator (Python lists of ``spectral_norm(Conv2d)`` and
``BatchNorm2d``, called in a double loop) are computed in one batched operation (:class:`BlockwiseConv`,
:class:`BlockwiseBatchNorm`); ``tests/test_parity_original.py`` checks that the outputs, the power iteration and the
BatchNorm statistics equal those of ``archive/generator.py``. Batching also makes the ``1 x 1`` "blocks" of PE and
random PE (1024 sub-networks, Sec. V) practical.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "BlockwiseConv",
    "BlockwiseBatchNorm",
    "AdaptationNetwork",
    "FeatureDecoder",
    "Generator",
    "InplaceSpectralNorm",
    "SelfAttention",
    "Discriminator",
]


# ------------------------------------------------------------------------------------------------------------------
# block-wise layers of the adaptation network
# ------------------------------------------------------------------------------------------------------------------
class BlockwiseConv(nn.Module):
    """An independent ``Conv2d(C, out, kernel_size=B, stride=B, bias=False)`` for every block position.

    ``(N, C, H, W) -> (N, out, H / B, W / B)`` with ``out[:, :, i, j] = conv_{i * nbw + j}(block (i, j))``. Every
    block has its own spectral normalisation with the semantics of ``torch.nn.utils.spectral_norm`` (``u``/``v``
    buffers, one power iteration per training forward pass, none in evaluation) and the default ``Conv2d``
    initialisation, as the 64 ``nn.utils.spectral_norm(nn.Conv2d(...))`` modules of ``archive/generator.py``.
    """

    def __init__(self, in_channels: int, out_channels: int, block_size: int, num_blocks: int, eps: float = 1e-12):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.block_size, self.num_blocks, self.eps = block_size, num_blocks, eps
        fan_in = in_channels * block_size * block_size
        self.weight = nn.Parameter(torch.empty(num_blocks, out_channels, in_channels, block_size, block_size))
        with torch.no_grad():  # nn.Conv2d default: kaiming_uniform_(a=sqrt(5)) = U(-1/sqrt(fan_in), 1/sqrt(fan_in))
            bound = 1.0 / math.sqrt(fan_in)
            self.weight.uniform_(-bound, bound)
        self.register_buffer("weight_u", F.normalize(torch.randn(num_blocks, out_channels), dim=1, eps=eps))
        self.register_buffer("weight_v", F.normalize(torch.randn(num_blocks, fan_in), dim=1, eps=eps))

    def normalized_weight(self) -> torch.Tensor:
        """The ``(nb, out, C B B)`` weight matrices divided by their spectral-norm estimates."""
        w = self.weight.reshape(self.num_blocks, self.out_channels, -1)
        u, v = self.weight_u, self.weight_v
        if self.training:
            with torch.no_grad():  # one power iteration, stored in the buffers (as nn.utils.spectral_norm)
                v = F.normalize(torch.bmm(w.transpose(1, 2), u.unsqueeze(2)).squeeze(2), dim=1, eps=self.eps)
                u = F.normalize(torch.bmm(w, v.unsqueeze(2)).squeeze(2), dim=1, eps=self.eps)
                self.weight_u.copy_(u)
                self.weight_v.copy_(v)
            u, v = u.clone(), v.clone()
        sigma = (u * torch.bmm(w, v.unsqueeze(2)).squeeze(2)).sum(dim=1)
        return w / sigma.view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        b = self.block_size
        if c != self.in_channels or h % b or w % b or (h // b) * (w // b) != self.num_blocks:
            raise ValueError(f"expected ({self.in_channels}, H, W) inputs with {self.num_blocks} blocks of {b}x{b}, "
                             f"got {tuple(x.shape)}")
        nbh, nbw = h // b, w // b
        blocks = x.reshape(n, c, nbh, b, nbw, b).permute(0, 2, 4, 1, 3, 5).reshape(n, nbh * nbw, c * b * b)
        out = torch.einsum("nki,koi->nko", blocks, self.normalized_weight())
        return out.permute(0, 2, 1).reshape(n, self.out_channels, nbh, nbw)

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, block_size={self.block_size}, num_blocks={self.num_blocks}"


class BlockwiseBatchNorm(nn.Module):
    """A separate ``BatchNorm2d(channels)`` for every block position of a ``(N, channels, nbh, nbw)`` map.

    Each per-block BatchNorm of the original code sees an ``(N, channels, 1, 1)`` tensor, so it normalises every
    (block, channel) pair with statistics over the mini-batch only: exactly ``BatchNorm1d(nb * channels)`` on the
    flattened map (feature ``block * channels + channel``).
    """

    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.channels, self.num_blocks = channels, num_blocks
        self.bn = nn.BatchNorm1d(channels * num_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, nbh, nbw = x.shape
        y = self.bn(x.permute(0, 2, 3, 1).reshape(n, nbh * nbw * c))
        return y.reshape(n, nbh, nbw, c).permute(0, 3, 1, 2)


def _hw(image_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return (image_size, image_size) if isinstance(image_size, int) else (int(image_size[0]), int(image_size[1]))


class AdaptationNetwork(nn.Module):
    """Adaptation network of the generator (Sec. IV-B1, Table 4, Fig. 4).

    Every ``B x B`` block of the scrambled image goes through its own sub-network ``f(x_b; theta_b)`` - an SN
    convolution with ``16 B^2`` outputs, BatchNorm and LeakyReLU(0.1) - and a pixel-shuffle layer (Shi et al. 2016)
    puts the ``16 B^2`` values back at the block's position: ``(N, 3, H, W) -> (N, 16, H, W)``. ``block_size`` is 4
    for LE, ELE and EtC and 1 for PE and random PE (Sec. V).
    """

    def __init__(self, block_size: int = 4, image_size: Union[int, Tuple[int, int]] = 32, in_channels: int = 3,
                 feature_channels: int = 16):
        super().__init__()
        h, w = _hw(image_size)
        if h % block_size or w % block_size:
            raise ValueError("image_size must be a multiple of block_size")
        nb = (h // block_size) * (w // block_size)
        c = feature_channels * block_size * block_size
        self.block_size, self.feature_channels = block_size, feature_channels
        self.convs0 = BlockwiseConv(in_channels, c, block_size, nb)
        self.bns0 = BlockwiseBatchNorm(c, nb)
        self.act = nn.LeakyReLU(0.1)
        self.ps = nn.PixelShuffle(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.act(self.bns0(self.convs0(x))))


def _res_block(ch: int) -> nn.Sequential:
    """ResBlock of Fig. 5 without the skip connection and the final activation (added in the decoder)."""
    sn = nn.utils.spectral_norm
    return nn.Sequential(
        sn(nn.Conv2d(ch, ch, 3, 1, 1, bias=False)), nn.BatchNorm2d(ch), nn.LeakyReLU(0.1),
        sn(nn.Conv2d(ch, ch, 3, 1, 1, bias=False)), nn.BatchNorm2d(ch))


class FeatureDecoder(nn.Module):
    """Feature decoder (Sec. IV-B2, Table 5, Fig. 5): ``(N, in_channels, H, W) -> (N, 3, H, W)`` in [-1, 1]."""

    def __init__(self, in_channels: int = 16, width: int = 64, num_resblocks: int = 3):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.channel_opt0 = sn(nn.Conv2d(in_channels, width, 3, 1, 1, bias=False))
        self.bn0 = nn.BatchNorm2d(width)
        self.reps = nn.ModuleList([_res_block(width) for _ in range(num_resblocks)])
        self.channel_opt1 = sn(nn.Conv2d(width, width, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(width)
        self.channel_opt2 = sn(nn.Conv2d(width, 3, 3, 1, 1, bias=False))
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn0(self.channel_opt0(x)))
        for rep in self.reps:
            x = self.act(rep(x) + x)
        x = self.act(self.bn1(self.channel_opt1(x)))
        return torch.tanh(self.channel_opt2(x))


class Generator(nn.Module):
    """SIA-GAN generator (Fig. 3): adaptation network + feature decoder.

    ``forward(z)`` takes scrambled images in [-1, 1] and returns ``(image in [-1, 1], adaptation feature)``.
    ``adaptation=False`` feeds the scrambled image straight into the feature decoder: our reading of the rows
    "without adaptation network" of Tables 2-3 (the released code only contains the generator with it).
    """

    def __init__(self, block_size: int = 4, image_size: Union[int, Tuple[int, int]] = 32, adaptation: bool = True,
                 feature_channels: int = 16):
        super().__init__()
        self.adaptation = AdaptationNetwork(block_size, image_size, 3, feature_channels) if adaptation else None
        self.decoder = FeatureDecoder(feature_channels if adaptation else 3)

    def forward(self, z: torch.Tensor):
        feature = self.adaptation(z) if self.adaptation is not None else z
        return self.decoder(feature), feature


# ------------------------------------------------------------------------------------------------------------------
# discriminator
# ------------------------------------------------------------------------------------------------------------------
class InplaceSpectralNorm(nn.Module):
    """Spectral normalisation as implemented in the original code (``SpectralNorm`` of dcgan_spec_acgan.py).

    Before every forward pass (training and evaluation) one power iteration starting from a fixed random vector
    ``u`` estimates the largest singular value of the wrapped layer's weight, and the weight *data* is divided by it
    in place, so the stored parameter itself stays (approximately) spectrally normalised. ``u`` is never updated.
    """

    def __init__(self, module: nn.Module, eps: float = 1e-12):
        super().__init__()
        self.module, self.eps = module, eps
        self.register_buffer("u", torch.randn(module.weight.size(0), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            w = self.module.weight
            wm = w.view(w.size(0), -1)
            v = wm.t() @ self.u
            v = v / (v.norm(p=2) + self.eps)
            u = wm @ v
            u = u / (u.norm(p=2) + self.eps)
            w.data /= u.t() @ wm @ v
        return self.module(x)


def _sn(module: nn.Module, mode: str) -> nn.Module:
    if mode == "original":
        return InplaceSpectralNorm(module)
    if mode == "torch":
        return nn.utils.spectral_norm(module)
    if mode == "none":
        return module
    raise ValueError("discriminator_sn must be 'original', 'torch' or 'none'")


class SelfAttention(nn.Module):
    """Self-attention layer of SAGAN (Zhang et al. 2018, ref. [25] of the paper), ``Self_Attn`` of the code."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h = x.size()
        q = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, w * h)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value_conv(x).view(b, -1, w * h)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, w, h)
        return self.gamma * out + x


class Discriminator(nn.Module):
    """SIA-GAN discriminator (Sec. IV-C, Table 6); returns one score per image.

    As in ``archive/dcgan_spec_acgan.py`` there is no activation between ``layer4`` (the convolution to ``ndf``
    channels) and the second self-attention layer, although Table 6 lists a LeakyReLU there.

    ``ndf`` is the channel count of the last stage (1024 in Table 6; smaller values only for quick tests).
    ``sn``: ``"original"`` (in-place spectral normalisation of the code, default), ``"torch"``
    (``torch.nn.utils.spectral_norm``) or ``"none"``.
    """

    def __init__(self, ndf: int = 1024, image_size: int = 32, sn: str = "original"):
        super().__init__()

        def stage(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(_sn(nn.Conv2d(i, o, 3, 1, 1), sn), nn.LeakyReLU(0.1),
                                 _sn(nn.Conv2d(o, o, 4, 2, 1), sn), nn.LeakyReLU(0.1))

        self.layer1 = stage(3, ndf // 8)
        self.layer2 = stage(ndf // 8, ndf // 4)
        self.layer3 = stage(ndf // 4, ndf // 2)
        self.attn1 = SelfAttention(ndf // 2)
        self.layer4 = _sn(nn.Conv2d(ndf // 2, ndf, 3, 1, 1), sn)
        self.attn2 = SelfAttention(ndf)
        self.linear = _sn(nn.Linear(ndf * (image_size // 8) ** 2, 1), sn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn1(self.layer3(self.layer2(self.layer1(x))))
        out = self.attn2(self.layer4(out))
        return self.linear(out.flatten(1)).view(-1)
