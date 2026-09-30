"""Losses of SIA-GAN (Sec. IV-D).

The paper trains with the hinge loss of SN-GAN (Miyato et al. 2018): Eq. 3 for the discriminator and Eq. 4 for the
generator, written as the code computes them (``archive/gan_attack_alpha_10.py``)::

    d_loss = relu(1 + D(G(x~))).mean() + relu(1 - D(x)).mean()
    g_loss = -D(G(x~)).mean()  (+ the classifier cross entropy of the code, see siagan.SIAGAN)

:func:`feature_total_variation` is the optional ``1e-3 * total_variation_norm(feature)`` term of
``archive/gan_attack_alpha_10.py`` (off by default, as in the paper).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["discriminator_hinge_loss", "generator_hinge_loss", "feature_total_variation"]


def discriminator_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss of the discriminator (Eq. 3): ``E[relu(1 - D(x))] + E[relu(1 + D(G(x~)))]``."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def generator_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Adversarial loss of the generator (Eq. 4): ``-E[D(G(x~))]``."""
    return -d_fake.mean()


def feature_total_variation(feature: torch.Tensor) -> torch.Tensor:
    """``total_variation_norm`` of ``archive/gan_attack_alpha_10.py`` (beta = 2).

    Sum over the first three channels of ``(f[h, w] - f[h + 1, w])^2 + (f[h, w] - f[h, w + 1])^2`` on the
    ``(H - 1) x (W - 1)`` top-left region, divided by the number of values of one feature map (16 x 32 x 32). The
    original function is adapted from ``total_variation_norm`` of utkuozbulak/pytorch-cnn-visualizations
    (inverted_representation.py).
    """
    f = feature[:, :3]
    base = f[:, :, :-1, :-1]
    tv = ((base - f[:, :, 1:, :-1]) ** 2 + (base - f[:, :, :-1, 1:]) ** 2).sum()
    return tv / feature[0].numel()
