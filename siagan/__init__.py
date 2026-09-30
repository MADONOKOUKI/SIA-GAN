"""siagan: official PyTorch implementation of "SIA-GAN: Scrambling Inversion Attack Using Generative Adversarial
Network" (Madono, Tanaka, Onishi, Ogawa; IEEE Access 9, 2021, doi:10.1109/ACCESS.2021.3112684).

* scramblers attacked in the paper (Table 1): :class:`PE`, :class:`RandomPE`, :class:`LE`, :class:`ELE`,
  :class:`EtC` and :class:`BlockShuffle`, for PIL images, numpy arrays and torch tensors, with the keys of the
  original experiments by default (:func:`paper_keys`);
* networks (Tables 4-6): :class:`Generator` = :class:`AdaptationNetwork` + :class:`FeatureDecoder`, and
  :class:`Discriminator`;
* the attack: :class:`SIAGAN` (``fit``, ``train_step``, ``reconstruct``, ``evaluate``, ``save``/``load``) with the
  hinge losses of Eqs. 3-4;
* metrics: :func:`psnr`, :func:`ssim` and :class:`LPIPS` (the metric of Table 3).
"""

from .classifier import ShakeDrop, ShakePyramidNet
from .losses import discriminator_hinge_loss, feature_total_variation, generator_hinge_loss
from .metrics import LPIPS, psnr, ssim
from .networks import (AdaptationNetwork, BlockwiseBatchNorm, BlockwiseConv, Discriminator, FeatureDecoder,
                       Generator, InplaceSpectralNorm, SelfAttention)
from .scrambling import (ELE, LE, PAPER_SEED, PE, SCHEMES, BlockShuffle, EtC, RandomPE, Scrambler, get_scrambler,
                         paper_keys, paper_pe_key)
from .trainer import SIAGAN, pick_device

__version__ = "1.0.0"

__all__ = [
    "Scrambler", "PE", "RandomPE", "LE", "ELE", "EtC", "BlockShuffle", "SCHEMES", "get_scrambler",
    "paper_keys", "paper_pe_key", "PAPER_SEED",
    "AdaptationNetwork", "FeatureDecoder", "Generator", "Discriminator", "InplaceSpectralNorm", "SelfAttention",
    "BlockwiseConv", "BlockwiseBatchNorm",
    "ShakeDrop", "ShakePyramidNet",
    "discriminator_hinge_loss", "generator_hinge_loss", "feature_total_variation",
    "SIAGAN", "pick_device",
    "psnr", "ssim", "LPIPS",
]
