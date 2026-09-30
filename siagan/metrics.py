"""Image-similarity metrics: PSNR, SSIM and LPIPS.

Table 3 of the paper reports LPIPS (Zhang et al. 2018, AlexNet) between the original test images and the scrambled
images or the images recovered by SIA-GAN; a *lower* LPIPS of the recovered images means a more successful attack.
The original evaluation (``archive/gan_attack_alpha_10.py``) passes images in [0, 1] to ``lpips.LPIPS(net='alex')``
without ``normalize=True`` (the network expects [-1, 1]); :class:`LPIPS` keeps this convention by default
(``normalize=False``) so that the numbers are comparable with Table 3.

All functions take pairs of images or batches as numpy arrays (channels-last) or torch tensors (channels-first),
``uint8`` (0..255) or float (0..1).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .scrambling import as_nhwc

__all__ = ["psnr", "ssim", "LPIPS"]


def _pair(x, y):
    a, _ = as_nhwc(x)
    b, _ = as_nhwc(y)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a = a.astype(np.float64) * (1.0 / 255.0 if a.dtype == np.uint8 else 1.0)
    b = b.astype(np.float64) * (1.0 / 255.0 if b.dtype == np.uint8 else 1.0)
    return a, b


def _reduce(val: np.ndarray, reduction: str):
    if reduction == "mean":
        return float(np.mean(val))
    if reduction == "none":
        return val
    raise ValueError("reduction must be 'mean' or 'none'")


def psnr(x, y, reduction: str = "mean"):
    """Peak signal-to-noise ratio in dB (peak 1 for [0, 1] images), per image; identical images give ``inf``."""
    a, b = _pair(x, y)
    mse = ((a - b) ** 2).reshape(a.shape[0], -1).mean(axis=1)
    with np.errstate(divide="ignore"):
        val = np.where(mse == 0, np.inf, 10.0 * np.log10(1.0 / np.maximum(mse, 1e-300)))
    return _reduce(val, reduction)


def _filter_same(img: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Separable 2-D correlation with zero padding over axes 1 and 2 of ``(N, H, W, C)``."""
    r = w.size // 2
    p = np.pad(img, ((0, 0), (r, r), (r, r), (0, 0)))
    h, wd = img.shape[1], img.shape[2]
    tmp = sum(w[k] * p[:, k:k + h, :, :] for k in range(w.size))
    return sum(w[k] * tmp[:, :, k:k + wd, :] for k in range(w.size))


def ssim(x, y, window_size: int = 11, sigma: float = 1.5, reduction: str = "mean"):
    """Structural similarity (Wang et al. 2004) per image: 11x11 Gaussian window (sigma 1.5) per channel with zero
    padding, ``C1 = 0.01^2``, ``C2 = 0.03^2``, averaged over pixels and channels (as ``pytorch_ssim.SSIM``)."""
    a, b = _pair(x, y)
    g = np.exp(-((np.arange(window_size) - window_size // 2) ** 2) / (2.0 * sigma ** 2))
    w = g / g.sum()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = _filter_same(a, w), _filter_same(b, w)
    s_aa = _filter_same(a * a, w) - mu_a ** 2
    s_bb = _filter_same(b * b, w) - mu_b ** 2
    s_ab = _filter_same(a * b, w) - mu_a * mu_b
    smap = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return _reduce(smap.reshape(smap.shape[0], -1).mean(axis=1), reduction)


def _to_tensor01(x) -> torch.Tensor:
    """Any image or batch -> float tensor ``(N, C, H, W)`` in [0, 1] on the CPU."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        t = t.unsqueeze(0) if t.dim() == 3 else t
        return t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
    arr, _ = as_nhwc(x)
    arr = arr.astype(np.float32) / (255.0 if arr.dtype == np.uint8 else 1.0)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))


class LPIPS:
    """Learned Perceptual Image Patch Similarity with the ``lpips`` package; ``LPIPS()(x, y)`` returns one distance per
    image (numpy array).

    ``net="alex"`` and ``normalize=False`` are the settings of the original evaluation (see the module docstring);
    ``normalize=True`` rescales [0, 1] inputs to [-1, 1] as ``lpips`` expects, for values comparable with other
    LPIPS numbers. Extra keyword arguments go to ``lpips.LPIPS`` (e.g. ``pnet_rand=True`` avoids downloading the
    AlexNet weights, for tests).
    """

    def __init__(self, net: str = "alex", normalize: bool = False, device=None, **kwargs):
        import lpips

        self.net, self.normalize = net, bool(normalize)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        kwargs.setdefault("verbose", False)
        with warnings.catch_warnings():  # lpips calls torchvision with the deprecated `pretrained=` argument
            warnings.simplefilter("ignore", UserWarning)
            self.model = lpips.LPIPS(net=net, **kwargs).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, x, y, batch_size: int = 256) -> np.ndarray:
        a, b = _to_tensor01(x), _to_tensor01(y)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        out = []
        for i in range(0, a.shape[0], batch_size):
            d = self.model(a[i:i + batch_size].to(self.device), b[i:i + batch_size].to(self.device),
                           normalize=self.normalize)
            out.append(d.reshape(-1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def __repr__(self) -> str:
        return f"LPIPS(net={self.net!r}, normalize={self.normalize})"
