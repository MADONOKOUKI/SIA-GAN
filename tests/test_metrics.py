"""PSNR, SSIM and LPIPS."""
import numpy as np
import pytest
import torch

import siagan


def test_psnr_and_ssim():
    rng = np.random.RandomState(0)
    x = rng.rand(3, 32, 32, 3).astype(np.float32)
    assert siagan.psnr(x, x) == np.inf and siagan.ssim(x, x) == pytest.approx(1.0)
    noisy = np.clip(x + 0.1 * rng.randn(*x.shape).astype(np.float32), 0, 1)
    mse = ((x.astype(np.float64) - noisy) ** 2).reshape(3, -1).mean(1)
    assert siagan.psnr(x, noisy, "none") == pytest.approx(10 * np.log10(1 / mse))
    assert 0 < siagan.ssim(x, noisy) < 1
    u8 = (x * 255).round().astype(np.uint8)  # uint8 and float, numpy and torch give the same values
    t = torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255
    assert siagan.psnr(u8, siagan.LE()(u8)) == pytest.approx(siagan.psnr(t, siagan.LE()(t)))
    with pytest.raises(ValueError):
        siagan.psnr(x, x[:2])


def test_lpips_default_is_the_original_convention():
    lp = siagan.LPIPS(pnet_rand=True)  # random backbone: no weight download in tests
    assert lp.net == "alex" and lp.normalize is False
    x = torch.rand(4, 3, 32, 32)
    d = lp(x, x)
    assert d.shape == (4,) and np.allclose(d, 0, atol=1e-6)
    assert (lp(x, siagan.EtC()(x)) > 0).all()
    u8 = (x.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    assert lp(u8, u8).shape == (4,)
