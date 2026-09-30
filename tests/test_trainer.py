"""SIAGAN: one optimisation step, fit (Same / Diff. real images), reconstruct, evaluate, save / load."""
import numpy as np
import pytest
import torch

import siagan


def small_attack(**kw):
    kw.setdefault("classifier", siagan.ShakePyramidNet(depth=8, alpha=12))  # the code's classifier, narrowed
    return siagan.SIAGAN(10, discriminator_width=64, device="cpu", **kw)


@pytest.mark.parametrize("kw", [dict(), dict(classifier=None, discriminator_sn="torch"),
                                dict(mixup_alpha=1.0, lambda_tv=1e-3), dict(adaptation=False, block_size=1)],
                         ids=["code", "adversarial-only", "mixup-tv", "no-adaptation"])
def test_one_step_updates_generator_and_discriminator(kw):
    torch.manual_seed(0)
    attack = small_attack(**kw)
    x, y = torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,))
    g0 = [p.detach().clone() for p in attack.generator.parameters()]
    d0 = [p.detach().clone() for p in attack.discriminator.parameters()]
    stats = attack.train_step(siagan.LE()(x[4:]), y[4:], x[:4])
    assert {"d_loss", "g_adv", "g_loss"} <= set(stats) and all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(g0, attack.generator.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(d0, attack.discriminator.parameters()))
    assert ("cls_loss" in stats) == (attack.classifier is not None)
    assert attack.step == 1


def test_default_settings_are_the_paper_settings():
    attack = siagan.SIAGAN(discriminator_width=64, classifier=None, device="cpu")
    assert attack.opt_g.defaults["lr"] == 1e-4 and attack.opt_d.defaults["lr"] == 4e-4
    assert attack.opt_g.defaults["betas"] == (0.0, 0.9) == attack.opt_d.defaults["betas"]
    code = siagan.SIAGAN(device="cpu", discriminator_width=64, classifier_depth=8, classifier_alpha=12)
    assert isinstance(code.classifier, siagan.ShakePyramidNet) and code.lambda_cls == 1.0
    assert code.config["classifier_depth"] == 8 and siagan.SIAGAN.__init__.__kwdefaults__["classifier_depth"] == 110


def loader(n=12, batch=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.rand(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, n, batch)]


def test_fit_same_and_diff_real_images():
    torch.manual_seed(0)
    attack = small_attack(classifier=None)
    hist = attack.fit(loader(), siagan.ELE(), epochs=2, log=None)  # "Same": halves of every batch
    assert [h["epoch"] for h in hist] == [1, 2] and attack.step == 6 and attack.epoch == 2
    real = [(x,) for x, _ in loader(n=4, seed=1)]  # "Diff.": a separate loader of real images (cycled)
    hist = attack.fit(loader(), siagan.RandomPE(seed=0), epochs=1, real_loader=real, log=None)
    assert hist[0]["epoch"] == 3 and attack.step == 9
    attack.fit(loader(), siagan.PE(), epochs=5, max_steps=10, log=None)
    assert attack.step == 10


def test_reconstruct_and_evaluate():
    torch.manual_seed(0)
    attack = small_attack(block_size=1, classifier=None)
    data = loader(n=6, batch=3)
    attack.fit(data, siagan.PE(), epochs=1, log=None)
    x = torch.cat([b[0] for b in data])
    r = attack.reconstruct(siagan.PE()(x))
    assert r.shape == x.shape and r.min() >= 0 and r.max() <= 1 and r.device.type == "cpu"
    assert attack.reconstruct(siagan.PE()(x[0])).shape == (3, 32, 32)
    x_u8 = (x.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)  # channels-last uint8 in -> same out
    r_u8 = attack.reconstruct(siagan.PE()(x_u8))
    assert r_u8.shape == x_u8.shape and r_u8.dtype == np.uint8
    res = attack.evaluate(data, siagan.PE(), metrics=("psnr", "ssim"))
    assert res["num_images"] == 6 and set(res["scrambled"]) == set(res["reconstructed"]) == {"psnr", "ssim"}
    assert res["scrambled"]["psnr"] == pytest.approx(siagan.psnr(x, siagan.PE()(x)))


def test_save_and_load(tmp_path):
    torch.manual_seed(0)
    attack = small_attack(classifier=None, lambda_tv=1e-3)
    attack.fit(loader(), siagan.LE(), epochs=1, log=None)
    path = tmp_path / "siagan.pt"
    attack.save(str(path))
    again = siagan.SIAGAN.load(str(path), device="cpu")
    assert (again.epoch, again.step, again.config) == (attack.epoch, attack.step, attack.config)
    s = siagan.LE()(torch.rand(4, 3, 32, 32))
    assert torch.equal(again.reconstruct(s), attack.reconstruct(s))
    custom = small_attack()
    custom.save(str(path))
    with pytest.raises(ValueError):
        siagan.SIAGAN.load(str(path))  # a custom classifier module must be passed back
    assert siagan.SIAGAN.load(str(path), classifier=siagan.ShakePyramidNet(8, 12)).classifier is not None
