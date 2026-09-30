"""The package reproduces the original research code kept in archive/ (same inputs and weights -> same outputs)."""
import ast
import os
import pickle
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

import siagan

ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")
if not os.path.isdir(ARCHIVE):  # pragma: no cover
    pytest.skip("archive/ not available", allow_module_level=True)
sys.path.insert(0, ARCHIVE)
sys.dont_write_bytecode = True  # keep archive/ free of __pycache__


def archived_function(filename, name, **namespace):
    """Load one top-level function of an archived script without running the script."""
    with open(os.path.join(ARCHIVE, filename)) as f:
        tree = ast.parse(f.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def images():  # ToTensor-like batch (N, C, H, W) of 8-bit values in [0, 1]
    return (np.random.RandomState(0).randint(0, 256, size=(5, 3, 32, 32)) / 255.0).astype(np.float32)


def randomize_bn(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
            m.running_mean.uniform_(-0.1, 0.1)
            m.running_var.uniform_(0.5, 1.5)


# ------------------------------------------------------------------------------------------------ scrambling
def test_paper_keys_are_the_original_key_files():
    from learnable_encryption import BlockScramble

    for i in range(64):
        orig = BlockScramble(os.path.join(ARCHIVE, "key4", f"{i}_.pkl"))  # the original loader
        assert list(orig.blockSize) == [4, 4, 3]
        assert np.array_equal(orig.key.astype(np.int64), siagan.paper_keys()[i])
        with open(os.path.join(ARCHIVE, "key4", f"{i}_.pkl"), "rb") as f:
            assert np.array_equal(pickle.load(f)[1].astype(np.int64), siagan.paper_keys()[i])


def test_le_matches_the_attack_scripts(images, monkeypatch):
    """gan_attack*.py scramble with blockwise_scramble(imgs, 0), which reads key4/0_.pkl from the working dir."""
    import Blockwise_scramble_LE as orig

    monkeypatch.chdir(ARCHIVE)
    ref = orig.blockwise_scramble(images.copy(), 0)  # (N, H, W, C)
    ours = siagan.LE()(torch.from_numpy(images)).numpy()
    assert np.array_equal(ours.transpose(0, 2, 3, 1), ref)
    ref_dec = orig.blockwise_decramble(np.ascontiguousarray(ref.transpose(0, 3, 1, 2)), 0)
    our_dec = siagan.LE().inverse(torch.from_numpy(np.ascontiguousarray(ref.transpose(0, 3, 1, 2)))).numpy()
    assert np.array_equal(our_dec.transpose(0, 2, 3, 1), ref_dec)
    assert np.array_equal(our_dec, images)


def test_random_pe_inverts_half_of_the_values_like_random_mask(images):
    random_mask = archived_function("gan_attack_alpha_10.py", "random_mask", torch=torch)
    x = torch.from_numpy(images)
    torch.manual_seed(0)
    ref = random_mask(x.clone())
    ours = siagan.RandomPE(seed=0)(x)
    for out in (ref, ours):
        inverted = out != x  # k/255 != 1 - k/255 for every 8-bit value
        assert inverted.flatten(1).sum(1).tolist() == [32 * 32 * 3 // 2] * x.shape[0]  # exactly half, every image
        assert torch.equal(out[inverted], 1 - x[inverted])
        assert not torch.equal(inverted[0], inverted[1])  # a new key for every image


# ------------------------------------------------------------------------------------------------ networks
def load_original_generator(new, orig):
    """Copy the weights of archive/generator.py (64 separate sub-networks) into siagan.Generator."""
    ad = new.adaptation
    c = ad.convs0.out_channels
    with torch.no_grad():
        for b in range(ad.convs0.num_blocks):
            ad.convs0.weight[b].copy_(orig.convs0[b].weight_orig)
            ad.convs0.weight_u[b].copy_(orig.convs0[b].weight_u)
            ad.convs0.weight_v[b].copy_(orig.convs0[b].weight_v)
            for name in ("weight", "bias", "running_mean", "running_var"):
                getattr(ad.bns0.bn, name)[b * c:(b + 1) * c].copy_(getattr(orig.bns0[b], name))
    dec = {}
    for k, v in orig.state_dict().items():
        if k.startswith(("convs0.", "bns0.")):
            continue
        for i in (1, 2, 3):
            k = k.replace(f"rep{i}.", f"reps.{i - 1}.") if k.startswith(f"rep{i}.") else k
        dec[k] = v
    new.decoder.load_state_dict(dec)


def test_generator_matches_the_original():
    from generator import Generator as OriginalGenerator

    torch.manual_seed(0)
    orig = OriginalGenerator()
    randomize_bn(orig)
    new = siagan.Generator(block_size=4)
    load_original_generator(new, orig)
    z = torch.rand(6, 3, 32, 32) * 2 - 1
    for mode in ("train", "train", "eval"):  # two training passes update u/v and the BN statistics
        orig.train(mode == "train")
        new.train(mode == "train")
        out_o, feat_o = orig(z, None)
        out_n, feat_n = new(z)
        assert torch.allclose(feat_n, feat_o, atol=1e-5, rtol=1e-4)
        assert torch.allclose(out_n, out_o, atol=1e-5, rtol=1e-4)
    ad = new.adaptation
    for b in (0, 17, 63):
        assert torch.allclose(ad.convs0.weight_u[b], orig.convs0[b].weight_u, atol=1e-6)
        assert torch.allclose(ad.convs0.weight_v[b], orig.convs0[b].weight_v, atol=1e-6)
        sl = slice(b * 256, (b + 1) * 256)
        assert torch.allclose(ad.bns0.bn.running_mean[sl], orig.bns0[b].running_mean, atol=1e-6)
        assert torch.allclose(ad.bns0.bn.running_var[sl], orig.bns0[b].running_var, atol=1e-5)
    # gradients through the spectral normalisation of every block
    orig.train()
    new.train()
    orig(z, None)[0].square().mean().backward()
    new(z)[0].square().mean().backward()
    for b in (0, 17, 63):
        assert torch.allclose(ad.convs0.weight.grad[b], orig.convs0[b].weight_orig.grad, atol=1e-6, rtol=1e-3)
    assert torch.allclose(new.decoder.reps[2][0].weight_orig.grad, orig.rep3[0].weight_orig.grad, atol=1e-6,
                          rtol=1e-3)


def test_discriminator_matches_the_original():
    from dcgan_spec_acgan import Discriminator as OriginalDiscriminator

    torch.manual_seed(0)
    orig = OriginalDiscriminator()  # ndf = 1024, in-place spectral normalisation
    new = siagan.Discriminator(ndf=1024, sn="original")
    state = {}
    for k, v in orig.state_dict().items():
        if k.startswith(("real.", "fake.")) or k.endswith(".module.v"):  # unused AC-GAN heads, unused v
            continue
        state[k.replace(".module.u", ".u")] = v
    new.load_state_dict(state)
    with torch.no_grad():
        for m in (orig, new):
            m.attn1.gamma.fill_(0.3)
            m.attn2.gamma.fill_(-0.2)
    x = torch.rand(3, 3, 32, 32) * 2 - 1
    for _ in range(2):  # every forward pass divides the stored weights in place
        assert torch.allclose(new(x), orig(x)[0], atol=1e-5, rtol=1e-4)
    assert torch.allclose(new.layer2[2].module.weight, orig.layer2[2].module.weight, atol=1e-7)
    assert torch.allclose(new.linear.module.weight, orig.linear.module.weight, atol=1e-7)


def test_classifier_matches_the_original(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)  # the original hard-codes .cuda()
    from no_adaptation_network import ShakePyramidNet as OriginalShakePyramidNet

    torch.manual_seed(0)
    orig = OriginalShakePyramidNet(depth=20, alpha=48, label=10)
    randomize_bn(orig)
    new = siagan.ShakePyramidNet(depth=20, alpha=48, num_classes=10)
    new.load_state_dict(orig.state_dict())  # same parameter names
    orig.eval()
    new.eval()
    x = torch.rand(4, 3, 32, 32)
    assert torch.allclose(new(x), orig(x), atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------------------------------ losses
def test_losses_match_the_original_expressions():
    tv = archived_function("gan_attack_alpha_10.py", "total_variation_norm")
    torch.manual_seed(0)
    feature = torch.randn(4, 16, 32, 32)
    assert torch.allclose(siagan.feature_total_variation(feature), tv(feature))
    judge_real, judge_fake = torch.randn(8), torch.randn(8)
    d_loss = nn.ReLU()(1.0 + judge_fake).mean() + nn.ReLU()(1.0 - judge_real).mean()  # gan_attack_alpha_10.py
    assert torch.allclose(siagan.discriminator_hinge_loss(judge_real, judge_fake), d_loss)
    assert torch.allclose(siagan.generator_hinge_loss(judge_fake), -judge_fake.mean())
