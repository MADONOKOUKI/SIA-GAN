"""The scramblers are bit-identical to those of scramblekit (the maintained library of this paper series) for the same
keys. Skipped when scramblekit is not installed (``pip install scramblekit``)."""
import numpy as np
import pytest
import torch
from PIL import Image

import siagan

pytest.importorskip("scramblekit")
sks = pytest.importorskip("scramblekit.scramble")


@pytest.fixture
def batches():
    u8 = np.random.RandomState(0).randint(0, 256, size=(4, 32, 32, 3)).astype(np.uint8)  # channels-last uint8
    f32 = torch.from_numpy(u8.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)       # ToTensor-like floats
    return u8, f32, Image.fromarray(u8[0])


PAIRS = [
    ("pe", lambda k: siagan.PE(k), lambda k: sks.PE(k, image_size=32)),
    ("pe-permutation", lambda k: siagan.PE(k, channel_mode="permutation"),
     lambda k: sks.PE(k, image_size=32, channel_mode="permutation")),
    ("le", lambda k: siagan.LE(k), lambda k: sks.LE(k)),
    ("ele", lambda k: siagan.ELE(k), lambda k: sks.ELE(k)),
    ("etc", lambda k: siagan.EtC(k), lambda k: sks.EtC(k)),
    ("etc-permutation", lambda k: siagan.EtC(k, channel_mode="permutation"),
     lambda k: sks.EtC(k, channel_mode="permutation")),
    ("blockshuffle", lambda k: siagan.BlockShuffle(k), lambda k: sks.BlockShuffle(k)),
]


@pytest.mark.parametrize("key", ["paper", 0, 7, 12345])
@pytest.mark.parametrize("name,ours,theirs", PAIRS, ids=[p[0] for p in PAIRS])
def test_same_outputs_as_scramblekit(batches, name, ours, theirs, key):
    u8, f32, pil = batches
    a, b = ours(key), theirs(key)
    assert np.array_equal(a(u8), b(u8))
    assert torch.equal(a(f32), b(f32))
    assert np.array_equal(np.asarray(a(pil)), np.asarray(b(pil)))
    assert a.invertible == b.invertible
    if a.invertible:
        assert np.array_equal(a.inverse(u8), b.inverse(u8))
        assert torch.equal(a.inverse(f32), b.inverse(f32))


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("channel_shuffle", [False, True])
def test_random_pe_same_key_stream_as_scramblekit(batches, seed, channel_shuffle):
    u8, f32, _ = batches
    a = siagan.RandomPE(seed=seed, channel_shuffle=channel_shuffle)
    b = sks.RandomPE(seed=seed, channel_shuffle=channel_shuffle)
    for x in (u8, f32, u8):  # successive calls draw new keys from the same stream
        out_a, out_b = a(x), b(x)
        assert np.array_equal(np.asarray(out_a), np.asarray(out_b))


def test_same_paper_keys_as_scramblekit():
    from scramblekit.scramble.keys import paper_le_keys, scramblemix_pe_keys

    assert np.array_equal(siagan.paper_keys(), paper_le_keys())
    inv, colors = siagan.paper_pe_key()
    ref_inv, ref_colors = scramblemix_pe_keys("cifar10")[0]
    assert np.array_equal(inv, ref_inv) and np.array_equal(colors, ref_colors)


def test_same_psnr_and_ssim_as_scramblekit(batches):
    from scramblekit import metrics as skm

    u8, f32, _ = batches
    s = siagan.LE()(f32)
    assert siagan.psnr(f32, s, "none") == pytest.approx(skm.psnr(f32, s, "none"), abs=0)
    assert siagan.ssim(u8, siagan.EtC()(u8), reduction="none") == pytest.approx(
        skm.ssim(u8, sks.EtC("paper")(u8), reduction="none"), abs=0)
