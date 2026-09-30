"""Scramblers: input types, determinism for a fixed key, invertibility and key handling."""
import numpy as np
import pytest
import torch
from PIL import Image

import siagan

KEYED = ["pe", "le", "ele", "etc", "blockshuffle"]


@pytest.fixture
def img():
    return np.random.RandomState(1).randint(0, 256, size=(32, 32, 3)).astype(np.uint8)


def test_paper_keys():
    keys = siagan.paper_keys()
    assert keys.shape == (64, 96)
    assert all(sorted(k) == list(range(96)) for k in keys)  # each key is a permutation of the 96 4-bit values
    with pytest.raises(ValueError):
        keys[0, 0] = 1  # read-only
    inv, colors = siagan.paper_pe_key()
    assert inv.shape == (3072,) and set(np.unique(inv)) <= {0, 1}
    assert colors.shape == (1024,) and colors.min() >= 0 and colors.max() <= 5


@pytest.mark.parametrize("name", KEYED + ["randompe"])
def test_input_types_shapes_and_dtypes(name, img):
    s = siagan.get_scrambler(name, "paper") if name != "randompe" else siagan.RandomPE(seed=0)
    out = s(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    batch = np.stack([img, img[::-1]])
    assert s(batch).shape == batch.shape
    pil = s(Image.fromarray(img))
    assert isinstance(pil, Image.Image) and pil.size == (32, 32) and pil.mode == "RGB"
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255
    assert s(t).shape == (3, 32, 32) and s(t).dtype == torch.float32
    tb = torch.stack([t, t])
    assert s(tb).shape == (2, 3, 32, 32)
    assert s(t.half()).dtype == torch.float16
    assert s(torch.from_numpy(img).permute(2, 0, 1).contiguous()).dtype == torch.uint8
    assert not np.array_equal(out, img)


@pytest.mark.parametrize("name", KEYED)
def test_same_key_same_output_and_types_agree(name, img):
    a, b = siagan.get_scrambler(name, "paper"), siagan.get_scrambler(name, "paper")
    assert np.array_equal(a(img), b(img))
    assert np.array_equal(siagan.get_scrambler(name, 5)(img), siagan.get_scrambler(name, 5)(img))
    assert not np.array_equal(siagan.get_scrambler(name, 5)(img), siagan.get_scrambler(name, 6)(img))
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255  # float path == uint8 path
    assert np.array_equal((a(t) * 255).round().byte().permute(1, 2, 0).numpy(), a(img))
    assert np.array_equal(np.asarray(a(Image.fromarray(img))), a(img))


@pytest.mark.parametrize("scrambler", [siagan.LE(), siagan.LE(3), siagan.ELE(), siagan.ELE(3), siagan.BlockShuffle(),
                                       siagan.EtC(channel_mode="permutation"), siagan.EtC(4, channel_mode="permutation"),
                                       siagan.PE(channel_mode="permutation"), siagan.PE(2, channel_mode="permutation")],
                         ids=repr)
def test_inverse(scrambler, img):
    assert scrambler.invertible
    assert np.array_equal(scrambler.inverse(scrambler(img)), img)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255
    assert torch.allclose(scrambler.inverse(scrambler(t)), t, atol=1e-6)


@pytest.mark.parametrize("scrambler", [siagan.EtC(), siagan.PE(), siagan.RandomPE()], ids=repr)
def test_not_invertible(scrambler, img):
    assert not scrambler.invertible
    with pytest.raises(ValueError):
        scrambler.inverse(img)


def test_le_uses_one_key_for_every_block(img):
    tiled = np.tile(img[:4, :4], (8, 8, 1))  # the same block everywhere
    out = siagan.LE()(tiled)
    assert np.array_equal(out, np.tile(out[:4, :4], (8, 8, 1)))
    assert not np.array_equal(siagan.ELE(key=(siagan.ELE().keys, np.arange(64)))(tiled), out)  # ELE: per block


def test_block_shuffle_moves_blocks(img):
    s = siagan.BlockShuffle()
    out = s(img)
    blocks = img.reshape(8, 4, 8, 4, 3).transpose(0, 2, 1, 3, 4).reshape(64, 4, 4, 3)
    out_blocks = out.reshape(8, 4, 8, 4, 3).transpose(0, 2, 1, 3, 4).reshape(64, 4, 4, 3)
    assert np.array_equal(out_blocks, blocks[s.perm])
    ele = siagan.ELE()
    assert np.array_equal(ele.perm, s.perm)  # the ELE block order of the paper is random.seed(30)


def test_random_pe(img):
    s = siagan.RandomPE(seed=0)
    first, second = s(img), s(img)
    assert not np.array_equal(first, second)  # a fresh key on every call
    assert np.array_equal(first, siagan.RandomPE(seed=0)(img))  # reproducible stream
    changed = first != img
    assert changed.sum() == 32 * 32 * 3 // 2 and np.array_equal(first[changed], 255 - img[changed])
    assert (siagan.RandomPE(ratio=0.0)(img) == img).all()
    assert (siagan.RandomPE(ratio=1.0)(img) == 255 - img).all()


def test_other_block_and_image_sizes():
    x = np.random.RandomState(2).randint(0, 256, size=(2, 16, 24, 3)).astype(np.uint8)
    for s in (siagan.LE(1, block_size=2), siagan.ELE(1, block_size=2, image_size=(16, 24)),
              siagan.BlockShuffle(1, block_size=8, image_size=(16, 24)),
              siagan.EtC(1, image_size=(16, 24), channel_mode="permutation"),
              siagan.PE(1, image_size=(16, 24), channel_mode="permutation")):
        assert np.array_equal(s.inverse(s(x)), x)
    assert siagan.LE()(x).shape == x.shape  # LE works for any multiple of the block size


def test_errors(img):
    with pytest.raises(ValueError):
        siagan.ELE()(np.zeros((16, 16, 3), np.uint8))  # paper key is for 32x32 images
    with pytest.raises(ValueError):
        siagan.ELE("paper", image_size=16)
    with pytest.raises(ValueError):
        siagan.LE("paper", block_size=2)
    with pytest.raises(ValueError):
        siagan.LE(key=np.arange(95))
    with pytest.raises(ValueError):
        siagan.LE()(np.zeros((30, 30, 3), np.uint8))  # not a multiple of the block size
    with pytest.raises(ValueError):
        siagan.LE()(np.zeros((32, 32), np.uint8))  # grayscale
    with pytest.raises(TypeError):
        siagan.LE()(img.astype(np.int32))
    with pytest.raises(ValueError):
        siagan.EtC(channel_mode="swap")
    with pytest.raises(ValueError):
        siagan.get_scrambler("jpeg")
    assert isinstance(siagan.get_scrambler("RandomPE", 3), siagan.RandomPE)
