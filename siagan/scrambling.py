"""Image scrambling schemes attacked in the paper (Sec. III-A, Table 1).

======================  =========================================================================================
scheme                  operation
======================  =========================================================================================
:class:`PE`             pixel-based image encryption (Sirichotedumrong et al. 2019): negative-positive transform of
                        individual values and colour-component shuffling of individual pixels, one fixed key
:class:`RandomPE`       "random PE" (Sirichotedumrong, Kinoshita and Kiya 2019): the negative-positive transform
                        with a fresh random key for every image, i.e. ``random_mask`` of the original SIA-GAN code
:class:`LE`             learnable image encryption (Tanaka 2018): the same pixel-shuffling and negative-positive key
                        in every 4x4 block (8-bit values split into 4-bit halves)
:class:`ELE`            extended learnable encryption (Madono et al. 2020): a different LE key in every block, then
                        block location shuffling
:class:`EtC`            encryption-then-compression (Chuman et al. 2018): block rotation, inversion,
                        negative-positive transform and colour-component shuffling per block, then block location
                        shuffling
:class:`BlockShuffle`   block location shuffling alone (the operation the paper finds decisive, Sec. V-A)
======================  =========================================================================================

Every scheme is a callable object: ``scheme(img)`` scrambles

* a ``PIL.Image`` (returns a ``PIL.Image``),
* a numpy array in channels-last layout, ``(H, W, 3)`` or ``(N, H, W, 3)``, ``uint8`` in 0..255 or float in [0, 1],
* a ``torch.Tensor`` ``(3, H, W)`` or ``(N, 3, H, W)``, float in [0, 1] (e.g. after ``transforms.ToTensor``) or
  ``uint8``,

and returns the same type, shape, dtype and device, so it can be used inside ``torchvision.transforms.Compose``. The
same key is applied to every image of a batch. ``scheme.inverse(img)`` undoes it when the scheme is invertible.

Keys (the same convention as the scramblekit library, whose scramblers give bit-identical outputs):

* ``key="paper"`` (the default) - the keys of the original experiments. LE uses ``key4/0_.pkl`` of the original code
  and ELE the 64 files ``key4/0_.pkl ... key4/63_.pkl`` (one per block); they ship with the package
  (:func:`paper_keys`). The ELE block order and the EtC key are those of ``random.seed(30)`` of the block-wise
  scrambling code (Madono et al. 2020). The SIA-GAN code does not contain its PE keys; ``PE("paper")`` is the first
  of the eight PE keys hard-coded for CIFAR-10 in the authors' ScrambleMix code (:func:`paper_pe_key`).
* an ``int`` seed - keys drawn with the random generator and the call order of the original code for that scheme
  (``np.random.seed(s)`` + ``BlockScramble`` for LE/ELE, ``random.seed(s)`` for block orders and EtC, ...).
* explicit key arrays in the original code's encoding (see each class).

Faithfulness notes (the new code follows the original code where it differs from the paper text):

* LE/ELE work on 8-bit values; float inputs are quantised with ``round(255 x)``. For images from ``ToTensor``
  (``k / 255``) this equals the original ``(x * 255).astype(np.uint8)``.
* ELE applies the per-block pixel operation first (the key of a block is indexed by its *source* position) and then
  shuffles the blocks, as the original code; Fig. 1 draws the block shuffling first.
* EtC and PE: the original ``channel_change`` / ``pixel_based_encryption`` assign colour channels in place through
  aliased views, so five of the six colour codes *duplicate* a channel instead of permuting it. ``channel_mode=
  "original"`` (default) reproduces this (then the scheme is not invertible); ``"permutation"`` applies the intended
  permutation.
* Random PE inverts exactly half of the values of every image (``ratio=0.5``) at positions drawn independently for
  every image, without colour shuffling, as ``random_mask`` of ``archive/gan_attack_alpha_10.py`` (Table 1 of the
  paper also lists colour shuffling: ``channel_shuffle=True``).
"""

from __future__ import annotations

import functools
import os
import random
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

__all__ = [
    "Scrambler",
    "PE",
    "RandomPE",
    "LE",
    "ELE",
    "EtC",
    "BlockShuffle",
    "SCHEMES",
    "get_scrambler",
    "paper_keys",
    "paper_pe_key",
    "PAPER_SEED",
    "ETC_CHANNEL_TABLES",
    "PE_CHANNEL_TABLES",
]

#: ``random.seed(30)`` of the block-wise scrambling code: the ELE block order and the EtC key of the experiments.
PAPER_SEED = 30
_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
ImageSize = Union[int, Tuple[int, int]]


# ------------------------------------------------------------------------------------------------------------------
# keys
# ------------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _paper_npz() -> Dict[str, np.ndarray]:
    with np.load(os.path.join(_RESOURCES, "paper_keys.npz")) as data:
        return {k: data[k] for k in data.files}


def paper_keys() -> np.ndarray:
    """The 64 LE keys of the original code, ``key4/0_.pkl ... key4/63_.pkl``; shape ``(64, 96)``.

    Row ``i`` is the key stored in ``archive/key4/<i>_.pkl`` (a permutation of ``range(96)`` for 4x4 RGB blocks).
    The attack scripts scramble with row 0 (``blockwise_scramble(imgs, 0)`` in ``archive/Blockwise_scramble_LE.py``);
    ELE uses row ``8 r + c`` for the block in grid row ``r``, column ``c`` of a 32x32 image.
    """
    keys = _paper_npz()["le_key4"].astype(np.int64)
    keys.setflags(write=False)
    return keys


def paper_pe_key() -> Tuple[np.ndarray, np.ndarray]:
    """``(inv, colors)`` of ``PE("paper")``: the first CIFAR-10 PE key hard-coded in the ScrambleMix code.

    ``inv``: 3072 values (32x32x3, row / column / channel order), 0 = negative-positive transform, 1 = keep;
    ``colors``: 1024 colour codes 0..5, one per pixel. The released SIA-GAN code does not contain the PE keys of the
    experiments; this is the key that scramblekit also uses for ``key="paper"``.
    """
    d = _paper_npz()
    inv = np.unpackbits(d["pe_inv"])[:3072].astype(np.int64)
    return inv, d["pe_colors"].astype(np.int64)


def _check_seed(seed) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= int(seed) < 2 ** 32:
        raise ValueError("seed must be in [0, 2**32)")
    return int(seed)


def _is_seed(key) -> bool:
    return isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_))


def _is_paper(key) -> bool:
    return isinstance(key, str) and key.lower() == "paper"


def le_keys_from_seed(seed: int, count: int = 1, block_size: int = 4, channels: int = 3) -> np.ndarray:
    """``count`` LE keys, shape ``(count, 2 B^2 C)``: ``np.random.seed(seed)`` followed by ``count`` calls of
    ``BlockScramble([B, B, C])`` (learnable_encryption.py), each shuffling ``np.arange(2 B^2 C)``."""
    rs = np.random.RandomState(_check_seed(seed))
    keys = []
    for _ in range(count):
        k = np.arange(2 * block_size * block_size * channels, dtype=np.int64)
        rs.shuffle(k)
        keys.append(k)
    return np.stack(keys)


def block_perm_from_seed(seed: int, n_blocks: int) -> np.ndarray:
    """``random.seed(seed); perm = list(range(n_blocks)); random.shuffle(perm)`` as in the training scripts."""
    perm = list(range(n_blocks))
    random.Random(_check_seed(seed)).shuffle(perm)
    return np.asarray(perm, dtype=np.int64)


def etc_key_from_seed(seed: int, n_blocks: int) -> Dict[str, np.ndarray]:
    """EtC key drawn like the module-level code of the original etc_encryption.py after ``random.seed(seed)``."""
    rng = random.Random(_check_seed(seed))
    rotate, negaposi, reverse, channel, shuffle = [], [], [], [], []
    for i in range(n_blocks):
        rotate.append(rng.randint(0, 3))
        reverse.append(rng.randint(0, 2))
        channel.append(rng.randint(0, 5))
        negaposi.append(1 if i % 2 == 0 else 0)
        shuffle.append(i)
    rng.shuffle(shuffle)
    rng.shuffle(negaposi)
    return {name: np.asarray(v, dtype=np.int64) for name, v in (
        ("rotate", rotate), ("negaposi", negaposi), ("reverse", reverse), ("channel", channel), ("shuffle", shuffle))}


def pe_key_from_seed(seed: int, height: int, width: int, channels: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """PE key ``(inv, colors)`` from ``np.random.RandomState(seed)``: ``randint(0, 2)`` per value, then
    ``randint(0, 6)`` per pixel."""
    rs = np.random.RandomState(_check_seed(seed))
    inv = rs.randint(0, 2, size=height * width * channels)
    colors = rs.randint(0, 6, size=height * width)
    return inv.astype(np.int64), colors.astype(np.int64)


# ------------------------------------------------------------------------------------------------------------------
# image types <-> channels-last batches
# ------------------------------------------------------------------------------------------------------------------
def as_nhwc(img) -> Tuple[np.ndarray, Callable[[np.ndarray], object]]:
    """``(x, restore)``: ``x`` is an ``(N, H, W, C)`` ndarray and ``restore(y)`` returns an array of that layout in the
    type, layout, dtype and device of ``img`` (PIL image, channels-last ndarray or channels-first tensor)."""
    if isinstance(img, Image.Image):
        mode = img.mode if img.mode in ("RGB", "L") else "RGB"
        arr = np.asarray(img.convert(mode), dtype=np.uint8)
        arr = arr[:, :, None] if arr.ndim == 2 else arr

        def restore_pil(y):
            y = np.asarray(y)[0]
            if y.dtype != np.uint8:
                y = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
            return Image.fromarray(np.ascontiguousarray(y[:, :, 0] if mode == "L" else y), mode=mode)

        return arr[None], restore_pil

    if isinstance(img, torch.Tensor):
        if img.dim() not in (3, 4):
            raise ValueError(f"expected a tensor of shape (C, H, W) or (N, C, H, W), got {tuple(img.shape)}")
        src = img.detach()
        if src.dtype in (torch.float16, torch.bfloat16):
            src = src.float()
        arr = src.cpu().numpy()
        single = arr.ndim == 3
        arr = (arr[None] if single else arr).transpose(0, 2, 3, 1)

        def restore_tensor(y):
            y = np.asarray(y).transpose(0, 3, 1, 2)
            out = torch.from_numpy(np.ascontiguousarray(y[0] if single else y))
            return out.to(device=img.device, dtype=img.dtype)

        return arr, restore_tensor

    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr[None, :, :, None], lambda y: np.asarray(y)[0, :, :, 0]
    if arr.ndim == 3:
        return arr[None], lambda y: np.asarray(y)[0]
    if arr.ndim == 4:
        return arr, lambda y: np.asarray(y)
    raise ValueError(f"expected an array of shape (H, W, C) or (N, H, W, C), got {arr.shape}")


def _max_value(x: np.ndarray):
    return 255 if x.dtype == np.uint8 else x.dtype.type(1.0)


def _to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def _from_uint8(x: np.ndarray, like: np.ndarray) -> np.ndarray:
    if like.dtype == np.uint8:
        return x
    return (x.astype(like.dtype) / like.dtype.type(255.0)).astype(like.dtype)


# ------------------------------------------------------------------------------------------------------------------
# block operations on (N, H, W, C) arrays; blocks are numbered row-major (block b = row * nbw + column)
# ------------------------------------------------------------------------------------------------------------------
def _to_blocks(x: np.ndarray, b: int) -> np.ndarray:
    """``(N, H, W, C) -> (N, nb, B, B, C)``."""
    n, h, w, c = x.shape
    return x.reshape(n, h // b, b, w // b, b, c).transpose(0, 1, 3, 2, 4, 5).reshape(n, (h // b) * (w // b), b, b, c)


def _from_blocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    n, _, b, _, c = blocks.shape
    return blocks.reshape(n, h // b, w // b, b, b, c).transpose(0, 1, 3, 2, 4, 5).reshape(n, h, w, c)


def _check_perm(perm, n: int, what: str) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64).reshape(-1)
    if perm.size != n or not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError(f"{what} must be a permutation of 0..{n - 1} (got {perm.size} entries)")
    return perm


def _shuffle_blocks(x: np.ndarray, perm: np.ndarray, b: int) -> np.ndarray:
    """Output block ``i`` is input block ``perm[i]`` (``block_location_shuffle`` of the original code)."""
    return _from_blocks(_to_blocks(x, b)[:, perm], x.shape[1], x.shape[2])


def _negative_positive(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``v -> max - v`` where ``mask`` is True (255 - v for uint8, 1 - v for floats)."""
    return np.where(mask, _max_value(x) - x, x).astype(x.dtype)


def _gather_channels(x: np.ndarray, index: np.ndarray, b: int) -> np.ndarray:
    """Output channel ``c`` of block (``b = 1``: pixel) ``k`` is input channel ``index[k, c]``."""
    blocks = _to_blocks(x, b)
    out = np.take_along_axis(blocks, np.broadcast_to(index[None, :, None, None, :], blocks.shape), axis=4)
    return _from_blocks(out, x.shape[1], x.shape[2])


def _le_transform(x: np.ndarray, key: np.ndarray, b: int, inverse: bool = False) -> np.ndarray:
    """Learnable image encryption (Tanaka 2018), vectorised ``BlockScramble.doScramble`` of learnable_encryption.py.

    Every ``B x B x C`` block is flattened in (row, column, channel) order and every 8-bit value is split into its
    lower and upper 4 bits (lower first, ``2 B^2 C`` values). The values at ``rev = key > len(key) / 2`` are inverted
    (``15 - v``), the values are permuted (output ``j`` = input ``key[j]``), the ``rev`` positions are inverted again
    and the halves are merged. ``key``: ``(2 B^2 C,)`` for all blocks (LE) or ``(nb, 2 B^2 C)`` (ELE). The inverse
    uses ``argsort(key)`` with the same ``rev`` mask, as ``BlockScramble.Decramble``.
    """
    xu = _to_uint8(x)
    n, h, w, c = xu.shape
    blocks = _to_blocks(xu, b).reshape(n, -1, b * b * c)
    nb, d = blocks.shape[1], blocks.shape[2]
    rev = key > (key.shape[-1] / 2)
    order = np.argsort(key, axis=-1) if inverse else key
    if key.ndim == 1:
        rev, order = np.broadcast_to(rev, (nb, 2 * d)), np.broadcast_to(order, (nb, 2 * d))
    nib = np.concatenate([blocks & 0xF, blocks >> 4], axis=2)
    nib = np.where(rev[None], 15 - nib, nib)
    nib = np.take_along_axis(nib, np.broadcast_to(order[None], nib.shape), axis=2)
    nib = np.where(rev[None], 15 - nib, nib).astype(np.uint8)
    merged = ((nib[:, :, d:] << 4) + nib[:, :, :d]).astype(np.uint8)
    return _from_uint8(_from_blocks(merged.reshape(n, nb, b, b, c), h, w), x)


def _rotate_blocks(x: np.ndarray, turns: np.ndarray, b: int) -> np.ndarray:
    """Rotate block ``k`` by ``turns[k]`` quarter turns (``np.rot90``, counter-clockwise)."""
    blocks = _to_blocks(x, b).copy()
    for t in (1, 2, 3):
        idx = np.nonzero(turns == t)[0]
        if idx.size:
            blocks[:, idx] = np.rot90(blocks[:, idx], k=t, axes=(2, 3))
    return _from_blocks(blocks, x.shape[1], x.shape[2])


def _flip_blocks(x: np.ndarray, mode: np.ndarray, b: int) -> np.ndarray:
    """Flip block ``k``: ``mode[k]`` 0 = none, 1 = along the rows, 2 = along the columns."""
    blocks = _to_blocks(x, b).copy()
    for m, axis in ((1, 2), (2, 3)):
        idx = np.nonzero(mode == m)[0]
        if idx.size:
            blocks[:, idx] = np.flip(blocks[:, idx], axis=axis)
    return _from_blocks(blocks, x.shape[1], x.shape[2])


# ------------------------------------------------------------------------------------------------------------------
# schemes
# ------------------------------------------------------------------------------------------------------------------
def _hw(image_size: Optional[ImageSize]) -> Optional[Tuple[int, int]]:
    if image_size is None:
        return None
    if isinstance(image_size, (int, np.integer)):
        return int(image_size), int(image_size)
    return int(image_size[0]), int(image_size[1])


class Scrambler:
    """Base class: type dispatch around ``_forward`` / ``_inverse`` on ``(N, H, W, 3)`` arrays."""

    #: name used by :func:`get_scrambler` and the command line (``--scheme``)
    name = "scrambler"

    def __init__(self, block_size: int = 4, image_size: Optional[ImageSize] = None):
        if int(block_size) < 1:
            raise ValueError("block_size must be a positive integer")
        self.block_size = int(block_size)
        self.image_size = _hw(image_size)

    @property
    def invertible(self) -> bool:
        """Whether :meth:`inverse` is defined."""
        return True

    def __call__(self, img):
        """Scramble ``img`` (PIL image, channels-last ndarray or channels-first tensor; one image or a batch)."""
        x, restore = as_nhwc(img)
        return restore(self._forward(self._check(x)))

    def inverse(self, img):
        """Undo :meth:`__call__` with the same key.

        Exact for ``uint8`` images and for LE/ELE/block shuffling; float images that went through a
        negative-positive transform (PE, EtC) come back within one float32 rounding step (``1 - (1 - x)``).
        """
        if not self.invertible:
            raise ValueError(self._why_not_invertible())
        x, restore = as_nhwc(img)
        return restore(self._inverse(self._check(x)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self._repr_args().items())})"

    # -- to be implemented by subclasses ---------------------------------------------------------------------------
    def _forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _inverse(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _repr_args(self) -> dict:
        return {"block_size": self.block_size}

    def _why_not_invertible(self) -> str:
        return f"{type(self).__name__} is not invertible"

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _check(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.uint8 and not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"images must be uint8 (0..255) or float (0..1), got dtype {x.dtype}")
        _, h, w, c = x.shape
        if c != 3:
            raise ValueError(f"{type(self).__name__} needs RGB images (3 channels), got {c}")
        if self.image_size is not None and (h, w) != self.image_size:
            raise ValueError(f"{type(self).__name__} was built for {self.image_size[0]}x{self.image_size[1]} images, "
                             f"got {h}x{w}; pass image_size=({h}, {w}) (and a seed or explicit key)")
        if h % self.block_size or w % self.block_size:
            raise ValueError(f"image size {h}x{w} is not a multiple of the block size {self.block_size}")
        return x

    def _n_blocks(self) -> int:
        h, w = self.image_size
        if h % self.block_size or w % self.block_size:
            raise ValueError(f"image size {h}x{w} is not a multiple of the block size {self.block_size}")
        return (h // self.block_size) * (w // self.block_size)


class LE(Scrambler):
    """Learnable image encryption (Tanaka, ICCE-TW 2018): the same key in every ``B x B`` block, no block shuffling.

    Port of ``BlockScramble`` (``archive/learnable_encryption.py``, after github.com/mastnk/ICCE-TW2018) as applied
    by ``blockwise_scramble`` of ``archive/Blockwise_scramble_LE.py``. ``key``: ``"paper"`` (``key4/0_.pkl``, the key
    of the SIA-GAN experiments), an int seed, or a permutation of ``0 .. 2 B^2 3 - 1`` (96 entries for 4x4 blocks).
    Works for any image whose sides are multiples of ``block_size``.
    """

    name = "le"

    def __init__(self, key: Union[str, int, np.ndarray] = "paper", block_size: int = 4):
        super().__init__(block_size)
        d = 2 * self.block_size ** 2 * 3
        if _is_paper(key):
            if self.block_size != 4:
                raise ValueError('key="paper" holds keys of 4x4 blocks; use an int seed for other block sizes')
            k = paper_keys()[0]
        elif _is_seed(key):
            k = le_keys_from_seed(key, 1, self.block_size)[0]
        else:
            k = key
        self.key = _check_perm(k, d, "LE key")
        self._key_desc = key if (_is_paper(key) or _is_seed(key)) else "custom"

    def _repr_args(self):
        return {"key": self._key_desc, "block_size": self.block_size}

    def _forward(self, x):
        return _le_transform(x, self.key, self.block_size)

    def _inverse(self, x):
        return _le_transform(x, self.key, self.block_size, inverse=True)


class ELE(Scrambler):
    """Extended learnable encryption (Madono et al., AAAI-20 WS): the LE operation with a different key in every
    block, then block location shuffling.

    ``key``: ``"paper"`` (``key4/0..63_.pkl`` for the 64 blocks of a 32x32 image and the block order of
    ``random.seed(30)``), an int seed (per-block keys from ``np.random.RandomState(seed)``, block order from
    ``random.Random(seed)``) or a tuple ``(keys, perm)`` with ``keys`` of shape ``(nb, 2 B^2 3)``. The key depends on
    the number of blocks: pass ``image_size`` for images other than 32x32.
    """

    name = "ele"

    def __init__(self, key: Union[str, int, tuple] = "paper", block_size: int = 4, image_size: ImageSize = 32):
        super().__init__(block_size, image_size)
        nb, d = self._n_blocks(), 2 * self.block_size ** 2 * 3
        if _is_paper(key):
            if self.block_size != 4 or nb != 64:
                raise ValueError('key="paper" holds the keys of 32x32 images with 4x4 blocks; use an int seed')
            keys, perm = paper_keys(), block_perm_from_seed(PAPER_SEED, nb)
        elif _is_seed(key):
            keys, perm = le_keys_from_seed(key, nb, self.block_size), block_perm_from_seed(key, nb)
        else:
            keys, perm = key
        keys = np.asarray(keys, dtype=np.int64)
        if keys.shape != (nb, d):
            raise ValueError(f"ELE keys must have shape {(nb, d)}, got {keys.shape}")
        self.keys = np.stack([_check_perm(k, d, "LE key") for k in keys])
        self.perm = _check_perm(perm, nb, "block permutation")
        self._key_desc = key if (_is_paper(key) or _is_seed(key)) else "custom"

    def _repr_args(self):
        return {"key": self._key_desc, "block_size": self.block_size, "image_size": self.image_size}

    def _forward(self, x):
        return _shuffle_blocks(_le_transform(x, self.keys, self.block_size), self.perm, self.block_size)

    def _inverse(self, x):
        y = _shuffle_blocks(x, np.argsort(self.perm), self.block_size)
        return _le_transform(y, self.keys, self.block_size, inverse=True)


class BlockShuffle(Scrambler):
    """Block location shuffling alone: output block ``i`` is input block ``perm[i]``.

    ``key``: ``"paper"`` (the block order of the ELE experiments, ``random.seed(30)``), an int seed
    (``random.seed(s); random.shuffle(list(range(nb)))``) or a permutation.
    """

    name = "blockshuffle"

    def __init__(self, key: Union[str, int, np.ndarray] = "paper", block_size: int = 4, image_size: ImageSize = 32):
        super().__init__(block_size, image_size)
        nb = self._n_blocks()
        if _is_paper(key):
            perm = block_perm_from_seed(PAPER_SEED, nb)
        elif _is_seed(key):
            perm = block_perm_from_seed(key, nb)
        else:
            perm = key
        self.perm = _check_perm(perm, nb, "block permutation")
        self._key_desc = key if (_is_paper(key) or _is_seed(key)) else "custom"

    def _repr_args(self):
        return {"key": self._key_desc, "block_size": self.block_size, "image_size": self.image_size}

    def _forward(self, x):
        return _shuffle_blocks(x, self.perm, self.block_size)

    def _inverse(self, x):
        return _shuffle_blocks(x, np.argsort(self.perm), self.block_size)


#: EtC colour codes: output channel ``c`` of a block is input channel ``TABLE[code][c]``. ``"original"`` is what
#: ``channel_change`` of the original etc_encryption.py computes (in-place assignment through aliased views
#: duplicates a channel for codes 1-5); ``"permutation"`` is the permutation each code was written to perform.
ETC_CHANNEL_TABLES = {
    "original": np.array([[0, 1, 2], [1, 1, 2], [2, 1, 2], [0, 2, 2], [2, 2, 2], [1, 2, 1]]),
    "permutation": np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [2, 0, 1], [1, 2, 0]]),
}
#: PE colour codes, the same convention for ``pixel_based_encryption`` of the ScrambleMix code.
PE_CHANNEL_TABLES = {
    "original": np.array([[0, 1, 2], [0, 2, 2], [1, 1, 2], [1, 2, 1], [2, 2, 2], [2, 1, 2]]),
    "permutation": np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]),
}
_ETC_ROT = np.array([1, 2, 3, 0])  # rotate code -> quarter turns (codes 0, 1, 2 = 90, 180, 270 degrees; 3 = none)
_ETC_FLIP = np.array([1, 2, 0])  # reverse code -> _flip_blocks mode (0 = rows, 1 = columns, 2 = none)
_ETC_FIELDS = ("rotate", "negaposi", "reverse", "channel", "shuffle")


def _check_channel_mode(mode: str) -> str:
    if mode not in ("original", "permutation"):
        raise ValueError("channel_mode must be 'original' (as the original code) or 'permutation'")
    return mode


class EtC(Scrambler):
    """Encryption-then-compression block scrambling (Chuman, Kurihara and Kiya 2018) as implemented in the
    block-wise scrambling code (etc_encryption.py).

    Every block is rotated, negative-positive transformed, flipped ("inversion") and colour shuffled with its own
    parameters; then the block locations are shuffled. Key encoding of the original code (one entry per block):
    ``rotate`` 0/1/2 = 90/180/270 degrees, 3 = none; ``negaposi`` 0 = invert, 1 = keep (half of the blocks);
    ``reverse`` 0 = flip rows, 1 = flip columns, 2 = none; ``channel`` 0..5 (see ``channel_mode``); ``shuffle``:
    output block ``i`` = input block ``shuffle[i]``.

    ``key``: ``"paper"`` (= seed 30, the key of the experiments), an int seed or a dict with these five entries.
    ``channel_mode="original"`` (default) reproduces the original code, which duplicates colour channels and cannot
    be inverted; ``"permutation"`` is the intended colour permutation (invertible).
    """

    name = "etc"

    def __init__(self, key: Union[str, int, dict] = "paper", block_size: int = 4, image_size: ImageSize = 32,
                 channel_mode: str = "original"):
        super().__init__(block_size, image_size)
        self.channel_mode = _check_channel_mode(channel_mode)
        nb = self._n_blocks()
        if _is_paper(key):
            k = etc_key_from_seed(PAPER_SEED, nb)
        elif _is_seed(key):
            k = etc_key_from_seed(key, nb)
        else:
            k = {name: key[name] for name in _ETC_FIELDS}
        k = {name: np.asarray(v, dtype=np.int64).reshape(-1) for name, v in k.items()}
        for name, hi in (("rotate", 3), ("negaposi", 1), ("reverse", 2), ("channel", 5)):
            if k[name].size != nb or k[name].min() < 0 or k[name].max() > hi:
                raise ValueError(f"EtC key '{name}' needs {nb} entries in 0..{hi}")
        self.rotate, self.negaposi, self.reverse, self.channel = k["rotate"], k["negaposi"], k["reverse"], k["channel"]
        self.shuffle = _check_perm(k["shuffle"], nb, "EtC shuffle")
        self._key_desc = key if (_is_paper(key) or _is_seed(key)) else "custom"

    @property
    def invertible(self) -> bool:
        return self.channel_mode == "permutation" or not np.any(self.channel)

    def _why_not_invertible(self):
        return ("EtC with channel_mode='original' duplicates colour channels (as the original code) and cannot be "
                "inverted; use channel_mode='permutation' for an invertible EtC")

    def _repr_args(self):
        return {"key": self._key_desc, "block_size": self.block_size, "image_size": self.image_size,
                "channel_mode": self.channel_mode}

    def _invert_blocks(self, x):
        blocks = _to_blocks(x, self.block_size)
        mask = (self.negaposi == 0)[None, :, None, None, None]
        return _from_blocks(np.where(mask, _max_value(x) - blocks, blocks).astype(x.dtype), x.shape[1], x.shape[2])

    def _forward(self, x):
        b = self.block_size
        y = self._invert_blocks(_rotate_blocks(x, _ETC_ROT[self.rotate], b))
        y = _flip_blocks(y, _ETC_FLIP[self.reverse], b)
        y = _gather_channels(y, ETC_CHANNEL_TABLES[self.channel_mode][self.channel], b)
        return _shuffle_blocks(y, self.shuffle, b)

    def _inverse(self, x):
        b = self.block_size
        y = _shuffle_blocks(x, np.argsort(self.shuffle), b)
        y = _gather_channels(y, np.argsort(ETC_CHANNEL_TABLES[self.channel_mode][self.channel], axis=1), b)
        y = self._invert_blocks(_flip_blocks(y, _ETC_FLIP[self.reverse], b))
        return _rotate_blocks(y, (4 - _ETC_ROT[self.rotate]) % 4, b)


class PE(Scrambler):
    """Pixel-based image encryption (Sirichotedumrong, Maekawa, Kinoshita and Kiya 2019): negative-positive transform
    of individual values followed by colour-component shuffling of individual pixels, with one fixed key.

    Key encoding of the original code: ``inv`` - ``H*W*3`` values (row, column, channel order), 0 = invert, 1 = keep;
    ``colors`` - ``H*W`` colour codes 0..5 (see ``channel_mode``, as for :class:`EtC`). ``key``: ``"paper"``
    (:func:`paper_pe_key`, 32x32 images), an int seed or a tuple ``(inv, colors)``.
    """

    name = "pe"

    def __init__(self, key: Union[str, int, tuple] = "paper", image_size: ImageSize = 32,
                 channel_mode: str = "original"):
        super().__init__(1, image_size)
        self.channel_mode = _check_channel_mode(channel_mode)
        h, w = self.image_size
        if _is_paper(key):
            if (h, w) != (32, 32):
                raise ValueError('key="paper" is a key for 32x32 images; use an int seed for other sizes')
            inv, colors = paper_pe_key()
        elif _is_seed(key):
            inv, colors = pe_key_from_seed(key, h, w)
        else:
            inv, colors = key
        inv, colors = np.asarray(inv, dtype=np.int64).reshape(-1), np.asarray(colors, dtype=np.int64).reshape(-1)
        if inv.size != h * w * 3 or not np.isin(inv, (0, 1)).all():
            raise ValueError(f"PE 'inv' needs {h * w * 3} values in {{0, 1}}")
        if colors.size != h * w or colors.min() < 0 or colors.max() > 5:
            raise ValueError(f"PE 'colors' needs {h * w} values in 0..5")
        self.inv, self.colors = inv, colors
        self._key_desc = key if (_is_paper(key) or _is_seed(key)) else "custom"

    @property
    def invertible(self) -> bool:
        return self.channel_mode == "permutation" or not np.any(self.colors)

    def _why_not_invertible(self):
        return ("PE with channel_mode='original' duplicates colour channels (as the original code) and cannot be "
                "inverted; use channel_mode='permutation' for an invertible PE")

    def _repr_args(self):
        return {"key": self._key_desc, "image_size": self.image_size, "channel_mode": self.channel_mode}

    def _mask(self):
        h, w = self.image_size
        return (self.inv == 0).reshape(1, h, w, 3)

    def _forward(self, x):
        y = _negative_positive(x, self._mask())
        return _gather_channels(y, PE_CHANNEL_TABLES[self.channel_mode][self.colors], 1)

    def _inverse(self, x):
        y = _gather_channels(x, np.argsort(PE_CHANNEL_TABLES[self.channel_mode][self.colors], axis=1), 1)
        return _negative_positive(y, self._mask())


class RandomPE(Scrambler):
    """Random PE: the negative-positive transform of ``ratio`` of the values with a fresh random key for every image
    and every call (no key is kept, so it is not invertible).

    ``random_mask`` of the original code (``archive/gan_attack_alpha_10.py``) inverts exactly half of the 3072 values
    of every image at random positions and applies no colour shuffling; ``channel_shuffle=True`` adds a random
    per-pixel colour permutation (the random PE of Table 1). ``seed`` makes the stream of keys reproducible.
    """

    name = "randompe"

    def __init__(self, ratio: float = 0.5, channel_shuffle: bool = False, seed: Optional[int] = None):
        super().__init__(1)
        if not 0.0 <= float(ratio) <= 1.0:
            raise ValueError("ratio must be in [0, 1]")
        self.ratio, self.channel_shuffle, self.seed = float(ratio), bool(channel_shuffle), seed
        self.rng = np.random.default_rng(seed)

    @property
    def invertible(self) -> bool:
        return False

    def _why_not_invertible(self):
        return "RandomPE draws a new key for every image and keeps none"

    def _repr_args(self):
        return {"ratio": self.ratio, "channel_shuffle": self.channel_shuffle, "seed": self.seed}

    def _forward(self, x):
        n, h, w, c = x.shape
        size = h * w * c
        k = int(round(self.ratio * size))
        out = np.empty_like(x)
        for i in range(n):
            mask = np.zeros(size, dtype=bool)
            mask[self.rng.permutation(size)[:k]] = True
            y = _negative_positive(x[i:i + 1], mask.reshape(1, h, w, c))
            if self.channel_shuffle:
                y = _gather_channels(y, PE_CHANNEL_TABLES["permutation"][self.rng.integers(0, 6, size=h * w)], 1)
            out[i] = y[0]
        return out


#: ``--scheme`` name -> class; the five schemes of Tables 2-3 and block shuffling alone.
SCHEMES = {"pe": PE, "randompe": RandomPE, "le": LE, "ele": ELE, "etc": EtC, "blockshuffle": BlockShuffle}


def get_scrambler(name: str, key: Union[str, int] = "paper", **kwargs) -> Scrambler:
    """Build a scheme by name (``pe``, ``randompe``, ``le``, ``ele``, ``etc``, ``blockshuffle``).

    ``key`` is ``"paper"`` or an int seed (for ``randompe`` an int seeds the stream of random keys and ``"paper"``
    leaves it unseeded); extra keyword arguments go to the class.
    """
    cls = SCHEMES.get(str(name).lower().replace("-", "").replace("_", ""))
    if cls is None:
        raise ValueError(f"unknown scheme {name!r}; choose from {sorted(SCHEMES)}")
    if cls is RandomPE:
        return RandomPE(seed=key if _is_seed(key) else kwargs.pop("seed", None), **kwargs)
    return cls(key, **kwargs)
