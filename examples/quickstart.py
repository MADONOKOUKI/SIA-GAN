"""Quick start: scramble an image with the five schemes attacked in the paper, then run a few SIA-GAN training steps.

    pip install -e ".[examples]"        # adds matplotlib and scikit-image (figure and sample image)
    python examples/quickstart.py       # writes assets/quickstart.png; a few seconds on a CPU, no downloads

The cat image ships with scikit-image. The SIA-GAN part trains on random tensors with a small discriminator, only to
show the API; `python attack.py` trains the attack on CIFAR with the paper's settings.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from skimage import data  # noqa: E402

import siagan  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "quickstart.png"

# 1) A CIFAR-sized image: 32 x 32 pixels = 8 x 8 blocks of 4 x 4 pixels, as in the paper's experiments.
cat = Image.fromarray(data.chelsea()[:, 75:375]).resize((32, 32), Image.BICUBIC)
img = np.asarray(cat)  # (32, 32, 3) uint8

# 2) The scrambling schemes of Tables 1-3, with the keys of the original experiments (key="paper", the default).
schemes = {
    "PE": siagan.PE(),
    "random PE": siagan.RandomPE(seed=0),  # a new random key for every image
    "LE": siagan.LE(),
    "ELE": siagan.ELE(),
    "EtC": siagan.EtC(),
}
block_shuffling = {"ELE", "EtC"}  # the paper: block shuffling is what resists SIA-GAN (Tables 2-3)
fig, axes = plt.subplots(1, len(schemes) + 1, figsize=(12.6, 2.9))
axes[0].imshow(img, interpolation="nearest")
axes[0].set_title("original\n(32 x 32)", fontsize=11)
print(f"{'scheme':10s} {'PSNR':>6s} {'SSIM':>6s}   (scrambled vs. original, this image)")
for ax, (name, scheme) in zip(axes[1:], schemes.items()):
    scrambled = scheme(img)
    ax.imshow(scrambled, interpolation="nearest")
    ax.set_title(f"{name}\n" + ("with block shuffling" if name in block_shuffling else "no block shuffling"),
                 fontsize=11, color="#1a7f37" if name in block_shuffling else "#b42318")
    print(f"{name:10s} {siagan.psnr(img, scrambled):6.2f} {siagan.ssim(img, scrambled):6.3f}   {scheme!r}")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Image scrambling schemes attacked by SIA-GAN (keys of the paper's experiments)", fontsize=12)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=100)
print(f"saved {OUT}")

# 3) A few SIA-GAN steps on synthetic data. With CIFAR-10 you would pass a DataLoader of (image, label) batches.
torch.manual_seed(0)
x, y = torch.rand(32, 3, 32, 32), torch.randint(0, 10, (32,))  # stand-in for a CIFAR-10 training set
loader = [(x[i:i + 16], y[i:i + 16]) for i in range(0, 32, 16)]  # each batch: 8 real + 8 scrambled images
attack = siagan.SIAGAN(num_classes=10, discriminator_width=64, classifier=None, device="cpu")  # small, for speed
history = attack.fit(loader, siagan.LE(), epochs=3, log=None)
for h in history:
    print(f"epoch {h['epoch']}: d_loss {h['d_loss']:.3f}  g_adv {h['g_adv']:.3f}")
recovered = attack.reconstruct(siagan.LE()(x[:4]))  # Eq. 5: x = G(scrambled), images in [0, 1]
print("recovered:", tuple(recovered.shape), recovered.dtype)
print("evaluation:", attack.evaluate(loader, siagan.LE(), metrics=("psnr", "ssim")))

paper = siagan.SIAGAN(num_classes=10, device="cpu")  # the paper's settings (Sec. V-A; the default)
sizes = {k: sum(p.numel() for p in getattr(paper, k).parameters()) / 1e6
         for k in ("generator", "discriminator", "classifier")}
print("paper settings: " + ", ".join(f"{k} {v:.2f}M" for k, v in sizes.items()) + " parameters")
