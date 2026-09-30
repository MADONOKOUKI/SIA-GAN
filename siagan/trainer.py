"""SIA-GAN training, inversion and evaluation (Sec. IV-D, IV-E and V).

>>> import siagan
>>> attack = siagan.SIAGAN(num_classes=10)                      # paper settings (Sec. V-A)
>>> attack.fit(train_loader, siagan.LE(), epochs=100)           # (image, label) batches, images in [0, 1]
>>> recovered = attack.reconstruct(siagan.LE()(test_images))    # Eq. 5: x = G(x^)
>>> attack.evaluate(test_loader, siagan.LE())                   # PSNR / SSIM / LPIPS vs. the original images

Training data. The attacker has scrambled images with their labels and natural images of a similar domain, but no
key (Sec. II). Without ``real_loader`` every mini-batch is split as in the original code: the first half are the
"real" images and the second half is scrambled and fed to the generator (the "Same" rows of Tables 2-3). With
``real_loader`` (e.g. CIFAR-100) the whole batch is scrambled and the real images come from that loader ("Diff.").

One step (:meth:`SIAGAN.train_step`) follows the loop of ``archive/gan_attack_alpha_10.py``: the discriminator is
updated with the hinge loss (Eq. 3) on real and generated images, then the generator with the adversarial loss
(Eq. 4) plus - as in the code - the cross entropy of a ShakeDrop PyramidNet-110 classifier trained jointly on the
generated images (``lambda_cls``; ``classifier=None`` gives the adversarial loss alone, the paper's Eq. 1).

Defaults: Adam with beta = (0, 0.9), learning rate 1e-4 for the generator (adaptation network + feature decoder)
and 4e-4 for the discriminator (Sec. V-A). The code variants disagree with each other here (gan_attack.py: 2e-4
for both with betas (0.9, 0.999) / (0.5, 0.999); gan_attack_alpha_10.py: 1e-4 with betas (0.5, 0.9) / 4e-4 with
(0, 0.9)), so the paper's values are used, as in the scramblekit library.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .classifier import ShakePyramidNet
from .losses import discriminator_hinge_loss, feature_total_variation, generator_hinge_loss
from .networks import Discriminator, Generator
from .scrambling import as_nhwc

__all__ = ["SIAGAN", "pick_device"]


def pick_device(device=None) -> torch.device:
    """``None``/``"auto"`` -> cuda > mps > cpu; otherwise ``torch.device(device)``."""
    if device is not None and str(device) != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SIAGAN:
    """The SIA-GAN attack: generator (adaptation network + feature decoder) vs. discriminator.

    Images are float tensors in [0, 1] ``(N, 3, H, W)``; the networks work in [-1, 1] as in the code.

    Args:
        num_classes: classes of the scrambled images' labels (for the classifier term).
        block_size: block size of the adaptation network: 4 for LE, ELE and EtC, 1 for PE and random PE (Sec. V).
        adaptation: ``False`` removes the adaptation network (the scrambled image goes straight into the decoder).
        classifier: ``"shakepyramidnet"`` (default, as the code), an ``nn.Module`` taking ``(N, 3, H, W)`` images,
            or ``None`` (adversarial loss only).
        classifier_depth / classifier_alpha: size of that ShakeDrop PyramidNet (110 / 270 in the code).
        lambda_cls: weight of the classifier's cross entropy (1 in the code).
        lambda_tv: weight of ``feature_total_variation`` (1e-3 in gan_attack_alpha_10.py; 0 = off, the default).
        mixup_alpha: mixup of the scrambled inputs, Beta(alpha, alpha) (1 in gan_attack.py; 0 = off, the default).
        lr_g, lr_d, betas: Adam settings (Sec. V-A).
        discriminator_sn: ``"original"`` (in-place spectral normalisation of the code) or ``"torch"``.
        discriminator_width: channels of the last discriminator stage (1024 in Table 6).
        device: ``None``/``"auto"`` picks cuda > mps > cpu.
    """

    def __init__(self, num_classes: int = 10, *, block_size: int = 4, image_size: int = 32, adaptation: bool = True,
                 classifier="shakepyramidnet", classifier_depth: int = 110, classifier_alpha: int = 270,
                 lambda_cls: float = 1.0, lambda_tv: float = 0.0, mixup_alpha: float = 0.0, lr_g: float = 1e-4,
                 lr_d: float = 4e-4, betas=(0.0, 0.9), discriminator_sn: str = "original",
                 discriminator_width: int = 1024, device=None):
        self.device = pick_device(device)
        self.generator = Generator(block_size, image_size, adaptation).to(self.device)
        self.discriminator = Discriminator(discriminator_width, image_size, discriminator_sn).to(self.device)
        if classifier is None or lambda_cls == 0:
            self.classifier, clf_name = None, None
        elif isinstance(classifier, nn.Module):
            self.classifier, clf_name = classifier.to(self.device), "custom"
        elif str(classifier).lower() == "shakepyramidnet":
            self.classifier = ShakePyramidNet(classifier_depth, classifier_alpha, num_classes).to(self.device)
            clf_name = "shakepyramidnet"
        else:
            raise ValueError("classifier must be 'shakepyramidnet', an nn.Module or None")
        params = list(self.generator.parameters())
        if self.classifier is not None:  # the code updates the classifier with the generator's optimiser
            params += list(self.classifier.parameters())
        self.opt_g = torch.optim.Adam(params, lr=lr_g, betas=tuple(betas))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=tuple(betas))
        self.lambda_cls, self.lambda_tv, self.mixup_alpha = float(lambda_cls), float(lambda_tv), float(mixup_alpha)
        self.config = dict(num_classes=num_classes, block_size=block_size, image_size=image_size,
                           adaptation=adaptation, classifier=clf_name, classifier_depth=classifier_depth,
                           classifier_alpha=classifier_alpha, lambda_cls=lambda_cls, lambda_tv=lambda_tv,
                           mixup_alpha=mixup_alpha, lr_g=lr_g, lr_d=lr_d, betas=list(betas),
                           discriminator_sn=discriminator_sn, discriminator_width=discriminator_width)
        self.step = 0   # optimisation steps done
        self.epoch = 0  # epochs completed by fit()

    def __repr__(self) -> str:
        c = self.config
        return (f"SIAGAN(block_size={c['block_size']}, adaptation={c['adaptation']}, classifier={c['classifier']}, "
                f"device={self.device})")

    # -- training ----------------------------------------------------------------------------------------------------
    def train(self, mode: bool = True) -> "SIAGAN":
        for m in (self.generator, self.discriminator, self.classifier):
            if m is not None:
                m.train(mode)
        return self

    def train_step(self, scrambled: torch.Tensor, labels: Optional[torch.Tensor], real: torch.Tensor,
                   scrambled2: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """One discriminator and one generator update, in the order of the original loop.

        ``scrambled``: generator inputs in [0, 1]; ``labels``: their classes (for the classifier term); ``real``:
        natural images in [0, 1]; ``scrambled2``: second scrambled copy for mixup (default: ``scrambled``).
        Returns the losses of the step.
        """
        dev = self.device
        z = scrambled.to(dev).float() * 2 - 1
        real = real.to(dev).float() * 2 - 1
        y = labels.to(dev) if labels is not None else None
        lam, index = 1.0, None
        if self.mixup_alpha > 0 and self.classifier is not None:  # gan_attack.py: mixup of the scrambled inputs
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            index = torch.randperm(z.size(0), device=dev)
            z2 = z if scrambled2 is None else scrambled2.to(dev).float() * 2 - 1
            z = lam * z + (1 - lam) * z2[index]
        gen, feature = self.generator(z)
        logits = self.classifier(gen) if self.classifier is not None else None

        d_loss = discriminator_hinge_loss(self.discriminator(real), self.discriminator(gen.detach()))  # Eq. 3
        self.opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        self.opt_d.step()

        g_adv = generator_hinge_loss(self.discriminator(gen))  # Eq. 4
        g_loss, stats = g_adv, {}
        if logits is not None and y is not None:
            ce = F.cross_entropy(logits, y)
            if index is not None:
                ce = lam * ce + (1 - lam) * F.cross_entropy(logits, y[index])
            g_loss = g_loss + self.lambda_cls * ce
            stats["cls_loss"] = float(ce.detach())
            stats["cls_acc"] = float((logits.argmax(1) == y).float().mean())
        if self.lambda_tv > 0:
            g_loss = g_loss + self.lambda_tv * feature_total_variation(feature)
        self.opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        self.opt_g.step()
        self.step += 1
        stats.update(d_loss=float(d_loss.detach()), g_adv=float(g_adv.detach()), g_loss=float(g_loss.detach()))
        return stats

    def fit(self, loader: Iterable, scrambler, epochs: int = 100, real_loader: Optional[Iterable] = None,
            log_every: int = 100, max_steps: Optional[int] = None, log=print) -> List[Dict[str, float]]:
        """Train for ``epochs`` more epochs over ``loader`` (batches ``(images, labels)``, images in [0, 1]).

        ``scrambler`` is applied to every generator input (e.g. ``siagan.LE()``). Without ``real_loader`` each batch
        is split into real images (first half) and generator inputs (second half); with ``real_loader`` the whole
        batch is scrambled and the real images come from ``real_loader``. Returns the per-epoch mean losses.
        """
        history = []
        for _ in range(epochs):
            self.train(True)
            t0, sums, count = time.time(), {}, 0
            real_iter = iter(real_loader) if real_loader is not None else None
            for images, labels in loader:
                if real_iter is None:
                    half = images.shape[0] // 2
                    if half == 0:
                        continue
                    real, x, y = images[:half], images[half:], labels[half:]
                else:
                    try:
                        real = next(real_iter)
                    except StopIteration:
                        real_iter = iter(real_loader)
                        real = next(real_iter)
                    real = real[0] if isinstance(real, (list, tuple)) else real
                    x, y = images, labels
                stats = self.train_step(scrambler(x), y, real)
                for k, v in stats.items():
                    sums[k] = sums.get(k, 0.0) + v
                count += 1
                if log is not None and log_every and self.step % log_every == 0:
                    log(f"[siagan] epoch {self.epoch + 1} step {self.step} "
                        + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
                if max_steps is not None and self.step >= max_steps:
                    break
            self.epoch += 1
            history.append({"epoch": self.epoch, "steps": self.step, "seconds": round(time.time() - t0, 2),
                            **{k: v / max(count, 1) for k, v in sums.items()}})
            if max_steps is not None and self.step >= max_steps:
                break
        return history

    # -- inversion and evaluation ------------------------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, scrambled, batch_size: int = 256):
        """Invert scrambled images with the generator (Eq. 5, ``x = G(x^)``), in evaluation mode.

        ``scrambled``: a float tensor ``(N, 3, H, W)`` / ``(3, H, W)`` in [0, 1] (returns a float tensor on the CPU),
        or a PIL image / channels-last numpy array / uint8 tensor (returns the same type, layout and dtype).
        """
        if isinstance(scrambled, torch.Tensor) and scrambled.is_floating_point():
            single = scrambled.dim() == 3
            out = self._reconstruct(scrambled.unsqueeze(0) if single else scrambled, batch_size)
            return out[0] if single else out
        x, restore = as_nhwc(scrambled)
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.float32))
        t = t / 255.0 if x.dtype == np.uint8 else t
        y = self._reconstruct(t, batch_size).permute(0, 2, 3, 1).numpy()
        y = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8) if x.dtype == np.uint8 else y.astype(x.dtype)
        return restore(y)

    def _reconstruct(self, s: torch.Tensor, batch_size: int) -> torch.Tensor:
        self.generator.eval()
        outs = []
        for i in range(0, s.shape[0], batch_size):
            g, _ = self.generator(s[i:i + batch_size].to(self.device).float() * 2 - 1)
            outs.append(((g + 1) / 2).clamp(0, 1).cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def evaluate(self, loader: Iterable, scrambler, metrics: Sequence[str] = ("psnr", "ssim", "lpips"),
                 max_batches: Optional[int] = None, lpips_normalize: bool = False) -> Dict[str, object]:
        """Similarity to the original images of the scrambled inputs and of the SIA-GAN reconstructions (Table 3).

        Returns ``{"scrambled": {...}, "reconstructed": {...}, "num_images": n}`` with the mean PSNR (capped at
        100 dB), SSIM and LPIPS (AlexNet, ``normalize=False`` as the original evaluation). Higher PSNR/SSIM and lower
        LPIPS of the reconstructions mean a more successful attack.
        """
        from .metrics import LPIPS, psnr, ssim

        lp = LPIPS(net="alex", normalize=lpips_normalize, device=self.device) if "lpips" in metrics else None
        sums = {"scrambled": {}, "reconstructed": {}}
        n = 0
        for b, batch in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            s = scrambler(x)
            r = self.reconstruct(s)
            for name, img in (("scrambled", s), ("reconstructed", r)):
                acc = sums[name]
                if "psnr" in metrics:
                    acc["psnr"] = acc.get("psnr", 0.0) + float(np.sum(np.minimum(psnr(x, img, "none"), 100.0)))
                if "ssim" in metrics:
                    acc["ssim"] = acc.get("ssim", 0.0) + float(np.sum(ssim(x, img, reduction="none")))
                if lp is not None:
                    acc["lpips"] = acc.get("lpips", 0.0) + float(np.sum(lp(x, img)))
            n += x.shape[0]
        out: Dict[str, object] = {k: {m: v / max(n, 1) for m, v in d.items()} for k, d in sums.items()}
        out["num_images"] = n
        return out

    # -- persistence -------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"config": self.config, "step": self.step, "epoch": self.epoch,
                "generator": self.generator.state_dict(), "discriminator": self.discriminator.state_dict(),
                "classifier": None if self.classifier is None else self.classifier.state_dict(),
                "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        if self.classifier is not None and state.get("classifier") is not None:
            self.classifier.load_state_dict(state["classifier"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.step, self.epoch = state.get("step", 0), state.get("epoch", 0)

    def save(self, path: str) -> None:
        """Save networks, optimisers and counters (resume with :meth:`load`)."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device=None, classifier: Optional[nn.Module] = None) -> "SIAGAN":
        """Rebuild an attack saved with :meth:`save` (pass ``classifier`` if a custom module was used)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(state["config"])
        name = cfg.pop("classifier", None)
        if name == "custom":
            if classifier is None:
                raise ValueError("this checkpoint used a custom classifier module: pass it as classifier=...")
            name = classifier
        obj = cls(**cfg, classifier=name, device=device)
        obj.load_state_dict(state)
        return obj
