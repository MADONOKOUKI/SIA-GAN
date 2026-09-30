#!/usr/bin/env python
"""SIA-GAN: train the attack on scrambled images, invert the test images and evaluate (Tables 2 and 3).

One run = one cell of Table 3 (LPIPS between the original and the recovered CIFAR-10 test images; the "Scrambled
image" row is reported as well) and one column of Table 2 (samples.png). All defaults are the paper's settings.

    python attack.py --scheme le                                # LE, with adaptation network, real images: CIFAR-10
    python attack.py --scheme ele --real-dataset cifar100       # real images from another dataset ("Diff.")
    python attack.py --scheme pe --no-adaptation                # generator without the adaptation network
    python attack.py --dataset fake --epochs 1 --batch-size 8 --ndf 64 --classifier none --no-lpips  # smoke test

Outputs in --out-dir (default runs/<dataset>_<scheme>_<real>_<adapt|noadapt>): history.csv (losses per epoch),
last.pt (checkpoint; --resume continues from it), metrics.json (PSNR / SSIM / LPIPS of the scrambled and of the
recovered test images) and samples.png (original / scrambled / recovered test images).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import siagan

HISTORY_FIELDS = ["epoch", "steps", "seconds", "d_loss", "g_adv", "g_loss", "cls_loss", "cls_acc",
                  "test_psnr", "test_ssim", "test_lpips"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("experiment (one cell of Table 3)")
    g.add_argument("--scheme", default="le", choices=list(siagan.SCHEMES),
                   help="scrambling attacked: pe | randompe | le | ele | etc (Tables 2-3), or blockshuffle alone")
    g.add_argument("--real-dataset", default=None, choices=["cifar10", "cifar100", "fake"],
                   help="natural images of the attacker. Default: the same dataset as --dataset ('Same' rows: the "
                        "first half of every batch, as the original code); cifar100 = the 'Diff.' rows")
    g.add_argument("--no-adaptation", action="store_true",
                   help="generator without the adaptation network (the rows without a check mark)")
    g.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "fake"],
                   help="scrambled images (CIFAR-10 in the paper); fake = random images, no download (smoke tests)")
    g.add_argument("--key", default="paper",
                   help="'paper' (the keys of the original experiments) or an integer seed for new keys")
    g.add_argument("--channel-mode", default="original", choices=["original", "permutation"],
                   help="PE/EtC colour shuffling: 'original' reproduces the original code (duplicated channels)")

    g = p.add_argument_group("training (defaults = paper, Sec. V-A)")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr-g", type=float, default=1e-4, help="Adam learning rate of the generator")
    g.add_argument("--lr-d", type=float, default=4e-4, help="Adam learning rate of the discriminator")
    g.add_argument("--betas", type=float, nargs=2, default=[0.0, 0.9], help="Adam betas")
    g.add_argument("--block-size", type=int, default=None,
                   help="block size of the adaptation network (default: 1 for pe/randompe, 4 otherwise, Sec. V)")
    g.add_argument("--classifier", default="shakepyramidnet", choices=["shakepyramidnet", "none"],
                   help="classifier trained on the generated images, as the original code; none = the paper's Eq. 1")
    g.add_argument("--classifier-depth", type=int, default=110)
    g.add_argument("--classifier-alpha", type=int, default=270)
    g.add_argument("--lambda-cls", type=float, default=1.0, help="weight of the classifier's cross entropy")
    g.add_argument("--lambda-tv", type=float, default=0.0, help="feature TV term of gan_attack_alpha_10.py (1e-3)")
    g.add_argument("--mixup-alpha", type=float, default=0.0, help="mixup of scrambled inputs of gan_attack.py (1.0)")
    g.add_argument("--discriminator-sn", default="original", choices=["original", "torch"],
                   help="spectral normalisation of the discriminator: in place as the original code, or torch's")
    g.add_argument("--ndf", type=int, default=1024, help="discriminator width (Table 6: 1024; smaller for smoke tests)")

    g = p.add_argument_group("evaluation")
    g.add_argument("--no-lpips", action="store_true", help="PSNR and SSIM only")
    g.add_argument("--lpips-normalize", action="store_true",
                   help="rescale images to [-1, 1] before LPIPS (the original evaluation, and Table 3, did not)")
    g.add_argument("--eval-every", type=int, default=0,
                   help="also evaluate on the test set every N epochs (0: only after the last epoch)")
    g.add_argument("--eval-batches", type=int, default=None, help="evaluate on the first N test batches only")

    g = p.add_argument_group("run")
    g.add_argument("--data-root", default="./data", help="CIFAR is downloaded here if missing")
    g.add_argument("--out-dir", default=None, help="default: runs/<dataset>_<scheme>_<real>_<adapt|noadapt>")
    g.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, cuda:1, mps or cpu")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--seed", type=int, default=0, help="seed for initialisation, data order and random PE keys")
    g.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt")
    g.add_argument("--fake-size", type=int, default=256, help="training images of --dataset fake")
    g.add_argument("--max-steps", type=int, default=None, help="stop training after this many steps")
    g.add_argument("--log-every", type=int, default=100, help="print the losses every N steps")
    args = p.parse_args(argv)
    if args.key != "paper":
        try:
            args.key = int(args.key)
        except ValueError:
            p.error("--key must be 'paper' or an integer seed")
    return args


def get_dataset(name: str, root: str, train: bool, fake_size: int = 256, offset: int = 0):
    """CIFAR-10/100 (downloaded to ``root``) or FakeData; augmentation (random crop + flip) before scrambling."""
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]) if train else T.ToTensor()
    if name == "fake":
        size = fake_size if train else max(fake_size // 2, 2)
        return torchvision.datasets.FakeData(size, (3, 32, 32), 10, transform=tf,
                                             random_offset=offset + (0 if train else 10 ** 6))
    cls = {"cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}[name]
    return cls(root, train=train, download=True, transform=tf)


def make_scrambler(args) -> siagan.Scrambler:
    if args.scheme == "randompe":  # no key: fresh random keys drawn from a generator seeded with --seed / --key
        return siagan.RandomPE(seed=args.key if isinstance(args.key, int) else args.seed)
    kwargs = {"channel_mode": args.channel_mode} if args.scheme in ("pe", "etc") else {}
    return siagan.get_scrambler(args.scheme, args.key, **kwargs)


def save_samples(path: str, rows, labels, scale: int = 3) -> None:
    """Save ``rows`` (float tensors ``(16, 3, 32, 32)`` in [0, 1]) as labelled 4x4 grids side by side (Table 2)."""
    grids = [torchvision.utils.make_grid(r[:16], nrow=4, padding=2, pad_value=1.0) for r in rows]
    tiles = [Image.fromarray((g.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for g in grids]
    tiles = [t.resize((t.width * scale, t.height * scale), Image.NEAREST) for t in tiles]
    w, h, top, gap = tiles[0].width, tiles[0].height, 24, 12
    canvas = Image.new("RGB", (len(tiles) * (w + gap) - gap, h + top), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        canvas.paste(tile, (i * (w + gap), top))
        draw.text((i * (w + gap) + 4, 5), label, fill="black")
    canvas.save(path)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = siagan.pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    real_name = args.real_dataset or args.dataset
    same = real_name == args.dataset
    adapt = "noadapt" if args.no_adaptation else "adapt"
    out_dir = args.out_dir or os.path.join("runs", f"{args.dataset}_{args.scheme}_{'same' if same else real_name}_"
                                                   f"{adapt}")
    os.makedirs(out_dir, exist_ok=True)

    scrambler = make_scrambler(args)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(get_dataset(args.dataset, args.data_root, True, args.fake_size), args.batch_size,
                              shuffle=True, **loader_kw)
    test_loader = DataLoader(get_dataset(args.dataset, args.data_root, False, args.fake_size), args.batch_size,
                             shuffle=False, **loader_kw)
    real_loader = None
    if not same:  # "Diff.": every batch of --dataset is scrambled, the real images come from another dataset
        real_set = get_dataset(real_name, args.data_root, True, args.fake_size, offset=2 * 10 ** 6)
        real_loader = DataLoader(real_set, args.batch_size, shuffle=True, **loader_kw)

    ckpt = os.path.join(out_dir, "last.pt")
    if args.resume and os.path.isfile(ckpt):
        attack = siagan.SIAGAN.load(ckpt, device=device)
        print(f"resumed from {ckpt} (epoch {attack.epoch}, step {attack.step})")
    else:
        if args.resume:
            print(f"no checkpoint at {ckpt}; starting from scratch")
        block = args.block_size or (1 if args.scheme in ("pe", "randompe") else 4)
        attack = siagan.SIAGAN(
            100 if args.dataset == "cifar100" else 10, block_size=block, adaptation=not args.no_adaptation,
            classifier=None if args.classifier == "none" else args.classifier, classifier_depth=args.classifier_depth,
            classifier_alpha=args.classifier_alpha, lambda_cls=args.lambda_cls, lambda_tv=args.lambda_tv,
            mixup_alpha=args.mixup_alpha, lr_g=args.lr_g, lr_d=args.lr_d, betas=tuple(args.betas),
            discriminator_sn=args.discriminator_sn, discriminator_width=args.ndf, device=device)

    n_params = {name: sum(p.numel() for p in m.parameters()) for name, m in
                (("generator", attack.generator), ("discriminator", attack.discriminator),
                 ("classifier", attack.classifier)) if m is not None}
    print(f"SIA-GAN vs {scrambler!r} on {args.dataset} | real images: "
          f"{'first half of every ' + args.dataset + ' batch (Same)' if same else real_name + ' (Diff.)'} | "
          f"adaptation network: {not args.no_adaptation} (block {attack.config['block_size']}) | "
          + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in n_params.items()) + f" params | device={device}")

    metrics_kw = dict(metrics=("psnr", "ssim") if args.no_lpips else ("psnr", "ssim", "lpips"),
                      max_batches=args.eval_batches, lpips_normalize=args.lpips_normalize)
    history_path = os.path.join(out_dir, "history.csv")
    if attack.epoch == 0 or not os.path.isfile(history_path):
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, HISTORY_FIELDS).writeheader()

    while attack.epoch < args.epochs and (args.max_steps is None or attack.step < args.max_steps):
        rec = attack.fit(train_loader, scrambler, epochs=1, real_loader=real_loader, log_every=args.log_every,
                         max_steps=args.max_steps)[0]
        if args.eval_every and attack.epoch % args.eval_every == 0 and attack.epoch < args.epochs:
            m = attack.evaluate(test_loader, scrambler, **metrics_kw)["reconstructed"]
            rec.update({f"test_{k}": v for k, v in m.items()})
        attack.save(ckpt)
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, HISTORY_FIELDS, restval="", extrasaction="ignore").writerow(rec)
        print(f"epoch {attack.epoch}/{args.epochs} step {attack.step} | "
              + " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k not in ("epoch", "steps", "seconds"))
              + f" | {rec['seconds']:.1f}s")

    result = attack.evaluate(test_loader, scrambler, **metrics_kw)
    summary = {"scheme": repr(scrambler), "dataset": args.dataset,
               "real_images": "same" if same else real_name, "adaptation": not args.no_adaptation,
               "epochs": attack.epoch, "steps": attack.step, "test": result, "num_params": n_params,
               "device": str(device), "args": vars(args), "siagan": siagan.__version__}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    x = next(iter(test_loader))[0][:16]
    s = scrambler(x)
    save_samples(os.path.join(out_dir, "samples.png"), [x, s, attack.reconstruct(s)],
                 ["original", f"scrambled ({args.scheme})", "SIA-GAN"])
    for name in ("scrambled", "reconstructed"):
        print(f"test {name:13s} " + " ".join(f"{k} {v:.4f}" for k, v in result[name].items()))
    print(f"results in {out_dir}")
    return summary


if __name__ == "__main__":
    main()
