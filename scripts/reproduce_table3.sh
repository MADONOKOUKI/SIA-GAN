#!/usr/bin/env bash
# Table 3 of the paper: LPIPS between the original CIFAR-10 test images and the images that SIA-GAN recovers from
# scrambled images (lower = stronger attack). 20 runs = 4 table rows (adaptation network yes/no x real images from
# CIFAR-100 "Diff." / CIFAR-10 "Same") x 5 scrambling schemes (columns). Every run trains SIA-GAN for 100 epochs with
# the paper's settings and the scrambling keys of the original experiments, then writes
# runs/cifar10_<scheme>_<same|cifar100>_<adapt|noadapt>/{metrics.json,history.csv,samples.png,last.pt}.
# test.reconstructed.lpips in metrics.json is the table cell; test.scrambled.lpips is the "Scrambled image" row
# (paper: PE 0.3255, random PE 0.3231, LE 0.3035, ELE 0.2947, EtC 0.2074), and samples.png is a column of Table 2.
# CIFAR-10/100 are downloaded to ./data on first use. The runs are independent: launch them in parallel on several
# GPUs (e.g. prefix CUDA_VISIBLE_DEVICES=i). Extra arguments go to every run, e.g.
#   bash scripts/reproduce_table3.sh --data-root /path/to/data
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- SIA-GAN without adaptation network, real images: CIFAR-100 (Diff.)   paper: 0.1845 / 0.1954 / 0.1069 / 0.1589 / 0.2628
python attack.py --scheme pe       --real-dataset cifar100 --no-adaptation "$@"
python attack.py --scheme randompe --real-dataset cifar100 --no-adaptation "$@"
python attack.py --scheme le       --real-dataset cifar100 --no-adaptation "$@"
python attack.py --scheme ele      --real-dataset cifar100 --no-adaptation "$@"
python attack.py --scheme etc      --real-dataset cifar100 --no-adaptation "$@"

# ---- SIA-GAN with adaptation network, real images: CIFAR-100 (Diff.)      paper: 0.1632 / 0.1960 / 0.1565 / 0.1483 / 0.2116
python attack.py --scheme pe       --real-dataset cifar100 "$@"
python attack.py --scheme randompe --real-dataset cifar100 "$@"
python attack.py --scheme le       --real-dataset cifar100 "$@"
python attack.py --scheme ele      --real-dataset cifar100 "$@"
python attack.py --scheme etc      --real-dataset cifar100 "$@"

# ---- SIA-GAN without adaptation network, real images: CIFAR-10 (Same)     paper: 0.1964 / 0.1509 / 0.1451 / 0.1897 / 0.2067
python attack.py --scheme pe       --real-dataset cifar10 --no-adaptation "$@"
python attack.py --scheme randompe --real-dataset cifar10 --no-adaptation "$@"
python attack.py --scheme le       --real-dataset cifar10 --no-adaptation "$@"
python attack.py --scheme ele      --real-dataset cifar10 --no-adaptation "$@"
python attack.py --scheme etc      --real-dataset cifar10 --no-adaptation "$@"

# ---- SIA-GAN with adaptation network, real images: CIFAR-10 (Same)        paper: 0.1719 / 0.1559 / 0.0914 / 0.1512 / 0.2090
python attack.py --scheme pe       --real-dataset cifar10 "$@"
python attack.py --scheme randompe --real-dataset cifar10 "$@"
python attack.py --scheme le       --real-dataset cifar10 "$@"
python attack.py --scheme ele      --real-dataset cifar10 "$@"
python attack.py --scheme etc      --real-dataset cifar10 "$@"
