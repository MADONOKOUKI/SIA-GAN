"""Smoke tests of the real command line (attack.py) on synthetic data (no downloads)."""
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = ["--dataset", "fake", "--fake-size", "16", "--batch-size", "8", "--ndf", "64", "--classifier", "none",
        "--no-lpips", "--workers", "0", "--device", "cpu", "--log-every", "0"]


def run(out_dir, *args, expect=0):
    cmd = [sys.executable, os.path.join(ROOT, "attack.py"), *FAST, "--out-dir", str(out_dir), *args]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert res.returncode == expect, res.stdout + res.stderr
    return res.stdout + res.stderr


def history(out_dir):
    with open(os.path.join(out_dir, "history.csv")) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("scheme,extra", [("le", []), ("pe", []), ("randompe", []), ("ele", ["--no-adaptation"]),
                                          ("etc", ["--classifier", "shakepyramidnet", "--classifier-depth", "8",
                                                   "--classifier-alpha", "12", "--eval-every", "1"])])
def test_attack_cli(tmp_path, scheme, extra):
    out = run(tmp_path, "--scheme", scheme, "--epochs", "2", *extra)
    assert "test reconstructed" in out
    rows = history(tmp_path)
    assert [r["epoch"] for r in rows] == ["1", "2"] and float(rows[-1]["d_loss"]) > 0
    summary = json.load(open(tmp_path / "metrics.json"))
    assert summary["epochs"] == 2 and summary["real_images"] == "same"
    assert set(summary["test"]["reconstructed"]) == {"psnr", "ssim"} and summary["test"]["num_images"] == 8
    assert summary["adaptation"] == ("--no-adaptation" not in extra)
    assert (tmp_path / "last.pt").exists() and (tmp_path / "samples.png").exists()
    if scheme in ("pe", "randompe"):
        assert "(block 1)" in out  # 1x1 kernels for pixel-wise scrambling (Sec. V)
    if "--eval-every" in extra:
        assert rows[0]["test_psnr"] != "" and "cls_loss" in rows[0] and rows[0]["cls_loss"] != ""


def test_resume_and_key_seed(tmp_path):
    run(tmp_path, "--scheme", "ele", "--key", "3", "--epochs", "1")
    out = run(tmp_path, "--scheme", "ele", "--key", "3", "--epochs", "2", "--resume")
    assert "resumed" in out and "ELE(key=3" in out
    assert [r["epoch"] for r in history(tmp_path)] == ["1", "2"]
    assert "--key must be" in run(tmp_path, "--key", "abc", expect=2)


def test_epochs_zero_evaluates_the_scrambled_images(tmp_path):
    run(tmp_path, "--scheme", "etc", "--epochs", "0")
    summary = json.load(open(tmp_path / "metrics.json"))
    assert summary["steps"] == 0 and "psnr" in summary["test"]["scrambled"]


def test_diff_real_images(tmp_path, monkeypatch, capsys):
    """The 'Diff.' rows: every batch is scrambled and the real images come from another dataset."""
    sys.path.insert(0, ROOT)
    import attack

    original, made = attack.get_dataset, []

    def fake_dataset(name, root, train, fake_size=256, offset=0):  # FakeData instead of a CIFAR-100 download
        made.append((name, train))
        return original("fake", root, train, fake_size, offset)

    monkeypatch.setattr(attack, "get_dataset", fake_dataset)
    summary = attack.main([*FAST, "--real-dataset", "cifar100", "--epochs", "1", "--out-dir", str(tmp_path)])
    assert made == [("fake", True), ("fake", False), ("cifar100", True)]
    assert summary["real_images"] == "cifar100" and "real images: cifar100 (Diff.)" in capsys.readouterr().out
