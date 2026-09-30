# Original research code (archived)

This directory contains the original research code used for the experiments of the paper
*SIA-GAN: Scrambling Inversion Attack Using Generative Adversarial Network* (IEEE Access, 2021).
It is kept **for reference and reproducibility only and is not maintained**. It targets the PyTorch of 2020 with
CUDA (`.cuda()`, `torch.cuda.FloatTensor`, `nn.DataParallel`), tensorboardX and lpips; the `*.sh` files are the job
scripts that ran it on the ABCI cluster (each holds the configuration of its last run). The scripts read the LE keys
`key4/<i>_.pkl` relative to the working directory, so run them from this directory.

Use the package and the command-line script at the repository root instead. The tests in
`tests/test_parity_original.py` import this code and check that the new implementation gives the same outputs: LE
scrambling with the key files, the generator (including the spectral-normalisation power iteration and the BatchNorm
statistics of its 64 block-wise sub-networks), the discriminator with its in-place spectral normalisation, the
ShakeDrop PyramidNet classifier, the random PE of `random_mask` and the losses.

## Old scripts -> new command

| archived script | what it runs | new command |
|---|---|---|
| `gan_attack_alpha_10.py` | SIA-GAN on CIFAR-10: each batch split into real images and generator inputs; `--block_scramble 1` = LE (key 0), `--random_pixel_inversion 1 --random_pixel_inversion_test 1` = random PE; LPIPS of the 10,000 test images after every epoch; always adds `1e-3 * TV(feature)`; optional mixup | `python attack.py --scheme le` / `--scheme randompe` (add `--lambda-tv 1e-3` for its TV term, `--eval-every 1` for LPIPS every epoch) |
| `gan_attack.py` | the same attack on LE with mixup of the scrambled inputs (`--mixup True`, alpha = 1) and Adam 2e-4 | `python attack.py --scheme le --mixup-alpha 1` |
| `gan_attack_alpha_10_mask.py` | `gan_attack.py` with learning rates 1e-4 / 4e-4, twice the loader batch and a configurable mixup alpha (its `random_mask` call is commented out) | `python attack.py --scheme le` |
| `gan_attack_alpha_enc_dec.py`, `gan_attack_alpha_enc_dec_instahide.py`, `gan_attack_sameimg.py` | exploratory variants (LE encryption or decryption chosen at random, then `random_mask`; `random_mask` followed by a random permutation of the first three pixel columns; mixup of two scrambled copies of the same images); they call the generator with a single return value and do not run with the final `generator.py` | not ported |
| `tanaka_cifar10_LE*.py` | classification of LE-scrambled CIFAR-10 with LE-AdaptNet + ShakeDrop PyramidNet (Tanaka 2018) and variants (mixup, noise, multitask key prediction, InstaHide-style mixing); side experiments that the paper does not report | not ported (classification of scrambled images: [Block-wise-Scrambled-Image-Recognition](https://github.com/MADONOKOUKI/Block-wise-Scrambled-Image-Recognition)) |

`scripts/reproduce_table3.sh` lists the 20 runs of Table 3 of the paper.

The released code implements LE and random PE only; PE, ELE and EtC (the other columns of Tables 2-3) are not in
this directory. The package takes them from the authors' other code, as the scramblekit library does: ELE and EtC
(with the key of `random.seed(30)`) from the block-wise scrambling code of the AAAI-20 workshop paper, and PE (with
the first CIFAR-10 key hard-coded there as `key="paper"`) from the ScrambleMix code. The PE keys of the SIA-GAN
experiments are not known.

## Old modules -> new code

| archived file | new code |
|---|---|
| `generator.py` (`Generator`) | `siagan.Generator` = `siagan.AdaptationNetwork` (batched `BlockwiseConv` + `BlockwiseBatchNorm`) + `siagan.FeatureDecoder` |
| `dcgan_spec_acgan.py` (`SpectralNorm`, `Self_Attn`, `Discriminator`) | `siagan.InplaceSpectralNorm`, `siagan.SelfAttention`, `siagan.Discriminator` (the unused AC-GAN heads are left out) |
| `no_adaptation_network.py`, `shakedrop.py` (after owruby/shake-drop_pytorch) | `siagan.ShakePyramidNet`, `siagan.ShakeDrop` (the classifier of the generator loss) |
| `learnable_encryption.py` (`BlockScramble`, from mastnk/ICCE-TW2018) + `Blockwise_scramble_LE.py` | `siagan.LE` (and the per-block keys of `siagan.ELE`) |
| `key4/<i>_.pkl` (64 LE keys) | `siagan.paper_keys()` (`siagan/resources/paper_keys.npz`), the default keys of `LE()` / `ELE()` |
| `random_mask` in `gan_attack_alpha_10.py` | `siagan.RandomPE` |
| `total_variation_norm` in `gan_attack_alpha_10.py` (`util_norm.py` for the classification scripts) | `siagan.feature_total_variation` (`SIAGAN(lambda_tv=...)`) |
| `mixup.py` / `mixup_data` (after facebookresearch/mixup-cifar10) | `SIAGAN(mixup_alpha=...)` |
| hinge losses in the training loops | `siagan.discriminator_hinge_loss`, `siagan.generator_hinge_loss` |
| `tanaka_adaptation_network.py`, `no_adaptation_network_multitask.py` | not ported (classification experiments) |
| `jigsaw_loop_4x4_size_mse_solve_shuffle_fullmlp.py` | not ported (unused; imports `linear` and `my_sinkhorn_ops`, which are not in the repository) |
| `scheduler.py` (`CyclicLR`, from thomasjpfan/pytorch after bckenstler/CLR) | not ported (imported only by the classification scripts) |
