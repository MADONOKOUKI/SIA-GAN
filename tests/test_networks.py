"""Architecture (Tables 4-6), block-wise layers, spectral normalisation and losses."""
import pytest
import torch
import torch.nn as nn

import siagan


def test_generator_shapes_table4_table5():
    torch.manual_seed(0)
    g = siagan.Generator(block_size=4)
    img, feat = g(torch.rand(2, 3, 32, 32) * 2 - 1)
    assert img.shape == (2, 3, 32, 32) and img.abs().max() <= 1  # Tanh output
    assert feat.shape == (2, 16, 32, 32)  # pixel shuffle output of Table 4
    assert g.adaptation.convs0.weight.shape == (64, 16 * 4 * 4, 3, 4, 4)  # {16 x B x B} x H x H, H = 32 / B
    assert len(g.decoder.reps) == 3 and g.decoder.channel_opt0.weight.shape == (64, 16, 3, 3)
    g1 = siagan.Generator(block_size=1)  # PE / random PE: 1x1 kernels, 1024 sub-networks
    assert g1.adaptation.convs0.weight.shape == (1024, 16, 3, 1, 1)
    assert g1(torch.rand(2, 3, 32, 32))[0].shape == (2, 3, 32, 32)
    g0 = siagan.Generator(adaptation=False)  # the scrambled image goes straight into the decoder
    assert g0.adaptation is None and g0(torch.rand(2, 3, 32, 32))[1].shape == (2, 3, 32, 32)


def test_discriminator_shapes_table6():
    d = siagan.Discriminator()
    assert d.linear.module.in_features == 1024 * 4 * 4  # flatten: 16384
    assert [d.layer1[0].module.out_channels, d.layer2[0].module.out_channels, d.layer3[0].module.out_channels,
            d.layer4.module.out_channels] == [128, 256, 512, 1024]
    small = siagan.Discriminator(ndf=64)
    assert small(torch.rand(3, 3, 32, 32)).shape == (3,)
    assert siagan.Discriminator(ndf=64, sn="torch")(torch.rand(2, 3, 32, 32)).shape == (2,)
    with pytest.raises(ValueError):
        siagan.Discriminator(sn="weight")


def test_blockwise_conv_equals_separate_spectral_norm_convs():
    torch.manual_seed(0)
    layer = siagan.BlockwiseConv(3, 8, block_size=2, num_blocks=16)
    convs = [nn.utils.spectral_norm(nn.Conv2d(3, 8, 2, 2, bias=False)) for _ in range(16)]
    with torch.no_grad():
        for b, c in enumerate(convs):
            c.weight_orig.copy_(layer.weight[b])
            c.weight_u.copy_(layer.weight_u[b])
            c.weight_v.copy_(layer.weight_v[b])
    x = torch.randn(5, 3, 8, 8)
    for training in (True, True, False):
        layer.train(training)
        out = layer(x)
        for b, c in enumerate(convs):
            c.train(training)
            i, j = divmod(b, 4)
            ref = c(x[:, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2])[:, :, 0, 0]
            assert torch.allclose(out[:, :, i, j], ref, atol=1e-5)
    with pytest.raises(ValueError):
        layer(torch.randn(1, 3, 6, 6))


def test_blockwise_batchnorm_normalises_every_block_separately():
    bn = siagan.BlockwiseBatchNorm(4, 9).train()
    x = torch.randn(16, 4, 3, 3) * torch.arange(1, 10.).view(1, 1, 3, 3) + 5
    y = bn(x)
    assert torch.allclose(y.mean(0), torch.zeros(4, 3, 3), atol=1e-5)
    assert torch.allclose(y.var(0, unbiased=False), torch.ones(4, 3, 3), atol=1e-3)


def test_inplace_spectral_norm_rescales_the_stored_weight():
    torch.manual_seed(0)
    lin = nn.Linear(20, 10)
    with torch.no_grad():
        lin.weight.mul_(10)
    sn = siagan.InplaceSpectralNorm(lin)
    u0 = sn.u.clone()
    for _ in range(20):
        sn(torch.rand(1, 20))
    sigma = torch.linalg.matrix_norm(lin.weight.detach(), ord=2).item()
    assert 0.9 < sigma < 1.5 and torch.equal(sn.u, u0)  # the stored weight is ~normalised, u is never updated


def test_hinge_losses():
    real, fake = torch.tensor([2.0, 0.5]), torch.tensor([-2.0, 0.0])
    assert siagan.discriminator_hinge_loss(real, fake).item() == pytest.approx((0 + 0.5) / 2 + (0 + 1) / 2)
    assert siagan.generator_hinge_loss(fake).item() == pytest.approx(1.0)


def test_shake_pyramidnet():
    torch.manual_seed(0)
    net = siagan.ShakePyramidNet(depth=8, alpha=12, num_classes=7)
    x = torch.rand(4, 3, 32, 32)
    out = net(x)
    assert out.shape == (4, 7)
    out.sum().backward()
    net.eval()
    assert torch.equal(net(x), net(x))  # deterministic at test time
    assert sum(p.numel() for p in siagan.ShakePyramidNet().parameters()) > 28e6  # depth 110, alpha 270
    with pytest.raises(ValueError):
        siagan.ShakePyramidNet(depth=10)
