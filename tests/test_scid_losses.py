import pytest

torch = pytest.importorskip("torch")

from scid_ad.config import SCIDConfig
from scid_ad.losses import gaussian_mmd2, SCIDLoss, soft_dtw, soft_dtw_normalized


def _hard_dtw(x, y):
    # Reference hard-DTW (single batch element), squared-euclidean cost.
    L, M = x.shape[0], y.shape[0]
    D = torch.cdist(x, y) ** 2
    R = torch.full((L + 1, M + 1), float("inf"))
    R[0, 0] = 0.0
    for i in range(1, L + 1):
        for j in range(1, M + 1):
            R[i, j] = D[i - 1, j - 1] + min(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1])
    return R[L, M]


def test_soft_dtw_self_near_zero():
    g = torch.Generator().manual_seed(0)
    x = torch.randn((2, 12, 3), generator=g)
    val = soft_dtw(x, x, gamma=1e-3)
    assert val.shape == (2,)
    assert torch.allclose(val, torch.zeros(2), atol=1e-2)


def test_soft_dtw_approaches_hard_dtw():
    g = torch.Generator().manual_seed(1)
    x = torch.randn((1, 8, 2), generator=g)
    y = torch.randn((1, 8, 2), generator=g)
    soft = soft_dtw(x, y, gamma=1e-3)[0]
    hard = _hard_dtw(x[0], y[0])
    assert torch.allclose(soft, hard, atol=1e-2)


def test_soft_dtw_differentiable():
    x = torch.randn((1, 6, 2), requires_grad=True)
    y = torch.randn((1, 6, 2))
    loss = soft_dtw(x, y, gamma=0.1).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_soft_dtw_normalized_scales_with_length():
    x = torch.randn((1, 10, 2))
    y = torch.randn((1, 10, 2))
    n = soft_dtw_normalized(x, y, gamma=0.1)
    assert n.dim() == 0
    assert torch.isfinite(n)


def test_mmd_zero_on_identical():
    g = torch.Generator().manual_seed(2)
    x = torch.randn((32, 4), generator=g)
    val = gaussian_mmd2(x, x, [0.5, 1.0, 2.0])
    assert abs(val.item()) < 1e-5


def test_mmd_positive_on_different():
    g = torch.Generator().manual_seed(3)
    x = torch.randn((64, 4), generator=g)
    y = torch.randn((64, 4), generator=g) + 3.0
    val = gaussian_mmd2(x, y, [0.5, 1.0, 2.0])
    assert val.item() > 0.05


def test_mmd_grad_flows():
    x = torch.randn((16, 3), requires_grad=True)
    y = torch.randn((16, 3)) + 1.0
    val = gaussian_mmd2(x, y, [1.0])
    val.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_combined_loss_warmup_excludes_prp():
    cfg = SCIDConfig.quick()
    loss = SCIDLoss(cfg)
    g = torch.Generator().manual_seed(4)
    recon = torch.randn((4, 8, 3), generator=g)
    pred = torch.randn((4, 8, 3), generator=g)
    target = torch.randn((4, 8, 3), generator=g)
    z = torch.randn((4, 5), generator=g)
    z_pos = torch.randn((4, 5), generator=g)
    z_neg = torch.randn((4, 5), generator=g)

    total_w, parts_w = loss(
        recon=recon,
        pred=pred,
        target=target,
        z=z,
        z_pos=z_pos,
        z_neg=z_neg,
        warmup=True,
    )
    # warm-up total must equal beta*dtw + gamma*mmd (no alpha*prp term)
    expect = cfg.beta * parts_w["dtw"] + cfg.gamma * parts_w["mmd"]
    assert abs(total_w.item() - expect) < 1e-5

    total_f, parts_f = loss(
        recon=recon,
        pred=pred,
        target=target,
        z=z,
        z_pos=z_pos,
        z_neg=z_neg,
        warmup=False,
    )
    expect_f = (
        cfg.alpha * parts_f["prp"]
        + cfg.beta * parts_f["dtw"]
        + cfg.gamma * parts_f["mmd"]
    )
    assert abs(total_f.item() - expect_f) < 1e-5
    # full objective includes the (positive) PRP term that warm-up omits
    assert parts_f["prp"] > 0
