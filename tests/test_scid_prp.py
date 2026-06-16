import pytest

torch = pytest.importorskip("torch")

from scid_ad.config import SCIDConfig
from scid_ad.masking import mask_rate
from scid_ad.model import PRPDecoder, SCIDModel


def test_prp_output_shapes():
    cfg = SCIDConfig.quick()
    V = 4
    prp = PRPDecoder(cfg, n_vars=V)
    prp.eval()
    x = torch.randn((3, cfg.window_size, V))
    recon, pred = prp(x, x)
    assert recon.shape == (3, cfg.window_size, V)
    assert pred.shape == (3, cfg.window_size, V)


def test_prediction_pass_has_no_future_leakage():
    cfg = SCIDConfig.quick()
    V = 3
    prp = PRPDecoder(cfg, n_vars=V)
    prp.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn((1, cfg.window_size, V), generator=g)
    out = prp.prediction_pass(x)

    # Perturb a *future* time step; outputs at earlier positions must not change.
    t = 10
    x2 = x.clone()
    x2[:, t + 5, :] += 5.0
    out2 = prp.prediction_pass(x2)
    assert torch.allclose(out[:, : t + 1, :], out2[:, : t + 1, :], atol=1e-6)
    # the perturbed position itself (or later) is allowed to change
    assert not torch.allclose(out[:, t + 5, :], out2[:, t + 5, :], atol=1e-6)


def test_scid_model_forward_shapes_eval():
    cfg = SCIDConfig.quick()
    V = 4
    model = SCIDModel(cfg, n_vars=V)
    model.eval()
    x = torch.randn((5, cfg.window_size, V))
    out = model(x)
    assert out.recon.shape == (5, cfg.window_size, V)
    assert out.pred.shape == (5, cfg.window_size, V)
    assert out.combined.shape == (5, cfg.window_size, V)
    assert torch.allclose(out.combined, 0.5 * (out.recon + out.pred), atol=1e-6)
    assert out.z_fact.shape == (5, cfg.embed_dim)
    assert out.var_weight.shape == (5, V)


def test_scid_model_forward_with_masking_and_adjacency():
    cfg = SCIDConfig.quick()
    V = 3
    model = SCIDModel(cfg, n_vars=V)
    g = torch.Generator().manual_seed(1)
    x = torch.randn((4, cfg.window_size, V), generator=g) + 2.0
    rate = mask_rate(torch.tensor([0.2, 0.3, 0.4]), cfg.mask_eta, cfg.mask_cap)
    adj = torch.tensor(
        [[False, True, False], [True, False, False], [False, False, False]]
    )
    out = model(x, rate=rate, adj=adj, generator=g)
    assert out.recon.shape == (4, cfg.window_size, V)
    assert torch.isfinite(out.recon).all()
    assert torch.isfinite(out.combined).all()
