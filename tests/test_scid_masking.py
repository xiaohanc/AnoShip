import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scid_ad")

from scid_ad.causal.counterfactual import counterfactual
from scid_ad.masking import (
    apply_input_mask,
    causal_lookahead_mask,
    coefficient_of_variation,
    mask_rate,
    NEG_INF,
    recon_key_mask,
)


def test_cv_matches_definition():
    g = torch.Generator().manual_seed(0)
    X = torch.rand((500, 3), generator=g) * torch.tensor([1.0, 5.0, 20.0]) + 10.0
    cv = coefficient_of_variation(X)
    expect = X.std(dim=0, unbiased=False) / X.mean(dim=0).abs()
    assert torch.allclose(cv, expect, atol=1e-5)
    assert cv.shape == (3,)


def test_cv_zero_mean_guarded():
    X = torch.zeros((50, 2))  # mean == 0 -> guarded denominator, no inf/nan
    cv = coefficient_of_variation(X)
    assert torch.isfinite(cv).all()


def test_mask_rate_formula_and_clamp():
    cv = torch.tensor([0.0, 1.0, 10.0])
    r = mask_rate(cv, eta=0.3, cap=0.5)
    assert torch.allclose(r, torch.tensor([0.0, 0.3, 0.5]))


def test_apply_input_mask_shape_and_zeroing():
    g = torch.Generator().manual_seed(1)
    x = torch.randn((4, 30, 3), generator=g) + 5.0
    rate = torch.tensor([0.0, 0.5, 1.0])
    xm, mask = apply_input_mask(x, rate, generator=g)
    assert xm.shape == x.shape and mask.shape == x.shape
    assert mask.dtype == torch.bool
    assert not mask[..., 0].any()  # variable 0 never masked
    assert mask[..., 2].all()  # variable 2 always masked
    assert (xm[mask] == 0).all()  # masked entries zeroed
    assert torch.equal(xm[~mask], x[~mask])  # rest untouched


def test_apply_input_mask_rate_approximate():
    g = torch.Generator().manual_seed(2)
    x = torch.randn((64, 64, 1), generator=g)
    rate = torch.tensor([0.3])
    _, mask = apply_input_mask(x, rate, generator=g)
    assert abs(mask.float().mean().item() - 0.3) < 0.03


def test_causal_mask_blocks_future_only():
    m = causal_lookahead_mask(5)
    assert m.shape == (5, 5)
    for i in range(5):
        for j in range(5):
            if j > i:
                assert m[i, j] == NEG_INF
            else:
                assert m[i, j] == 0.0


def test_recon_key_mask_blocks_fully_masked_steps():
    mask = torch.tensor([[[False, False], [True, True], [True, False]]])
    add = recon_key_mask(mask)
    assert add.shape == (1, 1, 3)
    assert add[0, 0, 0] == 0.0
    assert add[0, 0, 1] == NEG_INF
    assert add[0, 0, 2] == 0.0


def test_recon_key_mask_guards_all_blocked():
    mask = torch.ones((1, 4, 2), dtype=torch.bool)
    add = recon_key_mask(mask)
    assert torch.isfinite(add).all()
    assert (add == 0.0).all()


def test_counterfactual_is_per_variable_temporal_mean():
    g = torch.Generator().manual_seed(3)
    x = torch.randn((2, 10, 4), generator=g)
    cf = counterfactual(x)
    assert cf.shape == x.shape
    assert torch.allclose(cf[:, 0, :], cf[:, 5, :])  # constant along time
    assert torch.allclose(cf[:, 0, :], x.mean(dim=1))  # equals temporal mean


def test_counterfactual_rejects_bad_shape():
    with pytest.raises(ValueError):
        counterfactual(torch.zeros((3, 3)))
