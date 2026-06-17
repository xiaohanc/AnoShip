import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scid_ad")

from scid_ad.config import SCIDConfig
from scid_ad.masking import coefficient_of_variation, mask_rate
from scid_ad.model import SCIDModel
from scid_ad.training import SCIDTrainer


def _make_windows(cfg, n=24, V=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(cfg.window_size)
    base = np.sin(2 * np.pi * t / 11.0)[:, None]
    data = []
    for w in range(n):
        x = base + 0.1 * rng.normal(size=(cfg.window_size, V)) + rng.normal(size=V)
        data.append(x)
    return torch.tensor(np.stack(data), dtype=torch.float32)


def test_trainer_loss_decreases():
    cfg = SCIDConfig.quick()
    cfg.epochs = 10
    cfg.warmup_epochs = 2
    cfg.patience = 0  # disable early stop so we see the full trajectory
    V = 3
    windows = _make_windows(cfg, n=24, V=V)
    rate = mask_rate(
        coefficient_of_variation(windows.reshape(-1, V)), cfg.mask_eta, cfg.mask_cap
    )
    model = SCIDModel(cfg, n_vars=V)
    trainer = SCIDTrainer(model, cfg)
    history = trainer.fit(windows, rate)

    full = [h for h in history if h["warmup"] == 0.0]
    assert len(full) >= 2
    # the full-phase objective should trend downward
    assert full[-1]["total"] < full[0]["total"]
    # reconstruction error in particular should improve
    assert full[-1]["prp"] < full[0]["prp"]


def test_trainer_warmup_excludes_prp_term():
    cfg = SCIDConfig.quick()
    cfg.epochs = 6
    cfg.warmup_epochs = 3
    cfg.patience = 0
    V = 3
    windows = _make_windows(cfg, n=16, V=V, seed=1)
    rate = mask_rate(
        coefficient_of_variation(windows.reshape(-1, V)), cfg.mask_eta, cfg.mask_cap
    )
    trainer = SCIDTrainer(SCIDModel(cfg, n_vars=V), cfg)
    history = trainer.fit(windows, rate)

    warm = [h for h in history if h["warmup"] == 1.0]
    full = [h for h in history if h["warmup"] == 0.0]
    assert len(warm) == 3 and len(full) == 3
    # During warm-up the recorded total must equal beta*dtw + gamma*mmd (no PRP).
    for h in warm:
        expect = cfg.beta * h["dtw"] + cfg.gamma * h["mmd"]
        assert abs(h["total"] - expect) < 1e-4
    # The full phase's total includes the alpha*PRP term.
    for h in full:
        expect = cfg.alpha * h["prp"] + cfg.beta * h["dtw"] + cfg.gamma * h["mmd"]
        assert abs(h["total"] - expect) < 1e-4
