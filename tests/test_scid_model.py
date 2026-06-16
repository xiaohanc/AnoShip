import pytest

torch = pytest.importorskip("torch")

from scid_ad.config import SCIDConfig
from scid_ad.model import CounterfactualAttention, DCRE, GraphAttention


def test_gat_output_shapes():
    g = torch.Generator().manual_seed(0)
    h = torch.randn((3, 5, 8), generator=g)
    gat = GraphAttention(8, 6, heads=2, concat=True)
    out, attn = gat(h)
    assert out.shape == (3, 5, 12)  # heads * out_dim
    assert attn.shape == (3, 2, 5, 5)


def test_gat_attention_rows_sum_to_one():
    g = torch.Generator().manual_seed(1)
    h = torch.randn((2, 4, 8), generator=g)
    gat = GraphAttention(8, 6, heads=3)
    gat.eval()  # softmax rows sum to 1 (dropout would rescale in train mode)
    _, attn = gat(h)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_gat_respects_adjacency_and_self_loops():
    g = torch.Generator().manual_seed(2)
    h = torch.randn((1, 3, 8), generator=g)
    gat = GraphAttention(8, 6, heads=1, dropout=0.0)
    # node 0 connects only to node 1; self loops always added
    adj = torch.tensor(
        [[False, True, False], [False, False, False], [False, False, False]]
    )
    _, attn = gat(h, adj)
    a = attn[0, 0]  # (V, V)
    # row 0 may attend to {0 (self), 1}; node 2 must get zero weight
    assert a[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert a[0, 0].item() + a[0, 1].item() == pytest.approx(1.0, abs=1e-5)
    # row 1 has no outgoing edges -> only self loop
    assert a[1, 1].item() == pytest.approx(1.0, abs=1e-5)
    assert torch.isfinite(attn).all()


def test_mca_shapes_and_gate_range():
    g = torch.Generator().manual_seed(3)
    f = torch.randn((4, 6, 10), generator=g)
    cf = torch.randn((4, 6, 10), generator=g)
    mca = CounterfactualAttention(10)
    fused, gate, var_w, attn = mca(f, cf)
    assert fused.shape == (4, 6, 10)
    assert gate.shape == (4, 6)
    assert ((gate > 0) & (gate < 1)).all()
    assert attn.shape == (4, 6, 6)
    assert torch.allclose(var_w.sum(dim=1), torch.ones(4), atol=1e-5)


def test_dcre_forward_shapes():
    cfg = SCIDConfig.quick()
    V = 4
    dcre = DCRE(cfg, n_vars=V)
    g = torch.Generator().manual_seed(4)
    x = torch.randn((5, cfg.window_size, V), generator=g)
    x_cf = x.mean(dim=1, keepdim=True).expand_as(x).contiguous()
    out = dcre(x, x_cf)
    assert out.gate.shape == (5, V)
    assert out.var_weight.shape == (5, V)
    assert out.z_fact.shape == (5, cfg.embed_dim)
    assert out.z_cf.shape == (5, cfg.embed_dim)
    assert out.fused.shape == (5, V, cfg.embed_dim)
    assert torch.allclose(out.var_weight.sum(dim=1), torch.ones(5), atol=1e-5)


def test_dcre_runs_with_causal_adjacency():
    cfg = SCIDConfig.quick()
    V = 3
    dcre = DCRE(cfg, n_vars=V)
    x = torch.randn((2, cfg.window_size, V))
    x_cf = x.mean(dim=1, keepdim=True).expand_as(x).contiguous()
    adj = torch.tensor(
        [[False, True, False], [True, False, False], [False, False, False]]
    )
    out = dcre(x, x_cf, adj)
    assert torch.isfinite(out.z_fact).all()
    assert out.gate.shape == (2, V)
