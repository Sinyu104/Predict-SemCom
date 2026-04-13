"""
tests.py  —  Unit tests for all modules.

All tests run on CPU using OpenVLAStub so no GPU or model download
is required.

Run with:
    python tests.py
"""

import sys
import math
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from config import CONFIG
from models import (
    JsccEncoder, Reshaper, Predictor,
    Subtractor, Sparsifier, Reconstructor,
    RayleighChannel, Adder,
    PredictiveSemComSystem,
)
from openvla_agent import OpenVLAStub

# Use small obs size for speed
_CFG = {
    **CONFIG,
    "obs_channels": 3,
    "obs_height":   32,   # small for CPU tests
    "obs_width":    32,
    "latent_dim":   64,
    "hidden_dim":   128,
    "action_dim":   7,
    "batch_size":   4,
    "tau_mode":     "fixed",
    "tau":          0.05,
    "top_k":        None,
    "snr_db":       10.0,
}

B  = 4
C  = _CFG["obs_channels"]
H  = _CFG["obs_height"]
W  = _CFG["obs_width"]
LD = _CFG["latent_dim"]
HD = _CFG["hidden_dim"]
AD = _CFG["action_dim"]


# ── Helper ───────────────────────────────────────────────────────────── #
def build_stub():
    return OpenVLAStub(
        obs_channels=C, obs_height=H, obs_width=W, action_dim=AD
    )


# ── JsccEncoder ──────────────────────────────────────────────────────── #

def test_encoder_shapes():
    enc = JsccEncoder(C, H, W, LD)
    enc.train()
    mu, lv, z = enc(torch.rand(B, C, H, W), sample=True)
    assert mu.shape == (B, LD), f"mu shape: {mu.shape}"
    assert lv.shape == (B, LD), f"lv shape: {lv.shape}"
    assert z.shape  == (B, LD), f"z shape: {z.shape}"
    print("  [PASS] JsccEncoder shapes")

def test_encoder_inference_uses_mu():
    enc = JsccEncoder(C, H, W, LD)
    enc.eval()
    mu, lv, z = enc(torch.rand(B, C, H, W), sample=False)
    assert torch.allclose(mu, z), "At inference z should equal mu"
    print("  [PASS] JsccEncoder inference z=mu")

def test_encoder_kl_loss():
    mu  = torch.zeros(B, LD)
    lv  = torch.zeros(B, LD)          # standard normal → KL = 0
    kl  = JsccEncoder.kl_loss(mu, lv)
    assert abs(kl.item()) < 1e-5, f"KL should be ~0 for N(0,1): {kl.item()}"
    print(f"  [PASS] JsccEncoder KL loss = {kl.item():.6f}")

def test_encoder_kl_positive():
    mu  = torch.ones(B, LD)           # shifted → KL > 0
    lv  = torch.zeros(B, LD)
    kl  = JsccEncoder.kl_loss(mu, lv)
    assert kl.item() > 0, "KL should be positive for non-standard Gaussian"
    print(f"  [PASS] JsccEncoder KL > 0 for shifted Gaussian: {kl.item():.4f}")


# ── Reshaper ─────────────────────────────────────────────────────────── #

def test_reshaper_shape():
    rs = Reshaper(C, H, W, LD)
    y  = rs(torch.randn(B, LD))
    assert y.shape == (B, C, H, W), f"Reshaper output shape: {y.shape}"
    assert y.min() >= -1e-5 and y.max() <= 1.0 + 1e-5
    print("  [PASS] Reshaper shape and range")


# ── Predictor ────────────────────────────────────────────────────────── #

def test_predictor_shapes():
    pred = Predictor(LD, AD, HD)
    z_hat, h = pred(torch.randn(B, LD), torch.randn(B, AD))
    assert z_hat.shape  == (B, LD),    f"z_hat: {z_hat.shape}"
    assert h.shape      == (1, B, HD), f"h: {h.shape}"
    z_hat2, h2 = pred(torch.randn(B, LD), torch.randn(B, AD), h)
    assert z_hat2.shape == (B, LD)
    print("  [PASS] Predictor shapes")


# ── Subtractor ───────────────────────────────────────────────────────── #

def test_subtractor():
    sub = Subtractor()
    z, zp = torch.randn(B, LD), torch.randn(B, LD)
    dz = sub(z, zp)
    assert dz.shape == (B, LD)
    assert torch.allclose(dz, z - zp)
    print("  [PASS] Subtractor Δz = z − ẑ")


# ── Sparsifier ───────────────────────────────────────────────────────── #

def test_sparsifier_fixed_shape():
    sp  = Sparsifier(LD, tau_mode="fixed", tau_init=0.05)
    out = sp(torch.randn(B, LD))
    assert out["delta_z_sparse"].shape == (B, LD)
    assert out["mask"].shape           == (B, LD)
    assert 0.0 <= out["rate"].item()   <= 1.0
    print("  [PASS] Sparsifier (fixed) shapes")

def test_sparsifier_mask_correct():
    tau  = 0.3
    sp   = Sparsifier(LD, tau_mode="fixed", tau_init=tau)
    dz   = torch.randn(B, LD)
    out  = sp(dz, hard=True)
    below = dz.abs() <= tau
    above = dz.abs() >  tau
    assert (out["delta_z_sparse"][below] == 0).all(), \
        "Components ≤ tau must be zero"
    assert torch.allclose(out["delta_z_sparse"][above], dz[above]), \
        "Components > tau must be unchanged"
    print("  [PASS] Sparsifier mask correctness")

def test_sparsifier_high_tau_all_zero():
    sp  = Sparsifier(LD, tau_mode="fixed", tau_init=10.0)
    out = sp(torch.randn(B, LD), hard=True)
    assert out["delta_z_sparse"].abs().sum().item() == 0.0
    print("  [PASS] Sparsifier high tau → all zero")

def test_sparsifier_zero_tau_all_pass():
    sp  = Sparsifier(LD, tau_mode="fixed", tau_init=0.0)
    dz  = torch.randn(B, LD)
    out = sp(dz, hard=True)
    # Non-exactly-zero values should pass
    assert torch.allclose(out["delta_z_sparse"], dz)
    print("  [PASS] Sparsifier zero tau → all pass")

def test_sparsifier_top_k():
    k   = 5
    sp  = Sparsifier(LD, tau_mode="fixed", tau_init=0.0, top_k=k)
    out = sp(torch.randn(B, LD), hard=True)
    nnz = (out["delta_z_sparse"] != 0).sum(dim=-1)
    assert (nnz == k).all(), f"Expected nnz={k} per sample, got {nnz}"
    print(f"  [PASS] Sparsifier top_k={k}")

def test_sparsifier_learned_soft_differentiable():
    sp   = Sparsifier(LD, tau_mode="learned", tau_init=0.05)
    dz   = torch.randn(B, LD)
    out  = sp(dz, temperature=1.0, hard=False)
    out["rate"].backward()
    assert sp.tau.grad is not None
    assert sp.tau.grad.abs().sum().item() > 0
    print("  [PASS] Sparsifier learned soft-gate differentiable")

def test_sparsifier_learned_hard_matches_fixed():
    tau = 0.1
    sp  = Sparsifier(LD, tau_mode="learned", tau_init=tau)
    dz  = torch.randn(B, LD)
    out = sp(dz, hard=True)
    expected_mask = (dz.abs() > tau).float()
    assert torch.allclose(out["mask"], expected_mask)
    print("  [PASS] Sparsifier learned hard=True matches fixed")

def test_sparsifier_bits_formula():
    b1 = Sparsifier.bits_per_step(torch.tensor(10.0),  LD)
    b2 = Sparsifier.bits_per_step(torch.tensor(50.0),  LD)
    b3 = Sparsifier.bits_per_step(torch.tensor(128.0), LD)
    assert b1 < b2 < b3, "bits_per_step should increase with nnz"
    print(f"  [PASS] bits_per_step: nnz=10→{b1:.0f}  50→{b2:.0f}  128→{b3:.0f}")


# ── Reconstructor ────────────────────────────────────────────────────── #

def test_reconstructor_identity():
    rc  = Reconstructor()
    x   = torch.randn(B, LD)
    out = rc(x)
    assert torch.allclose(out, x), "Reconstructor must be identity"
    print("  [PASS] Reconstructor identity")


# ── RayleighChannel ──────────────────────────────────────────────────── #

def test_channel_shape():
    ch  = RayleighChannel(snr_db=10.0)
    rec = ch(torch.randn(B, LD))
    assert rec.shape == (B, LD)
    print("  [PASS] RayleighChannel shape")

def test_channel_zeros_preserved():
    ch  = RayleighChannel(snr_db=10.0)
    dz  = torch.randn(B, LD)
    dz[:, ::2] = 0.0      # suppress even dims
    rec = ch(dz)
    assert (rec[:, ::2] == 0).all(), \
        "Suppressed (zero) dims must remain zero after channel"
    print("  [PASS] RayleighChannel zeros preserved")

def test_channel_high_snr():
    ch  = RayleighChannel(snr_db=60.0)
    dz  = torch.ones(256, LD)
    rec = ch(dz)
    rel = (rec - dz).pow(2).mean() / dz.pow(2).mean()
    assert rel.item() < 0.05, f"High-SNR relative error too large: {rel:.4f}"
    print(f"  [PASS] RayleighChannel 60dB rel_err={rel:.5f}")


# ── Adder ────────────────────────────────────────────────────────────── #

def test_adder():
    adder = Adder()
    zp, r = torch.randn(B, LD), torch.randn(B, LD)
    assert torch.allclose(adder(zp, r), zp + r)
    print("  [PASS] Adder z_rec = ẑ + Δẑ")

def test_adder_suppressed_fallback():
    adder  = Adder()
    z_pred = torch.randn(B, LD)
    rec    = torch.zeros(B, LD)   # all suppressed
    assert torch.allclose(adder(z_pred, rec), z_pred), \
        "All-zero received → z_rec should equal z_pred exactly"
    print("  [PASS] Adder suppressed dims fall back to Predictor")


# ── OpenVLAStub ──────────────────────────────────────────────────────── #

def test_stub_task_loss_differentiable():
    stub = build_stub()
    y    = torch.rand(B, C, H, W, requires_grad=True)
    a    = torch.rand(B, AD)
    loss = stub.task_loss_forward(y, a)
    loss.backward()
    assert y.grad is not None, "Gradient should flow through stub task_loss"
    print("  [PASS] OpenVLAStub task_loss differentiable")

def test_stub_predict_action_shape():
    stub    = build_stub()
    actions = stub.predict_action(torch.rand(B, C, H, W))
    assert actions.shape == (B, AD)
    print(f"  [PASS] OpenVLAStub predict_action shape {actions.shape}")

def test_stub_frozen():
    stub = build_stub()
    assert all(not p.requires_grad for p in stub.parameters()), \
        "Stub parameters must be frozen"
    print("  [PASS] OpenVLAStub frozen")


# ── Full system ──────────────────────────────────────────────────────── #

def test_full_system_stage1_bypass():
    """Stage-1: bypass Predictor, no sparsification, variational encoder."""
    sys = PredictiveSemComSystem(_CFG, use_stub=True)
    sys.train()
    x   = torch.rand(B, C, H, W)
    zp  = torch.zeros(B, LD)
    ap  = torch.zeros(B, AD)
    out = sys(x, zp, ap, bypass_predictor=True, sparsify=False, sample_z=True)
    assert out["mu"].shape     == (B, LD)
    assert out["log_var"].shape== (B, LD)
    assert out["z_t"].shape    == (B, LD)
    assert out["y_rec"].shape  == (B, C, H, W)
    # bypass → z_pred = 0 → delta_z = z_t
    assert torch.allclose(out["delta_z"], out["z_t"], atol=1e-6)
    print("  [PASS] Full system Stage-1 bypass mode")

def test_full_system_stage1_loss_backward():
    """Stage-1: VIB loss can propagate gradients back through Reshaper → Encoder."""
    sys = PredictiveSemComSystem(_CFG, use_stub=True)
    sys.train()
    x   = torch.rand(B, C, H, W)
    zp  = torch.zeros(B, LD)
    ap  = torch.zeros(B, AD)
    a_gt= torch.rand(B, AD)
    out = sys(x, zp, ap, bypass_predictor=True, sparsify=False, sample_z=True)

    L_task  = sys.agent.task_loss_forward(out["y_rec"], a_gt)
    L_kl    = JsccEncoder.kl_loss(out["mu"], out["log_var"])
    loss    = L_task + L_kl
    loss.backward()

    # Check gradients exist for Encoder and Reshaper parameters
    enc_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in sys.jscc_encoder.parameters()
    )
    res_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in sys.reshaper.parameters()
    )
    assert enc_has_grad, "Encoder should receive gradients in Stage-1"
    assert res_has_grad, "Reshaper should receive gradients in Stage-1"
    print("  [PASS] Stage-1 VIB loss gradients flow through Encoder + Reshaper")

def test_full_system_stage2():
    """Stage-2: Predictor active, no sparsification."""
    sys = PredictiveSemComSystem(_CFG, use_stub=True)
    sys.eval()
    x   = torch.rand(B, C, H, W)
    zp  = torch.randn(B, LD)
    ap  = torch.randn(B, AD)
    out = sys(x, zp, ap, bypass_predictor=False, sparsify=False, sample_z=False)
    assert out["h_next"].shape == (1, B, HD)
    assert out["z_pred"].shape == (B, LD)
    print("  [PASS] Full system Stage-2 predictor active")

def test_full_system_with_sparsifier():
    """Stage-3 / inference: Sparsifier active, hard threshold."""
    cfg = {**_CFG, "tau": 0.1, "top_k": None}
    sys = PredictiveSemComSystem(cfg, use_stub=True)
    sys.eval()
    x   = torch.rand(B, C, H, W)
    out = sys(
        x, torch.randn(B, LD), torch.randn(B, AD),
        sparsify=True, sample_z=False, hard=True,
    )
    assert 0 <= out["nnz"].item() <= LD
    assert out["bits_per_step"].item() >= 0
    print(f"  [PASS] Full system with sparsifier  nnz={out['nnz'].item():.1f}/{LD}")

def test_full_system_learned_tau_soft():
    """Stage-3 training: soft gate produces differentiable rate."""
    cfg = {**_CFG, "tau_mode": "learned", "tau": 0.05}
    sys = PredictiveSemComSystem(cfg, use_stub=True)
    sys.train()
    out = sys(
        torch.rand(B, C, H, W),
        torch.randn(B, LD), torch.randn(B, AD),
        sparsify=True, sample_z=False, hard=False, temperature=1.0,
    )
    out["rate"].backward()
    assert sys.sparsifier.tau.grad is not None
    print(f"  [PASS] Full system learned tau soft-gate  rate={out['rate']:.3f}")

def test_innovation_smaller_with_good_predictor():
    sub   = Subtractor()
    z_t   = torch.randn(B, LD)
    z_pred= z_t + 0.01 * torch.randn(B, LD)
    dz    = sub(z_t, z_pred)
    assert dz.norm(dim=-1).mean() < z_t.norm(dim=-1).mean(), \
        "Good predictor should produce smaller innovation"
    print(f"  [PASS] Innovation property: "
          f"‖Δz‖={dz.norm(dim=-1).mean():.4f} < ‖z‖={z_t.norm(dim=-1).mean():.4f}")


# ========================================================================== #
#  Runner                                                                     #
# ========================================================================== #

TESTS = [
    test_encoder_shapes,
    test_encoder_inference_uses_mu,
    test_encoder_kl_loss,
    test_encoder_kl_positive,
    test_reshaper_shape,
    test_predictor_shapes,
    test_subtractor,
    test_sparsifier_fixed_shape,
    test_sparsifier_mask_correct,
    test_sparsifier_high_tau_all_zero,
    test_sparsifier_zero_tau_all_pass,
    test_sparsifier_top_k,
    test_sparsifier_learned_soft_differentiable,
    test_sparsifier_learned_hard_matches_fixed,
    test_sparsifier_bits_formula,
    test_reconstructor_identity,
    test_channel_shape,
    test_channel_zeros_preserved,
    test_channel_high_snr,
    test_adder,
    test_adder_suppressed_fallback,
    test_stub_task_loss_differentiable,
    test_stub_predict_action_shape,
    test_stub_frozen,
    test_full_system_stage1_bypass,
    test_full_system_stage1_loss_backward,
    test_full_system_stage2,
    test_full_system_with_sparsifier,
    test_full_system_learned_tau_soft,
    test_innovation_smaller_with_good_predictor,
]

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Predictive SemCom System — Unit Tests")
    print("=" * 60)
    passed = failed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print("=" * 60)
    print(f"  {passed} passed,  {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
