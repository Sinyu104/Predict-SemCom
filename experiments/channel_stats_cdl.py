import sys, numpy as np, torch
sys.path.insert(0, ".")
from baselines.traditional.sionna_channel import _CDLGains
R = 1500; out = {}
for ds in (50e-9, 100e-9):
    torch.manual_seed(0); g = _CDLGains(device="cuda", model="C", delay_spread=ds, carrier_frequency=3.5e9, max_speed=3.0)
    h = g(R * 14 * 128).reshape(R, 14, 128).cpu().numpy()
    out[ds] = h
    p = np.abs(h) ** 2
    print(f"\n=== CDL-C {ds*1e9:.0f} ns ===  E|h|^2 = {p.mean():.3f}")
    for thr in (-3, -6, -10, -15, -20): print(f"  P(|h|^2 < {thr:>3d} dB) = {np.mean(p < 10 ** (thr / 10)):.4f}", end="")
    print()
    # frequency: correlation across adjacent subcarriers (within one OFDM symbol); time: across OFDM symbols (same subcarrier)
    hf = h - 0; rho_f = [np.abs(np.mean(h[:, :, :128 - l] * np.conj(h[:, :, l:]))) / p.mean() for l in (1, 2, 4, 8, 16, 32, 64)]
    rho_t = [np.abs(np.mean(h[:, :14 - l] * np.conj(h[:, l:]))) / p.mean() for l in (1, 2, 4, 8, 13)]
    print("  freq corr |rho| at lag 1,2,4,8,16,32,64 subc (30 kHz each):", " ".join(f"{x:.2f}" for x in rho_f))
    print("  time corr |rho| at lag 1,2,4,8,13 OFDM symbols (~71 us each):", " ".join(f"{x:.2f}" for x in rho_t))
    # variation of |h|^2 within one realization (over the 128 subcarriers) vs across realizations
    within = np.mean(np.std(10 * np.log10(p.mean(1)), axis=1)); across = np.std(10 * np.log10(p.mean((1, 2))))
    print(f"  std of 10log10|h|^2 across the 128 subcarriers within a realization: {within:.2f} dB;  of the realization-mean across realizations: {across:.2f} dB")
    for snr in (4, 8, 12, 20):
        eff = snr + 10 * np.log10(p.reshape(-1) + 1e-12)
        print(f"  nominal {snr:>2d} dB -> post-ZF per-symbol SNR percentiles 1/5/25/50/75 %: " + " ".join(f"{np.percentile(eff, q):6.1f}" for q in (1, 5, 25, 50, 75)) + f" dB;  fraction of symbols below 0 dB: {np.mean(eff < 0):.3f}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); col = {50e-9: "tab:red", 100e-9: "tab:blue"}
for ds, h in out.items():
    p = np.abs(h) ** 2; x = np.sort(10 * np.log10(p.reshape(-1) + 1e-12))[::50]; ax[0].plot(x, np.linspace(0, 1, len(x)), color=col[ds], label=f"{ds*1e9:.0f} ns")
    ax[1].plot(10 * np.log10(p[3, 0]), color=col[ds], label=f"{ds*1e9:.0f} ns, realization 3")
    ax[1].plot(10 * np.log10(p[17, 0]), color=col[ds], ls="--", alpha=.6)
    lags = np.arange(0, 64); ax[2].plot(lags * 0.03, [np.abs(np.mean(h[:, :, :128 - l] * np.conj(h[:, :, l:]))) / p.mean() if l else 1 for l in lags], color=col[ds], label=f"{ds*1e9:.0f} ns")
ax[0].set_xlim(-30, 8); ax[0].set_xlabel("|h|^2 (dB)"); ax[0].set_ylabel("CDF"); ax[0].set_title("per-symbol channel power"); ax[0].grid(alpha=.3); ax[0].legend()
ax[1].set_xlabel("subcarrier"); ax[1].set_ylabel("|h|^2 (dB)"); ax[1].set_title("two example realizations (OFDM symbol 0)"); ax[1].grid(alpha=.3); ax[1].legend()
ax[2].set_xlabel("frequency separation (MHz)"); ax[2].set_ylabel("|correlation|"); ax[2].set_title("frequency correlation (grid = 3.84 MHz)"); ax[2].grid(alpha=.3); ax[2].legend()
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=110)
