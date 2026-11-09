"""Latent cache loader shared by train.py / eval.py (same cache the WZ codec trains on)."""
import glob, os, torch


def load_cache(dirs):
    ZT, ZP, DIST = [], [], []
    for f in sorted(f for d in dirs for f in glob.glob(os.path.join(d, "shard*.pt"))):
        d = torch.load(f, map_location="cpu")
        assert "ZPREV" in d, "need the ZPREV cache (outputs/rollout_disturbed_clean_s2_moe)"
        ZT.append(d["ZT"].float()); ZP.append(d["ZPREV"].float())
        DIST.append(d["DIST"].reshape(-1).float() if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
    ZT, ZP, DIST = torch.cat(ZT), torch.cat(ZP), torch.cat(DIST)
    # 2-view caches (N,4,28,56) are treated as one wider grid: DVST is fully conv, no change needed
    return ZT, ZP, DIST


def split(N, seed=0):
    """Same 90/10 split as experiments/stage2_twophase/train_masked_alloc_wz.py (seed 0 perm)."""
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    nv = int(N * 0.1)
    return perm[nv:], perm[:nv]            # train, val


def load_ordered(dirs):
    """Restore the global (episode, t) order of the chunk-30 caches.

    fast_chunk30_precompute.py shards by interleaving (shard s holds index[s::nshards], t_stride=4 steps apart), so
    global row g = r*nshards + s.  Returns ZT, ZP, DIST in that order, `orig` (each row's index in load_cache()'s
    shard-major concatenation, so split() indices can be mapped) and `ds` (dataset id; links never cross datasets).
    Row g+8 is the frame ~32 steps later (ZP[g+8] = z[t+31] vs ZT[g] = z[t+29]).
    """
    ZT, ZP, DIST, ORIG, DS, base = [], [], [], [], [], 0
    for di, d in enumerate(dirs):
        sh = [torch.load(f, map_location="cpu") for f in sorted(glob.glob(os.path.join(d, "shard*.pt")))]
        n = len(sh); N = sum(len(s["ZT"]) for s in sh)
        off = [0]
        for s in sh: off.append(off[-1] + len(s["ZT"]))
        zt = torch.empty(N, *sh[0]["ZT"].shape[1:]); zp = torch.empty_like(zt); dm = torch.zeros(N); og = torch.empty(N, dtype=torch.long)
        for g in range(N):
            s, r = g % n, g // n
            zt[g] = sh[s]["ZT"][r].float(); zp[g] = sh[s]["ZPREV"][r].float()
            dm[g] = sh[s]["DIST"].reshape(-1)[r].float() if "DIST" in sh[s] else 0.0
            og[g] = base + off[s] + r
        ZT.append(zt); ZP.append(zp); DIST.append(dm); ORIG.append(og); DS.append(torch.full((N,), di)); base += N
    return torch.cat(ZT), torch.cat(ZP), torch.cat(DIST), torch.cat(ORIG), torch.cat(DS)


def build_chains(ZT, ZP, DS, L, gap=8, thr=0.15):
    """Start rows g such that rows g, g+gap, ..., g+gap*(L-1) form a same-episode sequence.
    Frames of a chain: I = ZP[g] (intra), P_k = ZT[g + gap*(k-1)], k=1..L (consecutive gaps ~30-32 steps)."""
    N = len(ZT)
    link = torch.zeros(N, dtype=torch.bool)
    d = ((ZT[:-gap] - ZP[gap:]) ** 2).mean((1, 2, 3))
    link[:-gap] = (d < thr) & (DS[:-gap] == DS[gap:])
    ok = torch.ones(N, dtype=torch.bool)
    for k in range(L - 1):
        sh = torch.zeros(N, dtype=torch.bool); sh[:N - gap * k] = link[gap * k:]
        ok &= sh
    ok[N - gap * (L - 1):] = False
    return ok.nonzero().squeeze(1)


def chain_batch(ZT, ZP, starts, L, gap=8):
    """(B, L+1, C, H, W): [I, P1..PL]."""
    fr = [ZP[starts]] + [ZT[starts + gap * k] for k in range(L)]
    return torch.stack(fr, 1)
