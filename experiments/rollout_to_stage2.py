"""Convert rollout shards into the (ZT, ZH) layout twophase.py expects.

rollout_collect.py saves ZT over all frames but ZH only for the rollout steps, so
their leading dims differ.  twophase.py reshapes both to (-1, C, H, W) and pairs
them elementwise, so it needs the aligned pair: ZT_roll (true latent at step t)
against ZH (the z_hat the CLOSED LOOP produced at step t).

That pairing is the whole point — Stage 2 has only ever seen z_hat from
single-step prediction on clean history, where it is ~95% right, so it learned to
hold z_hat and ignore s~.  Training it against closed-loop z_hat gives it the
cases where the prediction is unreliable and the channel is what it has.

    python experiments/rollout_to_stage2.py \
        --in outputs/rollout_dist --out outputs/rollout_dist_s2
"""
import argparse, glob, os, torch

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", type=str, required=True)
ap.add_argument("--out", dest="dst", type=str, required=True)
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
tot = 0
for f in sorted(glob.glob(os.path.join(a.src, "shard*.pt"))):
    d = torch.load(f, map_location="cpu")
    zt, zh = d["ZT_roll"], d["ZH"]
    assert zt.shape == zh.shape, f"{f}: ZT_roll {tuple(zt.shape)} != ZH {tuple(zh.shape)}"
    out = os.path.join(a.dst, os.path.basename(f))
    torch.save({"ZT": zt, "ZH": zh,
                "SNR": d.get("SNR"), "CH": d.get("CH"),
                "meta": {**d.get("meta", {}), "converted_from": f}}, out)
    n = zt.shape[0] * zt.shape[1]
    tot += n
    dev = ((zt.float() - zh.float()) ** 2).mean().item()
    print(f"  {os.path.basename(f)} -> {n} samples  mean ||z_t - z_hat||^2 = {dev:.4f}")
print(f"total {tot} samples -> {a.dst}")
