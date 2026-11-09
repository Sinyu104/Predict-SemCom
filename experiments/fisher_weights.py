"""Task (Fisher) weights for the Stage-2 codec loss: per latent cell, how sensitive PI0Fast's action is to that cell.

For each training frame z (true latent, 4x28x56 = two cameras):
  1. x = VAE_decode(z) per camera (what the codec's target looks like to the policy).
  2. Generate the policy's own action tokens a* on x (greedy, the deployed decoding).
  3. Teacher-forced forward on [prefix(x), bos, a*]; L = log p(a* | x)  (FAST phase uses the deployed logit mask).
  4. g_x = dL/dx (backward through PI0Fast only), then g_z = J_dec^T g_x (vector-Jacobian product through the VAE decoder).
  5. F(cell) = sum_c g_z[c, cell]^2   -- the diagonal empirical Fisher of the policy w.r.t. the latent, one sample.
Locally KL(pi(.|x) || pi(.|x~)) ~ 1/2 d^T F d, so sum_cell F * (rec - z)^2 is the second-order task loss.
Writes fisher_shard<i>.pt = {"F": (N,1,28,56) float16, "idx": frame indices} next to each shard<i>.pt.
pi0fast env (needs diffusers for the VAE? no: the VAE is loaded via diffusers in vae_wrapper -> run in the pi0fast env only if diffusers is there;
otherwise --vae_device on a predictor-env helper).  One free GPU:
    CUDA_VISIBLE_DEVICES=1 .../envs/pi0fast/bin/python experiments/fisher_weights.py --pre outputs/rollout_pose_unf2 --limit 8
"""
import argparse, glob, math, os, pathlib, sys, time
import numpy as np, torch
REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "server"))
ap = argparse.ArgumentParser()
ap.add_argument("--pre", default="outputs/rollout_pose_unf2"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shards", type=int, nargs="*", default=None); ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
ap.add_argument("--cache", default="outputs/stage1_chunk30_2cam_ft/rollout_latents.pt")
a = ap.parse_args()
COLORS = ["blue", "orange", "purple", "red", "yellow"]           # sorted data/pick_*_cube_to_tray/policy_rollouts.hdf5 -> cache F index
FILE = list(torch.load(os.path.join(REPO, a.cache), map_location="cpu")["F"])
from vla_server import Pi0FastServer   # noqa: E402
from vae_wrapper import VAEWrapper     # noqa: E402
dev = "cuda:0"
srv = Pi0FastServer(model_dir="lerobot/pi0fast-base", lora_path=os.path.join(REPO, "outputs/pi0fast_5colors_lora_transition/checkpoint_epoch7/lora_weights.pt"),
                    full_weights=os.path.join(REPO, "outputs/pi0fast_5colors_scratch_dagger/checkpoint_epoch5/full_weights.pt"),
                    chunk_size=30, state_dim=6, action_dim=7, device=dev)
vae = VAEWrapper("stabilityai/sd-vae-ft-mse").to(dev).eval(); vae.requires_grad_(False); vae._load(); vae._vae.to(dev).enable_gradient_checkpointing()
def dec_grad(z):                                                     # VAEWrapper.decode is @no_grad; same math, differentiable
    return (vae._vae.decode(z / vae._scale).sample.clamp(-1, 1) + 1.0) / 2.0
pol, model, tok = srv.policy, srv.policy.model, srv._pali_tok
lm_head = model.paligemma_with_expert.paligemma.lm_head
model.gradient_checkpointing_enable()                                 # fp32 weights fill ~12 GB of the T4; recompute layers in backward (needs train mode, dropout is 0)
vocab = tok.vocab_size; fast_skip = pol.config.fast_skip_tokens; fast_start, fast_end = vocab - fast_skip - 2048, vocab - fast_skip
pipe_id = tok.convert_tokens_to_ids("|"); bos_id = tok.bos_token_id
n_text = len(tok.encode("Action: ", add_special_tokens=False))        # "Action", ":", "▁"
fmask = torch.full((vocab,), float("-inf"), device=dev); fmask[fast_start:fast_end] = 0.0; fmask[pipe_id] = 0.0

@torch.no_grad()
def generate_tokens(cam1, cam2, instr):
    """Greedy tokens after bos with a KV cache, exactly as Pi0FastServer.predict: n_text free text tokens, then FAST(+'|') tokens."""
    lang_ids, lang_mask = srv._build_language_tokens(instr, None)
    batch = {srv.cam1_key: cam1, srv.cam2_key: cam2, "observation.language.tokens": lang_ids, "observation.language.attention_mask": lang_mask.bool()}
    images, img_masks = pol._preprocess_images(batch)
    bos = torch.tensor([[bos_id]], device=dev)
    pe, pm, am, _, _ = model.embed_prefix_fast(images, img_masks, lang_ids, lang_mask.bool(), fast_action_tokens=bos, fast_action_masks=torch.ones_like(bos, dtype=torch.bool))
    pos = torch.cumsum(pm, 1) - 1; att = model._prepare_attention_masks_4d(am, dtype=pe.dtype)
    (o, _), kv = model.paligemma_with_expert.forward(attention_mask=att, position_ids=pos, past_key_values=None, inputs_embeds=[pe, None], use_cache=True, adarms_cond=[None, None])
    out = [int(lm_head(o[:, -1])[0].argmax())]; cur = pm
    while True:
        e = model.paligemma_with_expert.embed_language_tokens(torch.tensor([[out[-1]]], device=dev)); e = e * math.sqrt(e.shape[-1])
        cur = torch.cat([cur, torch.ones((1, 1), dtype=torch.bool, device=dev)], 1); p_ = (cur.sum(1, keepdim=True) - 1).long()
        (o, _), kv = model.paligemma_with_expert.forward(attention_mask=model._prepare_attention_masks_4d(cur.unsqueeze(1), dtype=e.dtype), position_ids=p_,
                                                         past_key_values=kv, inputs_embeds=[e, None], use_cache=True, adarms_cond=[None, None])
        lg = lm_head(o[:, -1])[0]
        if len(out) >= n_text: lg = lg + fmask
        t = int(lg.argmax()); out.append(t)
        if (len(out) > n_text and t == pipe_id) or len(out) >= n_text + pol.config.max_decoding_steps: break
    return out

def logp(cam1, cam2, instr, toks):
    lang_ids, lang_mask = srv._build_language_tokens(instr, None)
    batch = {srv.cam1_key: cam1, srv.cam2_key: cam2, "observation.language.tokens": lang_ids, "observation.language.attention_mask": lang_mask.bool()}
    images, img_masks = pol._preprocess_images(batch)
    ft = torch.tensor([[bos_id] + toks], device=dev)
    pe, pm, am, _, _ = model.embed_prefix_fast(images, img_masks, lang_ids, lang_mask.bool(), fast_action_tokens=ft, fast_action_masks=torch.ones_like(ft, dtype=torch.bool))
    pos = torch.cumsum(pm, 1) - 1; att = model._prepare_attention_masks_4d(am, dtype=pe.dtype)
    (o, _), _ = model.paligemma_with_expert.forward(attention_mask=att, position_ids=pos, past_key_values=None, inputs_embeds=[pe, None], use_cache=False, adarms_cond=[None, None])
    T = len(toks); lg = lm_head(o[0, -T - 1:-1])                        # logits that predict each of toks
    lg = torch.cat([lg[:n_text], lg[n_text:] + fmask], 0)
    return torch.log_softmax(lg.float(), -1)[torch.arange(T, device=dev), torch.tensor(toks, device=dev)].sum()

def fisher(z, instr):
    """z: (4,28,56) true latent -> (28,56) diagonal Fisher (one greedy sample)."""
    with torch.no_grad():
        x = [vae.decode(z[None, ..., off:off + 28]).clamp(0, 1) for off in (0, 28)]     # (1,3,224,224) each
    toks = generate_tokens(x[0], x[1], instr)
    xg = [xi.clone().requires_grad_(True) for xi in x]
    model.train(); L = logp(xg[0], xg[1], instr, toks); L.backward(); model.eval()
    gx = [xi.grad.detach() for xi in xg]; del L, xg; torch.cuda.empty_cache()
    gz = []
    for off, g in zip((0, 28), gx):
        zz = z[None, ..., off:off + 28].clone().requires_grad_(True)
        with torch.enable_grad():
            (dec_grad(zz) * g).sum().backward()
        gz.append(zz.grad[0])
    G = torch.cat(gz, -1)                                                 # (4,28,56)
    return (G ** 2).sum(0), len(toks)

for f in sorted(glob.glob(os.path.join(REPO, a.pre, "shard*.pt"))):
    si = int(os.path.basename(f)[5:-3])
    if a.shards is not None and si not in a.shards: continue
    d = torch.load(f); Z, EP = d["ZT"].float(), d["EP"]
    idx = list(range(0, len(Z), a.stride))[a.part::a.nparts]
    if a.limit: idx = idx[:a.limit]
    out = os.path.join(os.path.dirname(f), f"fisher_shard{si}" + (f"_p{a.part}" if a.nparts > 1 else "") + (f"_lim{a.limit}" if a.limit else "") + ".pt")
    Fs = torch.zeros(len(idx), 1, 28, 56, dtype=torch.float16); t0 = time.time()
    for j, i in enumerate(idx):
        instr = f"pick up the {COLORS[int(FILE[int(EP[i])])]} cube and place it on the tray"
        F, nt = fisher(Z[i].to(dev), instr); Fs[j, 0] = F.half().cpu()
        if j < 3 or j % 50 == 0:
            print(f"[fisher] shard{si} {j + 1}/{len(idx)} frame {i} tokens {nt} sumF {F.sum().item():.3e} max {F.max().item():.3e} "
                  f"top5%-cells share {torch.topk(F.flatten(), 78).values.sum().item() / F.sum().item():.2f}  {(time.time() - t0) / (j + 1):.1f}s/frame  "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
    torch.save({"F": Fs, "idx": torch.tensor(idx)}, out); print("saved", out, flush=True)
