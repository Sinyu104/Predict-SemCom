"""Precompute z_t and z_hat_t over the disturbed dataset, sharded across GPUs.

Needed for two things:
  1. deviation ||z_t - z_hat||^2 per sample, so we can train on HIGH-DISTURBANCE data only
  2. cached z_hat, which removes the 10-step DDIM from the training loop entirely
     (it was ~5.3s of every 7.8s iteration)

Run one process per GPU:  --shard i --nshards 4
"""
import os, sys, pathlib, glob, argparse, h5py, numpy as np, torch
REPO=str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0,REPO)
from config import CONFIG
from vae_wrapper import VAEWrapper
from ctrl_world_wrapper import CtrlWorldWrapper

ap=argparse.ArgumentParser()
ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=4)
ap.add_argument("--stride",type=int,default=3); ap.add_argument("--out",type=str,required=True)
a=ap.parse_args()
DEV=f"cuda:{a.shard}"; T_H=CONFIG["num_history"]; T_P=CONFIG["num_pred"]; CLIP=T_H+T_P

# enumerate every clip start across all disturbed episodes, then take this shard's slice
index=[]
for p in sorted(glob.glob(f"{REPO}/data/*/demos_disturbed.hdf5")):
    with h5py.File(p,"r") as f:
        for e in sorted(f.keys(),key=lambda k:int("".join(c for c in k if c.isdigit()))):
            n=len(f[e]["observations_cam1"])
            for s in range(0,n-CLIP-1,a.stride): index.append((p,e,s))
index=index[a.shard::a.nshards]
print(f"[shard {a.shard}] {len(index)} clips",flush=True)

vae=VAEWrapper(CONFIG["vae_model_name"]).to(DEV)
cw=CtrlWorldWrapper(action_dim=CONFIG["action_dim"],num_history=T_H,num_pred=T_P,
    svd_path=CONFIG["svd_model_name"],freeze_unet=True,
    finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn",False),dtype=torch.float16).to(DEV)
ck=torch.load(f"{REPO}/outputs/stage1_5cube_cam1_K8/stage1_best.pt",map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"],strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()

ZT=[]; ZH=[]
cur=None; fh=None
for i,(p,e,s) in enumerate(index):
    if p!=cur: 
        if fh: fh.close()
        fh=h5py.File(p,"r"); cur=p
    g=fh[e]
    x=torch.from_numpy(np.array(g["observations_cam1"][s:s+CLIP])).float()/255.
    x=x.permute(0,3,1,2)
    act=torch.from_numpy(np.array(g["actions"][s:s+CLIP-1],dtype=np.float32))
    with torch.no_grad():
        z=torch.cat([vae.encode(x[j:j+2].to(DEV)) for j in range(0,len(x),2)])
        a_full=torch.cat([act.unsqueeze(0).to(DEV),torch.zeros(1,1,act.shape[-1],device=DEV)],1)
        zh=cw.predict_next_latent(z[:T_H].unsqueeze(0),a_full,n_steps=CONFIG["ctrl_world_n_steps"]).float()
    ZT.append(z[T_H:].half().cpu()); ZH.append(zh[0].half().cpu())
    if i%200==0: print(f"[shard {a.shard}] {i}/{len(index)}",flush=True)
if fh: fh.close()
ZT=torch.stack(ZT); ZH=torch.stack(ZH)
torch.save({"ZT":ZT,"ZH":ZH}, a.out)
print(f"[shard {a.shard}] saved {a.out}  ZT={tuple(ZT.shape)}",flush=True)
