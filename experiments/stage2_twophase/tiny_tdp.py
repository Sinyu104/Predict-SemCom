"""Push t'' toward ZERO, and size the disturbed group to the real 5% event rate.

Rationale: only the cube moves; ~78% of latent variance is static background that
z_hat already has right. SDEdit adds noise UNIFORMLY to all 3136 elements, so it
damages the correct background to fix a small region. The limit t''->0 skips the
damage entirely and becomes a direct conditional correction:
      pred = _denoise_x0(z_hat, t=0, z_hat, s~_t)
Also groups by the top 5% of deviation (matching pose_prob=0.05 in the collector)
instead of the top 20%, which diluted the disturbed group ~4x.
"""
import sys, pathlib, glob, h5py, numpy as np, torch
REPO=str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0,REPO)
from config import CONFIG
from vae_wrapper import VAEWrapper
from ctrl_world_wrapper import CtrlWorldWrapper
from models import SemComSystem

DEV="cuda:0"; T_H=CONFIG["num_history"]; T_P=CONFIG["num_pred"]; CLIP=T_H+T_P
vae=VAEWrapper(CONFIG["vae_model_name"]).to(DEV)
Zs,As=[],[]
for p in sorted(glob.glob(f"{REPO}/data/*/demos_disturbed.hdf5")):
    with h5py.File(p,"r") as f:
        eps=sorted(f.keys(),key=lambda k:int("".join(c for c in k if c.isdigit())))
        for e in eps[:3]:
            obs,act=f[e]["observations_cam1"],f[e]["actions"]
            for s_ in range(0,len(obs)-CLIP-1,24):
                x=torch.from_numpy(np.array(obs[s_:s_+CLIP])).float()/255.
                x=x.permute(0,3,1,2)
                with torch.no_grad():
                    Zs.append(torch.cat([vae.encode(x[j:j+2].to(DEV)).cpu() for j in range(0,len(x),2)]))
                As.append(torch.from_numpy(np.array(act[s_:s_+CLIP-1],dtype=np.float32)))
Z=torch.stack(Zs); A=torch.stack(As)
cw=CtrlWorldWrapper(action_dim=CONFIG["action_dim"],num_history=T_H,num_pred=T_P,
    svd_path=CONFIG["svd_model_name"],freeze_unet=True,
    finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn",False),dtype=torch.float16).to(DEV)
ck=torch.load(f"{REPO}/outputs/stage1_5cube_cam1_K8/stage1_best.pt",map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"],strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()
ZH=[]
for i in range(len(Z)):
    zb=Z[i:i+1].to(DEV); ab=A[i:i+1].to(DEV)
    a_full=torch.cat([ab,torch.zeros(1,1,ab.shape[-1],device=DEV)],1)
    with torch.no_grad():
        ZH.append(cw.predict_next_latent(zb[:,:T_H],a_full,n_steps=CONFIG["ctrl_world_n_steps"]).float().cpu())
ZH=torch.cat(ZH); del cw; torch.cuda.empty_cache()

zt=Z[:,T_H:].reshape(-1,*Z.shape[2:]); zh=ZH.reshape(-1,*ZH.shape[2:])
dev=((zt-zh)**2).mean(dim=(1,2,3)); N=len(dev)
order=torch.argsort(dev,descending=True)
k5=max(16,int(0.05*N))
groups={f"top 5%  (n={k5}, real disturb rate)":order[:k5],
        f"bottom 50% (calm)":order[int(0.5*N):]}
print(f"{N} samples; top5% dev={dev[order[:k5]].mean():.4f}  calm dev={dev[order[int(0.5*N):]].mean():.4f}\n",flush=True)

sys_=SemComSystem(CONFIG).to(DEV)
sys_.load_state_dict(torch.load(f"{REPO}/outputs/stage2_twophase_full/phase2.pt",
                                map_location="cpu")["system_state"]); sys_.eval()
jscc,refine,chan=sys_.jscc_encoder,sys_.refinement_diffusion,sys_.channel
ab_=refine.alphas_cumprod

def run(z,h,mode,NL):
    with torch.no_grad():
        _,_,sg=jscc(z,sample=False); st=chan(sg)
        z0=torch.zeros_like(st)
        if mode=="direct":                       # t''=0 : no noise at all
            t0=torch.zeros(len(z),dtype=torch.long,device=z.device)
            a=refine._denoise_x0(h,t0,h,st); b=refine._denoise_x0(h,t0,h,z0)
        else:
            a=refine.sdedit_refine(h,st,noise_level=NL,n_steps=10)
            b=refine.sdedit_refine(h,z0,noise_level=NL,n_steps=10)
    return ((a-z)**2).mean(dim=(1,2,3)).cpu(), ((b-z)**2).mean(dim=(1,2,3)).cpu()

print(f"{'group':<34} {'tdp':>10} {'1-abar':>8}  {'A(s~)':>9} {'B(s~=0)':>9} {'C(zhat)':>9}  {'A vs B':>8} {'A vs C':>8}")
for name,idx in groups.items():
    zg,hg=zt[idx].to(DEV),zh[idx].to(DEV); ce=dev[idx].numpy()
    for mode,NL in (("direct",0),("sdedit",1),("sdedit",2),("sdedit",5),("sdedit",10)):
        ae,be=[],[]
        torch.manual_seed(0)
        for i in range(0,len(zg),16):
            a_,b_=run(zg[i:i+16],hg[i:i+16],mode,NL)
            ae.append(a_); be.append(b_)
        ae=torch.cat(ae).numpy(); be=torch.cat(be).numpy()
        lbl="0 (direct)" if mode=="direct" else str(NL)
        na=0.0 if mode=="direct" else (1-ab_[NL].item())
        print(f"{name:<34} {lbl:>6} {na:8.4f}  {ae.mean():9.5f} {be.mean():9.5f} {ce.mean():9.5f}  "
              f"{100*(be.mean()-ae.mean())/be.mean():+7.2f}% {100*(ce.mean()-ae.mean())/ce.mean():+7.1f}%")
    print()
print("A vs C > 0 => beats raw prediction;  A vs B > 0 => channel helps")
