"""Evaluate the two-phase model on the FULL distribution, not just the frames it saw.

Training used only the top 20% by deviation. At deployment ~95% of frames are calm and
z_hat is near-perfect — the regime where every earlier architecture damaged calm frames.
This scores the model across the whole deviation range, in deciles.

  A = refine with s~_t     B = refine with s~_t ZEROED     C = z_hat alone
"""
import os, sys, pathlib, glob, numpy as np, torch
REPO=str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0,REPO)
from config import CONFIG
from models import SemComSystem
SC=os.path.dirname(os.path.abspath(__file__)); DEV="cuda:0"

ZT=[];ZH=[]
for f in sorted(glob.glob(os.path.join(SC,"pre","shard*.pt"))):
    d=torch.load(f); ZT.append(d["ZT"]); ZH.append(d["ZH"])
ZT=torch.cat(ZT).float().reshape(-1,4,28,28); ZH=torch.cat(ZH).float().reshape(-1,4,28,28)
dev=((ZT-ZH)**2).mean(dim=(1,2,3)); N=len(dev)

# the top-20% slice the model trained on; everything else is unseen territory
k=int(0.20*N); trained_idx=set(torch.argsort(dev,descending=True)[:k].tolist())
print(f"total samples {N}   trained on top 20% ({k})   evaluating on ALL\n")

sysm=SemComSystem(CONFIG).to(DEV)
sysm.load_state_dict(torch.load(f"{REPO}/outputs/stage2_twophase_full/phase2.pt",
                                map_location="cpu")["system_state"]); sysm.eval()
A=[];B=[]
with torch.no_grad():
    for i in range(0,N,16):
        z=ZT[i:i+16].to(DEV); h=ZH[i:i+16].to(DEV); b=len(z)
        _,_,s=sysm.jscc_encoder(z,sample=False); st=sysm.channel(s)
        t0=torch.zeros(b,dtype=torch.long,device=DEV)
        A.append(((sysm.refinement_diffusion._denoise_x0(h,t0,h,st)-z)**2).mean(dim=(1,2,3)).cpu())
        B.append(((sysm.refinement_diffusion._denoise_x0(h,t0,h,torch.zeros_like(st))-z)**2).mean(dim=(1,2,3)).cpu())
A=torch.cat(A).numpy(); B=torch.cat(B).numpy(); C=dev.numpy()

print("OVERALL, full distribution (this is the deployment number):")
print(f"  A (with s~) = {A.mean():.5f}")
print(f"  B (s~ = 0)  = {B.mean():.5f}")
print(f"  C (z_hat)   = {C.mean():.5f}")
print(f"  A vs B = {100*(B.mean()-A.mean())/B.mean():+.3f}%   (channel contribution)")
print(f"  A vs C = {100*(C.mean()-A.mean())/C.mean():+.2f}%   (system vs raw prediction)")
rel=100*(B-A)/np.maximum(B,1e-12)
print(f"  per-sample: helped>1% {(rel>1).sum()}/{N} ({100*(rel>1).mean():.1f}%)   best {rel.max():+.2f}%")

print("\nBY DEVIATION DECILE (D1 calmest ... D10 most disturbed):")
print("   dec   n     dev     A(with)   B(s~=0)   C(zhat)   A vs B   A vs C   seen?")
q=np.percentile(C,np.arange(0,101,10))
for j in range(10):
    m=(C>=q[j])&((C<=q[j+1]) if j==9 else (C<q[j+1]))
    if not m.sum(): continue
    idx=np.where(m)[0]
    seen=100*np.mean([i in trained_idx for i in idx])
    print(f"   D{j+1:<2d} {m.sum():5d} {C[m].mean():7.4f}  {A[m].mean():8.5f} {B[m].mean():8.5f} {C[m].mean():8.5f} "
          f"{100*(B[m].mean()-A[m].mean())/B[m].mean():+7.2f}% {100*(C[m].mean()-A[m].mean())/C[m].mean():+7.1f}%  {seen:5.0f}%")
print("\n'seen?' = fraction of that decile that was in the training set (top 20% only)")
