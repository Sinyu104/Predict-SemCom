"""Per-SAMPLE channel benefit, not group means.

Every ablation so far compared averages. If the channel helps strongly on a handful of
frames and not at all on the rest, a mean over 62 samples would read ~0% and look dead.
This looks at the whole distribution of (B - A) per sample:
    B = reconstruction with s~_t ZEROED,  A = with the real signal.
    benefit_i = B_i - A_i  ( > 0 means the channel helped THAT frame )
"""
import os,sys,pathlib,argparse,numpy as np,torch
REPO=str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0,REPO)
from config import CONFIG
from models import SemComSystem

DEV="cuda:0"; T_H=CONFIG["num_history"]
_ap=argparse.ArgumentParser()
_ap.add_argument("--cache",required=True,help="z/z_hat cache from precompute.py")
_ap.add_argument("--ckpt",default=f"{REPO}/outputs/stage2_twophase_full/phase2.pt")
_a=_ap.parse_args()
d=torch.load(_a.cache); Z,ZH=d["Z"],d["ZH"]
zt=Z[:,T_H:].reshape(-1,4,28,28); zh=ZH.reshape(-1,4,28,28)
dev=((zt-zh)**2).mean(dim=(1,2,3)); N=len(dev)
print(f"N={N} samples\n")

sysm=SemComSystem(CONFIG).to(DEV)
sysm.load_state_dict(torch.load(_a.ckpt, map_location="cpu")["system_state"]); sysm.eval()
jscc,refine,chan=sysm.jscc_encoder,sysm.side_info_encoder,sysm.channel
jscc,refine,chan=sysm.jscc_encoder,sysm.refinement_diffusion,sysm.channel

A=[];B=[]
torch.manual_seed(0)
for i in range(0,N,32):
    z,h=zt[i:i+32].to(DEV),zh[i:i+32].to(DEV)
    with torch.no_grad():
        _,_,sg=jscc(z,sample=False); st=chan(sg)
        a=refine.sdedit_refine(h,st,noise_level=0)
        b=refine.sdedit_refine(h,torch.zeros_like(st),noise_level=0)
    A.append(((a-z)**2).mean(dim=(1,2,3)).cpu()); B.append(((b-z)**2).mean(dim=(1,2,3)).cpu())
A=torch.cat(A).numpy(); B=torch.cat(B).numpy(); D=dev.numpy()
ben = B - A                      # >0 => channel helped this frame
rel = 100*ben/np.maximum(B,1e-12)

print("per-sample channel benefit  (B - A), positive = channel helped that frame")
for q in (0,1,5,25,50,75,95,99,100):
    print(f"   p{q:<3d} rel = {np.percentile(rel,q):+8.4f}%")
print(f"\n   mean rel = {rel.mean():+.4f}%   std = {rel.std():.4f}%")
print(f"   frames helped by >1%  : {(rel>1).sum():4d} / {N}  ({100*(rel>1).mean():.2f}%)")
print(f"   frames helped by >0.1%: {(rel>0.1).sum():4d} / {N}  ({100*(rel>0.1).mean():.2f}%)")
print(f"   frames HURT by  >1%   : {(rel<-1).sum():4d} / {N}")

k=max(32,int(0.05*N)); top=np.argsort(D)[::-1][:k]
print(f"\nrestricted to the {k} MOST disturbed frames (dev {D[top].mean():.4f}):")
for q in (5,25,50,75,95,100):
    print(f"   p{q:<3d} rel = {np.percentile(rel[top],q):+8.4f}%")
print(f"   mean rel = {rel[top].mean():+.4f}%   best single frame = {rel[top].max():+.4f}%")
print(f"   of these, helped by >1%: {(rel[top]>1).sum()} / {k}")

print(f"\ncorrelation(deviation, channel benefit) = {np.corrcoef(D,ben)[0,1]:+.4f}")
srt=np.argsort(ben)[::-1][:10]
print("\n10 frames where the channel helped MOST:")
print("     dev      B(no ch)    A(with ch)   benefit    rel")
for i in srt:
    print(f"   {D[i]:.4f}   {B[i]:.5f}   {A[i]:.5f}   {ben[i]:+.6f}  {rel[i]:+7.3f}%")
