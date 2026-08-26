"""Two-phase Stage-2 training on HIGH-DISTURBANCE data only.

The measured failure: the decoder holds z_hat (~95% right), so ignoring the channel is
locally optimal; it suppresses the s~ pathway, |dL/ds~| collapses 500x, and the encoder
is never told what to encode. beta=0 does NOT fix this (250 bits/frame, still ignored) —
that changes the encoder's incentive, not the decoder's necessity.

  Phase 1: z_hat withheld from the refinement ENTIRELY (input, conditioning, and residual
           base). The decoder cannot produce anything without decoding s~, so it must
           build a working channel pathway.
  Phase 2: z_hat returns, but only on high-disturbance frames, plus z_hat-dropout WITHOUT
           the residual base (the escape hatch that made the earlier dropout useless).

Wyner-Ziv is preserved: the ENCODER never sees z_hat in either phase. z_hat enters only
through the SideInfoEncoder prior (the conditional rate term), exactly as before.

Everything runs off precomputed z_t / z_hat, so no VAE and no DDIM in the loop.
"""
import os, sys, pathlib, glob, argparse, numpy as np, torch, torch.nn as nn
REPO=str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0,REPO)
from config import CONFIG
from models import SemComSystem

ap=argparse.ArgumentParser()
ap.add_argument("--pre",  type=str, required=True, help="dir of precomputed shards")
ap.add_argument("--out",  type=str, required=True)
ap.add_argument("--top",  type=float, default=0.20, help="phase-1 filter: top fraction by deviation")
ap.add_argument("--top2", type=float, default=None,
                help="phase-2 filter (default = --top). 1.0 = full distribution, so the "
                     "model learns WHEN to correct, not just how. Phase 1 creates the "
                     "channel pathway; phase 2 needs calm frames or it over-corrects them.")
ap.add_argument("--ep1",  type=int, default=3,  help="phase-1 epochs (no z_hat)")
ap.add_argument("--ep2",  type=int, default=5,  help="phase-2 epochs (z_hat returns)")
ap.add_argument("--drop2",type=float, default=0.30, help="phase-2 z_hat dropout")
ap.add_argument("--bs",   type=int, default=64)
ap.add_argument("--lr",   type=float, default=1e-4)
ap.add_argument("--beta", type=float, default=0.01)
a=ap.parse_args()
DEV="cuda:0"; os.makedirs(a.out, exist_ok=True)

ZT=[];ZH=[]
for f in sorted(glob.glob(os.path.join(a.pre,"shard*.pt"))):
    d=torch.load(f); ZT.append(d["ZT"]); ZH.append(d["ZH"])
ZT=torch.cat(ZT).float(); ZH=torch.cat(ZH).float()
ZT=ZT.reshape(-1,4,28,28); ZH=ZH.reshape(-1,4,28,28)
dev=((ZT-ZH)**2).mean(dim=(1,2,3))
N=len(dev); order=torch.argsort(dev,descending=True)
top2 = a.top if a.top2 is None else a.top2
idx1 = order[:int(a.top*N)]          # phase-1 pool
idx2 = order[:int(top2*N)]           # phase-2 pool
g=torch.Generator().manual_seed(0)
perm=torch.randperm(N,generator=g); val_set=set(perm[:max(512,int(0.1*N))].tolist())
def split(ix):
    ix=ix.tolist()
    return (torch.tensor([i for i in ix if i not in val_set]),
            torch.tensor([i for i in ix if i in val_set]))
tr1,va1=split(idx1); tr2,va2=split(idx2)
print(f"total samples {N}   all-data mean dev={dev.mean():.4f}")
print(f"  phase 1: top {100*a.top:.0f}%  -> train {len(tr1)} val {len(va1)}  mean dev={dev[idx1].mean():.4f}")
print(f"  phase 2: top {100*top2:.0f}%  -> train {len(tr2)} val {len(va2)}  mean dev={dev[idx2].mean():.4f}")
print(flush=True)
devk=dev

sysm=SemComSystem(CONFIG).to(DEV)
params=list(sysm.jscc_encoder.parameters())+list(sysm.side_info_encoder.parameters())+ \
       list(sysm.refinement_diffusion.parameters())
opt=torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
sig_n=1.0/(10.0**(CONFIG["snr_db"]/10.0))

def run(idx, train, phase, drop):
    sysm.train(train)
    order=torch.randperm(len(idx)) if train else torch.arange(len(idx))
    D=R=G=0.0; nb=0; ng=0
    for i in range(0,len(order),a.bs):
        sel=idx[order[i:i+a.bs]]
        z=ZT[sel].to(DEV); h=ZH[sel].to(DEV); B=len(z)
        with torch.set_grad_enabled(train):
            _,lv_e,s=sysm.jscc_encoder(z,sample=train)
            mu_e,_,_=sysm.jscc_encoder(z,sample=False)
            st=sysm.channel(s)
            if train: st.retain_grad()
            mu_p,lv_p=sysm.side_info_encoder(h)         # prior always sees the REAL z_hat
            # what the REFINEMENT is allowed to see
            if phase==1:
                h_ref=torch.zeros_like(h)                # nothing: must decode s~
            elif drop>0 and train:
                m=(torch.rand(B,device=DEV)<drop).float().view(B,1,1,1)
                h_ref=h*(1-m)                            # dropped => base gone too
            else:
                h_ref=h
            pred=sysm.refinement_diffusion._denoise_x0(
                    h_ref, torch.zeros(B,dtype=torch.long,device=DEV), h_ref, st)
            d=((pred-z)**2).mean()
            q=lv_e.exp()+sig_n; pv=lv_p.exp().clamp(min=1e-8)
            r=(0.5*(pv.log()-q.log()+(q+(mu_e-mu_p).pow(2))/pv-1.0)).mean()
            loss=d+a.beta*r
        if train:
            opt.zero_grad(); loss.backward()
            if st.grad is not None:
                G+=(st.grad.norm()/st.detach().norm().clamp(min=1e-12)).item(); ng+=1
            nn.utils.clip_grad_norm_(params,5.0); opt.step()
        D+=d.item(); R+=r.item(); nb+=1
    return D/nb, R/nb, (G/max(ng,1))

for ph,neps,drop in ((1,a.ep1,0.0),(2,a.ep2,a.drop2)):
    itr,iva = (tr1,va1) if ph==1 else (tr2,va2)
    print(f"===== PHASE {ph} " + ("(z_hat WITHHELD from refinement)" if ph==1
          else f"(z_hat returns, dropout={drop}, no residual base when dropped)") + " =====",flush=True)
    for e in range(1,neps+1):
        d,r,g = run(itr,True ,ph,drop)
        vd,vr,_ = run(iva,False,ph,0.0)
        print(f"  p{ph} ep{e}/{neps}  train_dist={d:.5f} rate={r:.5f}  "
              f"val_dist={vd:.5f} val_rate={vr:.5f}  |dL/ds~|={g:.3e}",flush=True)
    torch.save({"system_state":sysm.state_dict(),"phase":ph},
               os.path.join(a.out,f"phase{ph}.pt"))

# ── final evaluation ─────────────────────────────────────────────────────────
#   A = refine with the real s~_t     B = refine with s~_t ZEROED
#   C = z_hat alone, no refinement
# A vs B isolates the channel: same net, same z_hat, same frames, only the
# transmission differs. A vs C says whether Stage 2 beats the raw prediction.
#
# The DECILE breakdown is not optional. An aggregate hides regime-specific
# failure: an earlier configuration read +5.3% overall while DOUBLING the error
# on the calmest decile, which is 90% of real frames.
def evaluate(idx, label):
    A=[];Bl=[]
    with torch.no_grad():
        for i in range(0,len(idx),a.bs):
            sel=idx[i:i+a.bs]; z=ZT[sel].to(DEV); h=ZH[sel].to(DEV); B=len(z)
            _,_,s=sysm.jscc_encoder(z,sample=False); st=sysm.channel(s)
            t0=torch.zeros(B,dtype=torch.long,device=DEV)
            pa=sysm.refinement_diffusion._denoise_x0(h,t0,h,st)
            pb=sysm.refinement_diffusion._denoise_x0(h,t0,h,torch.zeros_like(st))
            A.append(((pa-z)**2).mean(dim=(1,2,3)).cpu())
            Bl.append(((pb-z)**2).mean(dim=(1,2,3)).cpu())
    A=torch.cat(A).numpy(); Bl=torch.cat(Bl).numpy(); C=dev[idx].numpy()
    rel=100*(Bl-A)/np.maximum(Bl,1e-12)
    print(f"\n{label} (n={len(A)}):")
    print(f"  A (with s~) = {A.mean():.5f}")
    print(f"  B (s~ = 0)  = {Bl.mean():.5f}")
    print(f"  C (z_hat)   = {C.mean():.5f}")
    print(f"  A vs B = {100*(Bl.mean()-A.mean())/Bl.mean():+.3f}%   <-- channel contribution")
    print(f"  A vs C = {100*(C.mean()-A.mean())/C.mean():+.2f}%   (system vs raw prediction)")
    print(f"  per-sample: helped>1% {(rel>1).sum()}/{len(rel)} "
          f"({100*(rel>1).mean():.1f}%)   best {rel.max():+.2f}%")
    return A,Bl,C

sysm.eval()
evaluate(va2, "HELD-OUT")

allidx=torch.arange(len(ZT))
A,Bl,C = evaluate(allidx, "FULL DISTRIBUTION (deployment number)")
print("\nBY DEVIATION DECILE (D1 calmest ... D10 most disturbed):")
print("   dec   n      dev      A(with)   B(s~=0)   C(zhat)    A vs B    A vs C")
q=np.percentile(C,np.arange(0,101,10))
for j in range(10):
    m=(C>=q[j])&((C<=q[j+1]) if j==9 else (C<q[j+1]))
    if not m.sum(): continue
    print(f"   D{j+1:<2d} {m.sum():6d} {C[m].mean():8.4f}  {A[m].mean():8.5f} {Bl[m].mean():8.5f} "
          f"{C[m].mean():8.5f}  {100*(Bl[m].mean()-A[m].mean())/Bl[m].mean():+7.2f}% "
          f"{100*(C[m].mean()-A[m].mean())/C[m].mean():+7.1f}%")
print("\nA vs B > 0 => the channel helps;  A vs C > 0 => Stage 2 beats the raw prediction")
