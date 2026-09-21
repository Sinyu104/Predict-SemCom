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
ap.add_argument("--t2max", type=int, default=0,
                help="SDEdit level t'' for phase 2: sample t~U[1,t2max], noise z_hat, "
                     "denoise conditioned on (z_hat, s~). 0 = the current direct "
                     "single-step corrector, which has NO mechanism to distrust z_hat. "
                     "A prior sweep found 0 optimal, but that was measured when z_hat "
                     "was ~95%% correct; in the closed loop it carries 0.10-0.24 error, "
                     "so noising it costs far less.")
ap.add_argument("--eval_steps", type=int, default=10,
                help="DDIM steps at evaluation when --t2max > 0.")
ap.add_argument("--lam", type=float, default=0.0,
                help="Weight the distortion loss toward the cells where z_hat is ALREADY "
                     "wrong -- i.e. what the channel exists to fix. 0 = flat MSE (original). "
                     "Flat MSE gives the static table ~95%% of the gradient and the cube ~3%%, "
                     "so erasing a small fast object is cheaper under L2 than placing it "
                     "wrong: measured cube IoU falls 0.78 -> 0.12 across an episode while "
                     "MSE barely moves. The weight uses z_t and z_hat, both training-only "
                     "inputs, so nothing leaks to inference. Try 3-5.")
ap.add_argument("--lam_res", type=float, default=0.0,
                help="Residual penalty: lam_res * ||pred - z_hat||^2 on samples where the "
                     "refiner IS given the prior (phase-2, not dropped -- the deployment "
                     "mode). Makes 'change nothing' the default and forces the channel to "
                     "EARN every correction. Measured cause of the failure: the trained "
                     "refiner over-corrects and damages even a PERFECT prior (oracle_z VLA "
                     "action error 0.0 -> 0.77 as the residual is applied 0 -> full); a "
                     "residual-scale of ~0.4 kept most of the quality. This term is the soft "
                     "equivalent -- optimum pred = (z_t + lam_res*z_hat)/(1+lam_res), i.e. "
                     "effective scale 1/(1+lam_res). Start lam_res ~ 1.0 (scale ~0.5); "
                     "sweep 0.5/1/2. Select the checkpoint on the VLA action metric "
                     "(vla_attribution.py oracle_z), NEVER on latent MSE.")
ap.add_argument("--res_tau", type=float, default=0.0,
                help="Deviation gate for the residual penalty (only with --lam_res). The "
                     "penalty weight per frame becomes exp(-||z_t - z_hat||^2 / res_tau), so "
                     "it applies where z_hat is RELIABLE (calm, small deviation) and fades "
                     "where z_hat is wrong (disturbed) -- there the channel s~ must be free "
                     "to move pred away from z_hat. A flat penalty (res_tau=0) suppressed the "
                     "channel: measured channel contribution on disturbed frames was only "
                     "~3%%, and the reconstruction copied z_hat instead of trusting the "
                     "transmitter. Set near the calm/disturbed boundary, e.g. 0.05.")
ap.add_argument("--cube_eval", type=int, default=0,
                help="Decode this many eval samples to pixels and report cube IoU (overlap "
                     "between the cube in z_t and the cube in the reconstruction). 0 = off, "
                     "no VAE loaded, original behaviour. Latent MSE is nearly blind to the "
                     "cube: over one episode it moved 0.047 -> 0.065 while the cube went "
                     "209 px -> 39 px and IoU went 0.78 -> 0.12. 256 samples is enough to "
                     "rank two runs against each other.")
ap.add_argument("--cube_color", type=str, default="red", choices=["red", "green", "blue"],
                help="Which channel dominates on the cube. The table is blue, so "
                     "colour-dominance is a safe mask for the red/green tasks.")
ap.add_argument("--init", type=str, default=None,
                help="Warm-start from an existing checkpoint (e.g. a phase1.pt with "
                     "an already-working s~ decoder). Use with --ep1 0 to skip "
                     "phase 1 and spend the budget on phase 2 only.")
ap.add_argument("--D_jscc", type=int, default=None,
                help="Override CONFIG['D_jscc'] (channel symbols per step). The "
                     "channel can only help where it beats the predictor: phase-1 "
                     "val_dist is what s~ ALONE supports, so sweep this and compare "
                     "it against the mean deviation ||z_t - z_hat||^2.")
ap.add_argument("--spatial", action="store_true",
                help="Spatial JSCC code: conv encoder emits a (c,g,g) GRID where symbol "
                     "(k,i,j) describes image region (i,j), and the refinement's signal "
                     "tokens get a geometric 2D sin-cos embedding on that grid. Same rate "
                     "(c*g*g == D_jscc). The flat encoder moves 96%% of its symbols for a "
                     "change confined to one corner, so position has to travel through a "
                     "fully distributed code; measured live, colour survived and the cube "
                     "landed 19.6 px off -- wider than the cube. Not checkpoint-compatible "
                     "with flat runs: train from scratch, no --init.")
ap.add_argument("--grid", type=int, default=8,
                help="g for --spatial. D_jscc must be divisible by g*g (256 = 4 x 8 x 8).")
a=ap.parse_args()
if a.D_jscc is not None:
    CONFIG["D_jscc"]=a.D_jscc
CONFIG["jscc_spatial"]=a.spatial
CONFIG["jscc_grid"]=a.grid
if a.spatial:
    print(f"[cfg] spatial JSCC code: {CONFIG['D_jscc']//(a.grid*a.grid)} channels "
          f"x {a.grid}x{a.grid} grid = {CONFIG['D_jscc']} symbols (rate unchanged)")
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
if a.init:
    _d=torch.load(a.init,map_location="cpu")
    _st=_d.get("system_state",_d)
    _miss,_unexp=sysm.load_state_dict(_st,strict=False)
    print(f"[init] warm-started from {a.init}  "
          f"({len(_st)} tensors, {len(_miss)} missing, {len(_unexp)} unexpected)",flush=True)
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
                base_ok=torch.zeros(B,device=DEV)        # no prior -> no residual penalty
            elif drop>0 and train:
                m=(torch.rand(B,device=DEV)<drop).float().view(B,1,1,1)
                h_ref=h*(1-m)                            # dropped => base gone too
                base_ok=(1.0-m).view(B)
            else:
                h_ref=h
                base_ok=torch.ones(B,device=DEV)
            rd=sysm.refinement_diffusion
            if a.t2max>0 and phase==2:
                tmax=min(int(a.t2max), len(rd.alphas_cumprod)-1)
                tt=torch.randint(1,tmax+1,(B,),device=DEV)
                ab=rd.alphas_cumprod[tt].view(B,1,1,1).float()
                x_in=ab.sqrt()*h_ref+(1.0-ab).sqrt()*torch.randn_like(h_ref)
            else:
                tt=torch.zeros(B,dtype=torch.long,device=DEV); x_in=h_ref
            pred=rd._denoise_x0(x_in, tt, h_ref, st)
            if a.lam>0:
                # Weight by where z_hat is already wrong. Uses the REAL h even in phase 1
                # (where the refinement is shown zeros): the weight describes the DATA, not
                # what the decoder is allowed to see, so it does not reopen the phase-1
                # withholding. Renormalised by w.mean() so d stays on the flat-MSE scale and
                # --beta keeps its old meaning.
                e=((z-h)**2).mean(1,keepdim=True)
                w=1.0+a.lam*e/e.mean().clamp(min=1e-8)
                d=(w*(pred-z)**2).mean()/w.mean()
            else:
                d=((pred-z)**2).mean()
            q=lv_e.exp()+sig_n; pv=lv_p.exp().clamp(min=1e-8)
            r=(0.5*(pv.log()-q.log()+(q+(mu_e-mu_p).pow(2))/pv-1.0)).mean()
            loss=d+a.beta*r
            if a.lam_res>0:
                # ||pred - z_hat||^2 only where the refiner was actually given the prior
                # (base_ok): penalising the phase-1 / dropped samples would fight the very
                # withholding that builds the channel pathway. z_hat here is the REAL h.
                sq=((pred-h)**2).mean(dim=(1,2,3))       # (B,)
                # DEVIATION-AWARE gate. A flat penalty punishes moving away from z_hat
                # everywhere -- including disturbed frames where the truth IS far from
                # z_hat and the channel is what carries it, so it suppresses exactly the
                # channel corrections we want. Gate the penalty to frames where z_hat is
                # RELIABLE (small deviation): stay near the prior when it is right, move
                # freely (let s~ decide) when it is wrong. res_tau=0 keeps the flat penalty.
                if a.res_tau>0:
                    dev=((z-h)**2).mean(dim=(1,2,3))     # (B,) true deviation z_t vs z_hat
                    calm=torch.exp(-dev/a.res_tau)       # ~1 calm, ~0 disturbed
                else:
                    calm=torch.ones_like(sq)
                wgt=base_ok*calm
                res=(wgt*sq).sum()/wgt.sum().clamp(min=1.0)
                loss=loss+a.lam_res*res
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
_vae=[None]
def vae():
    """Loaded lazily so --cube_eval 0 keeps the no-VAE promise in the docstring."""
    if _vae[0] is None:
        from vae_wrapper import VAEWrapper
        _vae[0]=VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
    return _vae[0]

def cube_mask(im):
    """im: (B,H,W,3) in 0..255.  Colour-dominance; the table is blue and the gripper grey."""
    c={"red":0,"green":1,"blue":2}[a.cube_color]
    o=[i for i in range(3) if i!=c]
    return (im[...,c]>90)&(im[...,c]-im[...,o[0]]>45)&(im[...,c]-im[...,o[1]]>45)

@torch.no_grad()
def cube_stats(z_true, z_rec):
    dec=lambda z: vae().decode(z).clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.
    mg,mr=cube_mask(dec(z_true)),cube_mask(dec(z_rec))
    keep=mg.sum((1,2))>20                      # only frames where the cube is really visible
    if not keep.any(): return None
    inter=(mg&mr).sum((1,2))[keep]; union=np.maximum((mg|mr).sum((1,2))[keep],1)
    return inter/union, mg.sum((1,2))[keep], mr.sum((1,2))[keep]

def evaluate(idx, label):
    A=[];Bl=[]
    with torch.no_grad():
        for i in range(0,len(idx),a.bs):
            sel=idx[i:i+a.bs]; z=ZT[sel].to(DEV); h=ZH[sel].to(DEV); B=len(z)
            _,_,s=sysm.jscc_encoder(z,sample=False); st=sysm.channel(s)
            if a.t2max>0:
                pa=sysm.refinement_diffusion.sdedit_refine(h,st,noise_level=a.t2max,
                                                           n_steps=a.eval_steps)
                pb=sysm.refinement_diffusion.sdedit_refine(h,torch.zeros_like(st),
                                                           noise_level=a.t2max,
                                                           n_steps=a.eval_steps)
            else:
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
    if a.cube_eval>0:
        torch.cuda.empty_cache()          # the VAE lands on top of the training allocator
        # Seeded RANDOM subsample, not idx[:n]: idx is in storage order for the
        # full-distribution call, so a prefix is the opening frames of one episode
        # (calm, cube at rest) rather than a sample of the distribution.  Same seed
        # => same frames across runs, so the number is comparable run-to-run.
        gsub=torch.Generator().manual_seed(1234)
        sub=idx[torch.randperm(len(idx),generator=gsub)[:a.cube_eval]]
        U=[];G=[];R2=[]
        with torch.no_grad():
            for i in range(0,len(sub),16):
                sel=sub[i:i+16]; z=ZT[sel].to(DEV); h=ZH[sel].to(DEV); B=len(z)
                _,_,s=sysm.jscc_encoder(z,sample=False); st=sysm.channel(s)
                if a.t2max>0:
                    p=sysm.refinement_diffusion.sdedit_refine(h,st,noise_level=a.t2max,
                                                              n_steps=a.eval_steps)
                else:
                    t0=torch.zeros(B,dtype=torch.long,device=DEV)
                    p=sysm.refinement_diffusion._denoise_x0(h,t0,h,st)
                r=cube_stats(z,p)
                if r is not None: U.append(r[0]); G.append(r[1]); R2.append(r[2])
        if U:
            U=np.concatenate(U); G=np.concatenate(G); R2=np.concatenate(R2)
            print(f"  cube IoU    = {U.mean():.3f}  (px: true {G.mean():.0f} -> "
                  f"recon {R2.mean():.0f}, n={len(U)})   <-- MSE cannot see this")
        else:
            print(f"  cube IoU    = n/a (no sample had a visible {a.cube_color} cube)")
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
