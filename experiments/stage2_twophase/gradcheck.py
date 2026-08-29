"""Is the gradient to the encoder actually dead?

dL/dtheta_encoder flows through dL/ds~. If the trained decoder has learned to ignore
s~, that derivative collapses and no gradient can tell the encoder what to encode.
Compare a TRAINED decoder against a RANDOMLY INITIALISED one on the same data.
Uses the cached z_t / z_hat, so no DDIM needed.
"""
import os,sys,pathlib,argparse,torch
REPO=str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0,REPO)
from config import CONFIG
from models import SemComSystem
SC=os.path.dirname(os.path.abspath(__file__))
d=torch.load(os.path.join(SC,"zcache.pt")); Z,ZH=d["Z"],d["ZH"]
DEV="cuda:0"; T_H=CONFIG["num_history"]
zt=Z[:,T_H:].reshape(-1,4,28,28); zh=ZH.reshape(-1,4,28,28)
dev=((zt-zh)**2).mean(dim=(1,2,3))
top=torch.argsort(dev,descending=True)[:256]          # most-disturbed frames

def grad_wrt_signal(sysm, idx, tag, BS=16):
    gsq=0.0; ssq=0.0; L=0.0; n=0
    for i in range(0,len(idx),BS):
        sub=idx[i:i+BS]
        z=zt[sub].to(DEV); h=zh[sub].to(DEV)
        with torch.no_grad():
            _,_,s=sysm.jscc_encoder(z,sample=False); st=sysm.channel(s)
        st=st.detach().requires_grad_(True)
        loss=sysm.refinement_diffusion.forward_ddpm(z,h,st,0)
        g,=torch.autograd.grad(loss,st)
        gsq+=g.pow(2).sum().item(); ssq+=st.detach().pow(2).sum().item()
        L+=loss.item(); n+=1
        del z,h,st,g,loss
        torch.cuda.empty_cache()
    gn=gsq**0.5; sn=ssq**0.5
    print(f"  {tag:<26} |dL/ds~| = {gn:.3e}   relative = {gn/max(sn,1e-30):.3e}   loss = {L/n:.5f}")
    return gn

print("gradient reaching the encoder, on the 256 most-disturbed frames:\n")
trained=SemComSystem(CONFIG).to(DEV)
trained.load_state_dict(torch.load(_a.ckpt, map_location="cpu")["system_state"])
gt=grad_wrt_signal(trained, top, "TRAINED (beta=0 ckpt)")
torch.manual_seed(0)
fresh=SemComSystem(CONFIG).to(DEV)
gf=grad_wrt_signal(fresh, top, "RANDOMLY INITIALISED")
print(f"\n  ratio trained/fresh = {gt/max(gf,1e-30):.4f}")
print("  << 1 means the trained decoder stopped listening, so no gradient can")
print("  tell the encoder what to send. That is the coordination failure.")
