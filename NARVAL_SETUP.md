# Running this repo on Narval (Digital Research Alliance / Compute Canada)

This is a **separate deployment target** from the `worldmodel` conda env described in
`README.md` (that setup is for the lab's own 4×T4 server). This file documents what it
actually takes to fine-tune PI0Fast here, and the mistakes already made once — don't
repeat them.

## 1. Login node vs compute node — this is not optional

Narval's login node (`narval1`) is shared infrastructure. Alliance staff actively monitor
it and will email/suspend accounts for sustained high CPU/mem processes. We already
triggered one such warning in this project (a background HF download + an attempted full
model load on the login node, ~600% CPU for several minutes).

**Rule: the login node is only for file management, editing, and package installs.**
Anything that constructs a model, allocates real tensors, or runs a training/inference
loop — even briefly, even on CPU — goes through `sbatch`/`salloc`. No exceptions, no
"just testing real quick."

Concretely:
- ✅ `pip install`, `git clone`, small config/tokenizer downloads (KB–low MB, seconds)
- ✅ `huggingface_hub.snapshot_download(...)` for model weights (I/O-bound, not compute)
- ❌ `PI0FastPolicy.from_pretrained(...)` or any full model construction — this
  materializes a 3.45B-param model in memory even just to "check it loads." Don't.
- ❌ Anything you'd background with `&` or `run_in_background` hoping it finishes
  quietly — if it's heavy enough to background, it's heavy enough to need a job.

Because compute nodes have **no internet access**, this creates a real constraint: any
step that needs both (a) network access and (b) real compute must be split — do the
network part on the login node in isolation (see §3), and only the offline compute part
in the job.

## 2. Environment: dedicated venv, not the OpenVLA one

`~/venv/predict_semcom` is the existing venv used by `slurm_finetune_dragger.sh`
(OpenVLA). It does **not** have `lerobot` installed, and shouldn't be modified for this —
it may be in active use for OpenVLA jobs.

PI0Fast/lerobot work uses a separate venv: **`~/venv/pi0fast_lerobot`** (Python 3.12.4).

### Module load order matters, and doesn't persist across shells

Two of lerobot's dependencies (`opencv-python-headless`, `pyarrow`) are shipped in
Compute Canada's wheelhouse as intentionally-broken "dummy" packages (version `9999`)
that print a helpful error telling you to load a module instead of actually installing
from PyPI/wheelhouse. **The real package only becomes importable if you `module load`
it before activating the venv** — the module's own site-packages gets exposed to the
venv. Every single shell/job that touches this venv must do, in this exact order:

```bash
module load gcc opencv/4.12.0 arrow/25.0.0 python/3.12.4
source ~/venv/pi0fast_lerobot/bin/activate
```

Get the opencv version wrong (e.g. the default `4.14.0`) and it silently fails to
satisfy lerobot's `<4.13.0` pin, falling through to a PyPI source build of opencv that
then fails on an unrelated numpy pin — don't chase that error, just fix the module
version.

**This state does not persist between separate command invocations/tool calls** — an
agent (or a human in a new terminal) must redo the `module load` + `source activate`
every time, in the same shell. Forgetting this was the cause of several early failures
in this project (e.g. `h5py` appearing "not installed" when it was, because the venv
was never actually activated with modules loaded in that particular shell).

### `rerun-sdk` is unresolvable here

It's not in Compute Canada's wheelhouse or reachable from PyPI on this cluster, and
`pi0fast_server.py` never imports it (it's a lerobot dependency for their own
visualization CLI, unused by our training script). Satisfy it with an empty local stub
package rather than fighting the resolver:

```bash
mkdir -p /tmp/fake_rerun_sdk/rerun_sdk && touch /tmp/fake_rerun_sdk/rerun_sdk/__init__.py
cat > /tmp/fake_rerun_sdk/pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
[project]
name = "rerun-sdk"
version = "0.26.2"
requires-python = ">=3.9"
[tool.setuptools]
packages = ["rerun_sdk"]
EOF
pip install /tmp/fake_rerun_sdk
```

### Install lerobot 0.5.0, not 0.4.4 — despite what requirements.txt says

`requirements.txt` (captured from the lab server, 2026-05-13) pins `lerobot==0.4.4`.
**Do not install that version for PI0Fast.** Its `modeling_pi0_fast.py` unconditionally
requires `transformers.models.siglip.check`, a module that does not exist in *any*
transformers 4.x release — including the exact `transformers==4.57.6` that 0.4.4's own
`pyproject.toml` declares as the compatible upper range. This is a genuine upstream
packaging bug in the 0.4.4 release (confirmed against the immutable `v0.4.4` git tag),
not an environment misconfiguration — you cannot pip your way out of it while pinned to
0.4.4.

Use `lerobot==0.5.0` instead, which removes that broken check and requires
`transformers>=5.3.0,<6.0.0`. Verified compatible with everything `pi0fast_server.py`
touches (`_paligemma_tokenizer`, `action_tokenizer`, `config.max_action_tokens` /
`fast_skip_tokens` / `max_action_dim`, and the manual LoRA attribute path
`policy.model.paligemma_with_expert.paligemma.model.language_model.layers[i].self_attn.
{q,k,v,o}_proj` / `paligemma.lm_head`) — these are all unchanged between 0.4.4 and 0.5.0.

Install command (module order per above):
```bash
pip install "lerobot[transformers-dep]==0.5.0"
```
The `[transformers-dep]` extra is required — bare `lerobot==0.5.0` doesn't pull in
`transformers` at all (PI0Fast needs it; it's an optional extra in lerobot's packaging).

## 3. Priming the HF cache for offline compute-node runs

Compute nodes have no internet. Everything `PI0FastPolicy.from_pretrained(...)` needs
must already be in `$SCRATCH/hf_cache` (`HF_HOME`) before the job runs. On the login
node (lightweight steps only, per §1):

```bash
export HF_HOME=$SCRATCH/hf_cache
huggingface_hub.snapshot_download('lerobot/pi0fast-base')   # ~5GB, I/O-bound, fine here
```

`google/paligemma-3b-pt-224` (the tokenizer PI0Fast borrows from) is **gated** — the HF
account running this needs to have accepted its license on huggingface.co, and be
logged in here via `hf auth login` (do this interactively, `! hf auth login`, so the
token never lands in a transcript). Prime just the tokenizer, not the model, to keep it
lightweight:

```bash
python3 -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224', trust_remote_code=True,
                               add_eos_token=True, add_bos_token=False)
"
```

### Known gotcha: `lerobot/fast-action-tokenizer` doesn't load offline

This repo (the FAST action tokenizer PI0Fast uses) has **no `config.json`** anywhere in
it (root or `bpe_tokenizer/` subfolder — confirmed via `list_repo_files`). Loading it via
`AutoProcessor.from_pretrained(..., trust_remote_code=True)` handles that missing file
gracefully online (a 404 is treated as "doesn't exist, use defaults") but **hard-fails
offline** (`LocalEntryNotFoundError`) — huggingface_hub's negative-cache markers for this
specific nested lookup don't get written even after repeated online priming attempts.
This looks like a real `huggingface_hub`/`transformers` offline-mode bug for
config-less custom processors, not something fixable by re-downloading harder.

**Working fix** — bypass Hub resolution for this one repo entirely by pointing at a
local directory instead of the Hub id:

```bash
python3 -c "
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('lerobot/fast-action-tokenizer', trust_remote_code=True)
p.save_pretrained('$SCRATCH/hf_cache/local_fast_action_tokenizer')
"
```

Then patch **our local cached copy** of `pi0fast-base`'s config (not upstream, not
`pi0fast_server.py`) to point at that local directory:

```bash
python3 -c "
import json, glob
path = glob.glob('$SCRATCH/hf_cache/hub/models--lerobot--pi0fast-base/snapshots/*/config.json')[0]
d = json.load(open(path))
d['action_tokenizer_name'] = '$SCRATCH/hf_cache/local_fast_action_tokenizer'
json.dump(d, open(path, 'w'), indent=2)
"
```

Local paths bypass HTTP/Hub resolution entirely in `from_pretrained`, sidestepping the
offline-cache bug. Verify with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` set before
trusting this in a job — don't assume it worked just because the patch applied cleanly.

## 4. Job sizing: host RAM, not just GPU memory

`pi0fast_server.py`'s FSDP setup loads the **full fp32 model on CPU independently per
rank** before sharding across GPUs (a deliberate workaround so `from_pretrained` doesn't
OOM the GPUs before FSDP can shard). This means host RAM scales with
`world_size × full_model_size` (~13.8GB fp32), not just per-GPU VRAM. A `--mem` request
sized only for "the model" (e.g. 64G) will get OOM-killed by the kernel (`oom_kill`,
`exitcode -9`, SIGKILL) partway through loading — this is silent until it happens, no
graceful error, just a killed process.

For `--nproc_per_node=2` on Narval's standard 4×A100/249GB/48-core nodes, request at
least `--mem=120G` (the proportional 2-of-4-GPU share of a node). Don't undersize this.

## 5. `transformers` pin: the version *within* lerobot's declared range matters

Installing `lerobot[transformers-dep]==0.5.0` bare resolves to whatever's newest inside
its declared `transformers>=5.3.0,<6.0.0` range — at time of writing that's `5.16.1`.
That newer version has drifted internally from what 0.5.0 was actually validated
against: the SigLIP vision-tower `state_dict` key nesting changed (checkpoint has
`vision_tower.vision_model.embeddings...`, model expects `vision_tower.embeddings...`
without the extra nesting — logs a "Could not load state dict: Missing/Unexpected
key(s)" warning covering the *entire* vision tower, meaning it silently trains with a
randomly-initialized vision encoder if you don't catch it), and separately
`create_causal_mask()`'s signature dropped/changed the `cache_position` kwarg, which
crashes the first forward pass with `TypeError: create_causal_mask() got an unexpected
keyword argument 'cache_position'`.

**Fix**: explicitly pin the lower bound after installing lerobot:
```bash
pip install "transformers==5.3.0"
```
Verified working end-to-end (job 1802424: clean model load, no state_dict warning, real
training steps with decreasing loss). Don't trust "no error during `pip install`" — the
state_dict warning above is silent/non-fatal by design (a `try/except` around
`load_state_dict` in `pi0fast_server.py`), so you must actually grep the job log for
`Could not load state dict|Missing key|Unexpected key`, not just check the exit code.

## 6. Walltime: one epoch is ~65-70 min on 2×A100 — size `--time` for the real epoch count

Narval's GPU partitions bucket jobs into tiers by requested walltime
(`gpubase_bynode_b1`=3h, `b2`=12h, `b3`=24h, ...) — asking for exactly 3h for a 10-epoch
run silently under-budgets it. On this dataset (7864 samples, batch=4, chunk=30,
2×A100), each epoch takes ~65-70 min, so 10 epochs needs **~11-12h**, not 3h.

Job 1802424 (`--time=03:00:00`, 10 epochs requested) trained epochs 1-2 successfully
(loss 6.17 → 4.98, checkpoints saved cleanly), then stopped silently mid-epoch-3 with
**exit code 0, no traceback, no OOM** (peak RSS 78.7GB/120GB) — `sacct` even reported
`COMPLETED`, not `TIMEOUT`. The exact kill mechanism wasn't confirmed (Slurm's own
accounting didn't flag it as a timeout), but the timing lines up with the 3h budget
being simply too small for the requested epoch count, and that's the fix regardless of
mechanism. **Don't trust `State=COMPLETED` / exit code 0 as proof all requested epochs
ran** — grep the log for `Epoch N/M  loss=` lines and count them against what you asked
for.

**Recovery pattern** — resume from the last good checkpoint rather than restarting:
```bash
--resume_from $OUTPUT/checkpoint_epochN --epochs $((TOTAL - N))   # NOT --epochs $TOTAL
```
(`start_epoch` is parsed from the checkpoint dir name; `total_epochs = start_epoch +
args.epochs`, so `--epochs` must be the *remaining* count, not the original total.)
Budget `--time` generously above the estimated need (e.g. request the `b2` 12h tier for
an ~9-10h estimated run) rather than cutting it close.

## 7. Reference: known-working setup

- Venv: `~/venv/pi0fast_lerobot` (Python 3.12.4, `lerobot[transformers-dep]==0.5.0`,
  `transformers==5.3.0` — see §5, do not let pip drift this to a newer version)
- Job script: `slurm_finetune_pi0fast_so101.sh` (2×A100, 120G mem, 11h) — copy this as
  the template for future PI0Fast fine-tuning jobs on Narval rather than starting from
  the README's lab-server instructions.
- `HF_HOME=$SCRATCH/hf_cache`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` inside the
  job; unset those three on the login node when priming the cache.
