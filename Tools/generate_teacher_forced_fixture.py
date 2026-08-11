"""
Teacher-forced PyTorch reference generator.

Problem it solves: a normal capture records a CHAINED forward, so layer i's output was
computed from that run's own layer i-1. Comparing two chained runs at depth therefore
measures accumulated divergence, not per-layer fidelity — which is why PyT-CPU vs PyT-MPS
fell from 99.9% at layer 0 to 23.9% at layer 30, and why PyT-MPS vs MLX-Metal was invalid
at depth (MLX was fed CPU boundaries, PyTorch-MPS was fed its own).

Fix: capture the boundary tensors once on a chosen reference device, then re-run with a
forward_pre_hook on every decoder layer that REPLACES hidden_states with the stored
boundary. Every layer then computes from identical inputs regardless of backend, so all
four cells of the {PyTorch,MLX} x {CPU,Metal} grid are comparable at every depth.

The model still does its own mask / RoPE / position plumbing — only the hidden state is
swapped — so nothing about attention is reimplemented here.

  pass 1:  gen_teacher_forced.py <out_dir> cpu            -> writes boundaries + tf_<dev>.safetensors
  pass 2:  gen_teacher_forced.py <out_dir> mps <boundaries.safetensors>
"""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from safetensors.torch import save_file, load_file

MODEL_DIR = os.environ.get("APERTURA_MODEL",
    "/Volumes/Macintosh HD - Data/Users/apocryphx/Models/gemma-4-31B")
OUTDIR    = sys.argv[1]
DEV       = sys.argv[2] if len(sys.argv) > 2 else "cpu"
BOUNDS    = sys.argv[3] if len(sys.argv) > 3 else None
PROMPT    = "Water boils at a temperature of"

torch.set_grad_enabled(False)
os.makedirs(OUTDIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
ids = tok(PROMPT, return_tensors="pt", add_special_tokens=True).input_ids.to(DEV)

t0 = time.time()
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager", device_map={"": DEV}).eval()
print(f"[tf] loaded on {DEV} in {time.time()-t0:.1f}s", flush=True)

tm = model.model.language_model if hasattr(model.model, "language_model") else model.model
layers = tm.layers
print(f"[tf] {len(layers)} layers", flush=True)

captured = {}
def keep(name):
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            captured[name] = t.detach().to(torch.float32).cpu().contiguous()
    return hook

# Always record each layer's OUTPUT and its INPUT (the boundary entering it).
for i, layer in enumerate(layers):
    layer.register_forward_hook(keep(f"layer_out.{i}"))

def capture_input(i):
    def pre(mod, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if torch.is_tensor(hs):
            captured[f"layer_in.{i}"] = hs.detach().to(torch.float32).cpu().contiguous()
        return args, kwargs
    return pre

def inject_input(i, bounds):
    key = f"layer_in.{i}"
    def pre(mod, args, kwargs):
        if key not in bounds:
            return args, kwargs
        # exact widening back to bf16: the stored tensor is an fp32 view of bf16 values
        hs = bounds[key].to(device=DEV, dtype=torch.bfloat16)
        captured[f"layer_in.{i}"] = hs.detach().to(torch.float32).cpu().contiguous()
        if args:
            return (hs,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = hs
        return args, kwargs
    return pre

mode = "TEACHER-FORCED" if BOUNDS else "chained (boundary capture)"
if BOUNDS:
    bounds = load_file(BOUNDS)
    print(f"[tf] injecting {sum(1 for k in bounds if k.startswith('layer_in.'))} boundaries", flush=True)
    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(inject_input(i, bounds), with_kwargs=True)
else:
    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(capture_input(i), with_kwargs=True)

t0 = time.time()
out = model(input_ids=ids, use_cache=False)
print(f"[tf] {mode} forward in {time.time()-t0:.1f}s  tensors={len(captured)}", flush=True)

lg = out.logits.detach().to(torch.float32).cpu()
captured["logits"] = lg[0].clone().contiguous()
captured["input_ids"] = ids[0].to(torch.int32).cpu().clone().contiguous()

tag = "tf" if BOUNDS else "chain"
path = os.path.join(OUTDIR, f"{tag}_{DEV}.safetensors")
save_file(captured, path)
json.dump({"model_id": MODEL_DIR, "device": DEV, "mode": mode, "prompt": PROMPT,
           "boundaries_from": BOUNDS, "tensor_keys": sorted(captured.keys())},
          open(path.replace(".safetensors", "_meta.json"), "w"), indent=2)
print(f"[tf] wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
print(f"[tf] argmax(last)={int(lg[0,-1].argmax())}", flush=True)
