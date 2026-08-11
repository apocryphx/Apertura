"""
CPU/bf16 reference fixtures for several prompts, model loaded ONCE.

bf16 (not fp32) so precision matches MLX's hardcoded computeDtype exactly and
only the implementation differs. Prompts chosen to be mostly non-degenerate:
the original "quick brown fox" drives the model into a repetition loop where
decision margins are tiny, which is the weakest possible place to test.
"""
import json, math, os, sys, time
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from safetensors.torch import save_file

MODEL_DIR = os.environ.get("APERTURA_MODEL",
    "/Volumes/Macintosh HD - Data/Users/apocryphx/Models/gemma-4-31B")
OUTDIR    = sys.argv[1]
N_GREEDY  = 12

# Op-level probes are ~500 KB/layer; layer_out is ~45 KB/layer. Capturing every layer's ops
# gives a 40 MB fixture, so default to a spread covering both attention types and full depth.
# Set APERTURA_PROBE_LAYERS=all to capture every layer (use for diagnosis, not for check-in).
_pl = os.environ.get("APERTURA_PROBE_LAYERS", "0,5,15,30,45,59")
PROBE_LAYERS = None if _pl == "all" else {int(v) for v in _pl.split(",")}

PROMPTS = [
    ("fox",     "The quick brown fox"),          # control: matches earlier runs
    ("paris",   "The capital of France is"),
    ("moon",    "In 1969, humans first walked on"),
    ("code",    "def fibonacci(n):"),
    ("water",   "Water boils at a temperature of"),
]

torch.set_grad_enabled(False)
os.makedirs(OUTDIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
t0 = time.time()
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager", device_map={"": "cpu"})
model.eval()
print(f"[ref] loaded in {time.time()-t0:.1f}s", flush=True)

tm = model.model.language_model if hasattr(model.model, "language_model") else model.model
layers, H = tm.layers, tm.config.hidden_size

captured, CAPTURE = {}, {"on": True}
def keep(name):
    def hook(mod, inp, out):
        if not CAPTURE["on"]:
            return
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            captured[name] = t.detach().cpu().contiguous()
    return hook

tm.embed_tokens.register_forward_hook(keep("embed_scaled"))
tm.norm.register_forward_hook(keep("final_norm"))
model.lm_head.register_forward_hook(keep("logits_pre_softcap"))
for i, layer in enumerate(layers):
    layer.register_forward_hook(keep(f"layer_out.{i}"))   # every layer: cheap, localises depth
    if PROBE_LAYERS is not None and i not in PROBE_LAYERS:
        continue
    sa = layer.self_attn
    layer.input_layernorm.register_forward_hook(keep(f"L{i}.input_layernorm"))
    layer.post_attention_layernorm.register_forward_hook(keep(f"L{i}.post_attention_layernorm"))
    layer.pre_feedforward_layernorm.register_forward_hook(keep(f"L{i}.pre_feedforward_layernorm"))
    layer.post_feedforward_layernorm.register_forward_hook(keep(f"L{i}.post_feedforward_layernorm"))
    layer.mlp.register_forward_hook(keep(f"L{i}.mlp"))
    sa.o_proj.register_forward_hook(keep(f"L{i}.attn_oproj"))
    for nm in ("q_norm", "k_norm", "v_norm"):
        if getattr(sa, nm, None) is not None:
            getattr(sa, nm).register_forward_hook(keep(f"L{i}.{nm}"))


# ---- RoPE capture -------------------------------------------------------------
# Gemma-4 applies RoPE as a free function, not a module, so forward hooks never see it.
# Patch the function instead. It is called once per tensor, so q and k arrive as separate
# calls; they are told apart by head count (num_attention_heads vs num_key_value_heads)
# rather than call order, which is implementation-dependent. p-RoPE zeroes 192 of 256
# frequencies on global layers, so this is exactly where a subtle bug would hide.
import transformers.models.gemma4.modeling_gemma4 as _g4
_orig_rope = _g4.apply_rotary_pos_emb
_rope_state = {"layer": 0, "seen_k": False}
_NQ = tm.config.num_attention_heads

def _rope_probe(x, cos, sin, unsqueeze_dim=1, **kw):
    out = _orig_rope(x, cos, sin, unsqueeze_dim, **kw)
    if CAPTURE["on"]:
        # layout is [batch, seq, heads, head_dim] with unsqueeze_dim=2 (verified by probe),
        # so heads is axis 2. q has num_attention_heads (32); k has the per-layer kv count
        # (16 sliding / 4 global), which is why heads==NQ is the discriminator, not a fixed pair.
        heads = x.shape[2] if x.ndim == 4 else -1
        which = "q_rope" if heads == _NQ else "k_rope"
        L = _rope_state["layer"]
        if PROBE_LAYERS is None or L in PROBE_LAYERS:
            captured[f"L{L}.{which}"] = out.detach().cpu().contiguous()
        if which == "k_rope":
            _rope_state["layer"] += 1
    return out

_g4.apply_rotary_pos_emb = _rope_probe

summary = {}
for tag, prompt in PROMPTS:
    captured.clear()
    _rope_state['layer'] = 0
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    CAPTURE["on"] = True
    t0 = time.time()
    out = model(input_ids=ids, use_cache=False)
    lg = out.logits.detach().to(torch.float32)
    captured["logits"] = lg[0].clone().contiguous()
    captured["logits_last"] = lg[0, -1].clone().contiguous()
    captured["input_ids"] = ids[0].to(torch.int32).clone().contiguous()
    last_argmax = int(lg[0, -1].argmax())

    # margin at the final position: how decisive is this prompt?
    top = torch.topk(lg[0, -1], 2)
    margin = (top.values[0] - top.values[1]).item()
    rel_margin = margin / abs(top.values[0].item())

    CAPTURE["on"] = False
    cur, greedy = ids.clone(), []
    for _ in range(N_GREEDY):
        l2 = model(input_ids=cur, use_cache=False).logits[0, -1]
        nxt = int(l2.argmax()); greedy.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=cur.dtype)], dim=1)
    captured["greedy_tokens"] = torch.tensor(greedy, dtype=torch.int32)

    st = os.path.join(OUTDIR, f"ref_{tag}.safetensors")
    save_file(captured, st)
    meta = {"model_id": MODEL_DIR, "reference": "cpu/bfloat16", "prompt": prompt,
            "seq_len": int(ids.shape[1]), "hidden_size": H, "embed_scale": math.sqrt(H),
            "last_argmax": last_argmax, "n_greedy": N_GREEDY,
            "probe_layers": sorted(PROBE_LAYERS) if PROBE_LAYERS else list(range(len(layers))),
            "tensor_keys": sorted(captured.keys())}
    json.dump(meta, open(st.replace(".safetensors", "_meta.json"), "w"), indent=2)
    summary[tag] = {"prompt": prompt, "greedy": greedy, "rel_margin": rel_margin}
    print(f"[ref] {tag:6s} {time.time()-t0:5.1f}s  margin={rel_margin:.3e}  greedy={greedy}", flush=True)

json.dump(summary, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
print("[ref] done", flush=True)
