#!/usr/bin/env python3
"""
Apertura conformance fixture generator — Gemma-4 31B text decoder.

Runs the HuggingFace PyTorch reference (the structural oracle) over a fixed,
deterministic prompt and serializes every intermediate activation Apertura needs
to verify its forward pass, op-for-op.

DECISIONS (see plan):
  - BF16 compute. A full F32 forward of 31B needs ~131 GB > 128 GB RAM, so we run
    in the canonical weight dtype (bf16). Per-token argmax is the exact gate;
    numeric tolerances are set to the bf16 floor on the Apertura side.
  - attn_implementation="eager" so attention scores (pre/post-softmax) are real
    tensors we can hook, not fused kernels.
  - Probe layers: 0 (sliding/local) and 5 (full/global). These exercise both
    attention regimes incl. dual KV heads, dual head_dim, and dual RoPE.

Captures (all saved into fixtures.safetensors, bf16 unless noted):
  input_ids (int32), position_ids (int32)
  embed_scaled                      embedding lookup * sqrt(hidden_size)
  layer_out.{i}  for i in 0..59     decoder layer output (post layer_scalar)  -> chain validation
  final_norm                        model.norm output
  logits_pre_softcap                lm_head(hidden) before tanh softcap
  logits                            final logits (post softcap)  -> argmax gate at last position
  greedy_tokens (int32)             20-token greedy continuation

  For each probe layer L in {0,5}:
    L{L}.input_layernorm
    L{L}.q_prenorm_postrope? -> we capture:
       L{L}.q_norm            q after q_norm, BEFORE RoPE   (shape [b,seq,heads,head_dim])
       L{L}.k_norm            k after k_norm, BEFORE RoPE
       L{L}.v_norm            v after v_norm (no RoPE ever)
       L{L}.q_rope            q after RoPE
       L{L}.k_rope            k after RoPE
       L{L}.rope_cos / .rope_sin   the cos/sin used (for isolated RoPE tests)
       L{L}.scores_pre        attn weights after scaling+mask, pre-softmax
       L{L}.scores_post       attn weights post-softmax
       L{L}.attn_oproj        self_attn output (post o_proj)
       L{L}.post_attention_layernorm
       L{L}.pre_feedforward_layernorm
       L{L}.mlp               mlp output
       L{L}.post_feedforward_layernorm

Run once inside ~/torch-ref; commit fixtures. Ground truth — do not regenerate casually.
"""

import os
import json
import math
import torch
from safetensors.torch import save_file

MODEL_ID = os.environ.get("APERTURA_MODEL", "google/gemma-4-31b-it")
FIX_NAME = os.environ.get("APERTURA_FIXTURES", "fixtures")  # output basename
PROMPT = "The quick brown fox"
PROBE_LAYERS = [0, 5]
N_GREEDY = 20

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aptransformerTests", "Fixtures")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(0)

import transformers.models.gemma4.modeling_gemma4 as G4

captured = {}  # name -> cpu tensor


def stash(name, t):
    captured[name] = t.detach().to("cpu").contiguous()


# ---- monkeypatch apply_rotary_pos_emb to capture pre/post-RoPE for probe layers ----
# Calls happen in deterministic order: for each layer, q then k. Layer L -> calls (2L, 2L+1).
_orig_rope = G4.apply_rotary_pos_emb
_rope_call = {"n": 0}


def traced_rope(x, cos, sin, unsqueeze_dim=1):
    out = _orig_rope(x, cos, sin, unsqueeze_dim=unsqueeze_dim)
    idx = _rope_call["n"]
    layer = idx // 2
    is_q = (idx % 2) == 0
    if layer in PROBE_LAYERS:
        tag = "q" if is_q else "k"
        stash(f"L{layer}.{tag}_rope", out)
        if is_q:  # cos/sin identical for q and k within a layer; store once
            stash(f"L{layer}.rope_cos", cos)
            stash(f"L{layer}.rope_sin", sin)
    _rope_call["n"] += 1
    return out


G4.apply_rotary_pos_emb = traced_rope

# ---- monkeypatch eager_attention_forward to capture scores for probe layers ----
# Called once per layer, in order. Use module.layer_idx to identify the layer.
_orig_eager = G4.eager_attention_forward


def traced_eager(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kwargs):
    li = getattr(module, "layer_idx", -1)
    if li in PROBE_LAYERS:
        if scaling is None:
            scaling = module.head_dim ** -0.5
        ks = G4.repeat_kv(key, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if softcap is not None:
            w = torch.tanh(w / softcap) * softcap
        if attention_mask is not None:
            w = w + attention_mask
        stash(f"L{li}.scores_pre", w)
        wp = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        stash(f"L{li}.scores_post", wp)
    return _orig_eager(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, softcap=softcap, **kwargs)


G4.eager_attention_forward = traced_eager


def main():
    from transformers import AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    print(f"[fixtures] loading {MODEL_ID} (bf16, mps) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Load the exact checkpoint architecture (weight names are model.language_model.*).
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": "mps"},
        attn_implementation="eager",
    )
    model.eval()

    # The multimodal wrapper exposes the text decoder under .model.language_model.
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    cfg = text_model.config
    H = cfg.hidden_size
    print(f"[fixtures] hidden_size={H} layers={cfg.num_hidden_layers} "
          f"embed_scale=sqrt(H)={math.sqrt(H):.4f}")

    # ---- register forward hooks ----
    handles = []

    def hook(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            stash(name, t)
        return fn

    handles.append(text_model.embed_tokens.register_forward_hook(hook("embed_scaled")))
    handles.append(text_model.norm.register_forward_hook(hook("final_norm")))
    for i, layer in enumerate(text_model.layers):
        handles.append(layer.register_forward_hook(hook(f"layer_out.{i}")))
    for L in PROBE_LAYERS:
        layer = text_model.layers[L]
        sa = layer.self_attn
        handles.append(layer.input_layernorm.register_forward_hook(hook(f"L{L}.input_layernorm")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(hook(f"L{L}.post_attention_layernorm")))
        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(hook(f"L{L}.pre_feedforward_layernorm")))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(hook(f"L{L}.post_feedforward_layernorm")))
        handles.append(layer.mlp.register_forward_hook(hook(f"L{L}.mlp")))
        handles.append(sa.q_norm.register_forward_hook(hook(f"L{L}.q_norm")))
        if getattr(sa, "k_norm", None) is not None:
            handles.append(sa.k_norm.register_forward_hook(hook(f"L{L}.k_norm")))
        if getattr(sa, "v_norm", None) is not None:
            handles.append(sa.v_norm.register_forward_hook(hook(f"L{L}.v_norm")))
        handles.append(sa.o_proj.register_forward_hook(hook(f"L{L}.attn_oproj")))
    # lm_head pre-softcap
    handles.append(model.lm_head.register_forward_hook(hook("logits_pre_softcap")))

    # ---- tokenize (raw, deterministic; no chat template for conformance forward) ----
    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to("mps")
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, device="mps").unsqueeze(0)
    print(f"[fixtures] prompt={PROMPT!r} -> {seq} tokens: {input_ids[0].tolist()}")

    # ---- single forward pass (captures all hooks + patched scores/rope) ----
    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    logits = out.logits  # post softcap
    stash("logits", logits)
    stash("input_ids", input_ids.to(torch.int32))
    stash("position_ids", position_ids.to(torch.int32))
    last_argmax = int(torch.argmax(logits[0, -1]).item())
    print(f"[fixtures] argmax(last logits) = {last_argmax} -> {tok.decode([last_argmax])!r}")

    for h in handles:
        h.remove()

    # ---- greedy continuation (20 tokens) ----
    print("[fixtures] greedy generating ...")
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            max_new_tokens=N_GREEDY,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    greedy = gen[0, seq:].to(torch.int32).cpu()
    stash("greedy_tokens", greedy)
    print(f"[fixtures] greedy tokens: {greedy.tolist()}")
    print(f"[fixtures] greedy text: {tok.decode(greedy.tolist())!r}")

    # ---- serialize ----
    # safetensors wants contiguous; bf16 supported. ints kept as int32.
    save_path = os.path.join(OUT_DIR, FIX_NAME + ".safetensors")
    save_file(captured, save_path)
    meta = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "probe_layers": PROBE_LAYERS,
        "seq_len": seq,
        "hidden_size": H,
        "embed_scale": math.sqrt(H),
        "last_argmax": last_argmax,
        "n_greedy": N_GREEDY,
        "tensor_keys": sorted(captured.keys()),
    }
    with open(os.path.join(OUT_DIR, FIX_NAME + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fixtures] wrote {save_path} ({len(captured)} tensors)")
    print(f"[fixtures] wrote fixtures_meta.json")


if __name__ == "__main__":
    main()
