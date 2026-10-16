#!/usr/bin/env python3
"""
Apertura vision conformance fixtures — Gemma-4 vision tower + multimodal prefill.

Runs the HuggingFace PyTorch reference (Gemma4ForConditionalGeneration, bf16, eager
attention) over ONE deterministic test image and a chat prompt that contains it, and
serializes every intermediate Apertura needs to verify the port op-for-op:
the image processor, the vision tower (patch embedder, 16 encoder layers, pooler), the
multimodal embedder, the soft-token scatter into the text embeddings, the decoder stack,
and a greedy continuation.

Device: CPU by default (bit-deterministic across runs, and the same device the E2B text
fixtures used). APERTURA_DEVICE=mps for a faster run.

The test image is drawn here (gradient background + three coloured shapes), saved as
`<FIX_NAME>.png` beside the fixture so the Apertura side can run the SAME file through
its own preprocessor. 640x400 is deliberate: it needs a resize (the processor path most
ports get wrong) and lands on 624x1008 -> 39x63 = 2457 patches -> 273 soft tokens, so the
2520-patch budget is padded (63 padding patches at (-1,-1)) and the soft-token count is
NOT the nominal 280 — the chat-template expansion must follow the processor, not the
constant.

Captures (fixtures safetensors; bf16 unless noted):
  image processor
    pixel_values            f32  [1, 2520, 768]   patches of the resized, 1/255-rescaled image (0 = padding)
    image_position_ids      i32  [1, 2520, 2]     (x, y) patch coordinates, (-1, -1) = padding
    num_soft_tokens         i32  [1]              valid soft tokens (273)
  vision tower (model.vision_tower)
    vision.patch_embed           [1, 2520, 768]   patch embedder output (input_proj + 2-D positions)
    vision.rope_cos/.rope_sin    [1, 2520, 64]    axial 2-D RoPE tables (H-H-W-W layout)
    V0.input_layernorm           [1, 2520, 768]   layer-0 op probes
    V0.q_norm V0.k_norm V0.v_norm[1, 2520, 12, 64] post-norm, pre-RoPE
    V0.q_rope V0.k_rope          [1, 2520, 12, 64]
    V0.attn_oproj                [1, 2520, 768]
    V0.pre_feedforward_layernorm [1, 2520, 768]
    V0.mlp                       [1, 2520, 768]
    vision.layer_out.{i} i<16    [1, 2520, 768]   encoder layer outputs (padding rows are garbage-but-masked)
    vision.pooler           f32  [1, 280, 768]    3x3 average pool x sqrt(768), f32 (before strip)
    vision.tower_out             [273, 768]       Gemma4VisionModel output (padding stripped, cast bf16)
  embedder
    embed_vision                 [273, 1536]      Gemma4MultimodalEmbedder output = the soft tokens
  text model (one forward of the chat prompt with the image)
    input_ids               i32  [1, S]
    mm_token_type_ids       i32  [1, S]           1 at image tokens
    inputs_embeds                [1, S, 1536]     text embeddings with soft tokens scattered in
    per_layer_inputs             [1, S, 35, 256]  PLE token-identity component (image tokens -> pad row)
    layer_out.{i} i<35           [1, S, 1536]
    final_norm                   [1, S, 1536]
    logits_last                  [262144]         last-position logits (post softcap)
    greedy_tokens           i32  [20]

Run once inside ~/.venvs/apertura-torch:
  python Tools/generate_vision_fixtures.py
Env: APERTURA_MODEL (snapshot dir), APERTURA_FIXTURES (basename, default fixtures_vision_e2b),
     APERTURA_DEVICE (cpu|mps), APERTURA_IMAGE (use an existing image instead of drawing one).
"""

import json
import math
import os
import time

import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file

MODEL_DIR = os.environ.get(
    "APERTURA_MODEL",
    os.path.expanduser("~/.cache/huggingface/sources/gemma-4-E2B-it-qat-q4_0-unquantized"))
FIX_NAME = os.environ.get("APERTURA_FIXTURES", "fixtures_vision_e2b")
DEVICE = os.environ.get("APERTURA_DEVICE", "cpu")
N_GREEDY = 20
USER_TEXT = "Describe this picture in one sentence."

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "aptransformerTests", "Fixtures"))
os.makedirs(OUT_DIR, exist_ok=True)
IMAGE_PATH = os.environ.get("APERTURA_IMAGE", os.path.join(OUT_DIR, FIX_NAME + ".png"))

torch.manual_seed(0)
torch.set_grad_enabled(False)

import transformers.models.gemma4.modeling_gemma4 as G4  # noqa: E402

captured = {}


def stash(name, t):
    captured[name] = t.detach().to("cpu").contiguous()


def draw_test_image(path):
    """Deterministic 640x400 PNG: horizontal RGB gradient, red disc, blue box, green triangle."""
    w, h = 640, 400
    img = Image.new("RGB", (w, h))
    px = img.load()
    for x in range(w):
        r = int(255 * x / (w - 1))
        for y in range(h):
            g = int(255 * y / (h - 1))
            b = 255 - (r + g) // 2
            px[x, y] = (r, g, b)
    d = ImageDraw.Draw(img)
    d.ellipse([60, 80, 260, 280], fill=(220, 30, 30), outline=(0, 0, 0), width=4)
    d.rectangle([330, 60, 560, 200], fill=(30, 60, 220), outline=(255, 255, 255), width=4)
    d.polygon([(320, 370), (470, 230), (620, 370)], fill=(30, 200, 60), outline=(0, 0, 0))
    img.save(path, format="PNG")
    return img


def main():
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    if os.path.exists(IMAGE_PATH) and "APERTURA_IMAGE" in os.environ:
        img = Image.open(IMAGE_PATH).convert("RGB")
    else:
        img = draw_test_image(IMAGE_PATH)
    print(f"[vision] image {IMAGE_PATH} {img.size}")

    print(f"[vision] loading {MODEL_DIR} (bf16, {DEVICE}) ...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map={"": DEVICE}, attn_implementation="eager")
    model.eval()
    print(f"[vision] loaded in {time.time() - t0:.1f}s")

    cfg = model.config
    vt = model.model.vision_tower
    tm = model.model.language_model
    vcfg = cfg.vision_config
    print(f"[vision] vision hidden={vcfg.hidden_size} layers={vcfg.num_hidden_layers} heads={vcfg.num_attention_heads} "
          f"head_dim={vcfg.head_dim} rope_theta={vcfg.rope_parameters['rope_theta']} clipped={vcfg.use_clipped_linears} "
          f"standardize={vcfg.standardize} | text use_bidirectional_attention={cfg.text_config.use_bidirectional_attention}")

    # ---- processor: chat prompt with one image ----
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}]}]
    prompt_text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=prompt_text, images=[img], return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"[vision] {k}: {tuple(v.shape)} {v.dtype}")
    pixel_values = inputs["pixel_values"]
    pos_ids = inputs["image_position_ids"]
    n_soft = int(inputs["num_soft_tokens_per_image"][0]) if "num_soft_tokens_per_image" in inputs \
        else int((pos_ids[0, :, 0] >= 0).sum()) // 9
    stash("pixel_values", pixel_values.to(torch.float32))
    stash("image_position_ids", pos_ids.to(torch.int32))
    stash("num_soft_tokens", torch.tensor([n_soft], dtype=torch.int32))
    stash("input_ids", inputs["input_ids"].to(torch.int32))
    if "mm_token_type_ids" in inputs:
        stash("mm_token_type_ids", inputs["mm_token_type_ids"].to(torch.int32))
    ids = inputs["input_ids"][0].tolist()
    n_img_tok = sum(1 for t in ids if t == cfg.image_token_id)
    print(f"[vision] prompt: {len(ids)} tokens, {n_img_tok} image tokens (soft tokens {n_soft})")
    assert n_img_tok == n_soft, "image placeholder count must equal the processor's soft-token count"

    # ---- hooks ----
    handles = []

    def hook(name, index=None):
        def fn(mod, inp, out):
            t = out
            if isinstance(out, (tuple, list)):
                t = out[0] if index is None else out[index]
            if hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            stash(name, t)
        return fn

    H = handles.append
    H(vt.patch_embedder.register_forward_hook(hook("vision.patch_embed")))
    H(vt.encoder.rotary_emb.register_forward_hook(hook("vision.rope_cos", 0)))
    H(vt.encoder.rotary_emb.register_forward_hook(hook("vision.rope_sin", 1)))
    for i, layer in enumerate(vt.encoder.layers):
        H(layer.register_forward_hook(hook(f"vision.layer_out.{i}")))
    L0 = vt.encoder.layers[0]
    H(L0.input_layernorm.register_forward_hook(hook("V0.input_layernorm")))
    H(L0.self_attn.q_norm.register_forward_hook(hook("V0.q_norm")))
    H(L0.self_attn.k_norm.register_forward_hook(hook("V0.k_norm")))
    H(L0.self_attn.v_norm.register_forward_hook(hook("V0.v_norm")))
    H(L0.self_attn.o_proj.register_forward_hook(hook("V0.attn_oproj")))
    H(L0.pre_feedforward_layernorm.register_forward_hook(hook("V0.pre_feedforward_layernorm")))
    H(L0.mlp.register_forward_hook(hook("V0.mlp")))
    H(vt.pooler.register_forward_hook(hook("vision.pooler", 0)))
    H(vt.register_forward_hook(hook("vision.tower_out")))
    H(model.model.embed_vision.register_forward_hook(hook("embed_vision")))

    # Multidimensional RoPE: layer 0 calls it for q then k.
    _orig_mrope = G4.apply_multidimensional_rope
    rope_calls = {"n": 0}

    def traced_mrope(x, cos, sin, position_ids, unsqueeze_dim=2):
        out = _orig_mrope(x, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim)
        n = rope_calls["n"]
        if n == 0:
            stash("V0.q_rope", out)
        elif n == 1:
            stash("V0.k_rope", out)
        rope_calls["n"] += 1
        return out
    G4.apply_multidimensional_rope = traced_mrope

    # Text side: what the language model receives after the scatter.
    def lm_pre(mod, args, kwargs):
        if kwargs.get("inputs_embeds") is not None:
            stash("inputs_embeds", kwargs["inputs_embeds"])
        if kwargs.get("per_layer_inputs") is not None:
            stash("per_layer_inputs", kwargs["per_layer_inputs"])
    H(tm.register_forward_pre_hook(lm_pre, with_kwargs=True))
    for i, layer in enumerate(tm.layers):
        H(layer.register_forward_hook(hook(f"layer_out.{i}")))
    H(tm.norm.register_forward_hook(hook("final_norm")))

    # ---- one forward ----
    dev_inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask", "pixel_values", "image_position_ids",
                           "mm_token_type_ids")}
    t0 = time.time()
    out = model(**dev_inputs, use_cache=False)
    logits = out.logits[0, -1]
    stash("logits_last", logits)
    last_argmax = int(torch.argmax(logits.float()).item())
    print(f"[vision] forward {time.time() - t0:.1f}s; argmax(last) = {last_argmax} -> {proc.tokenizer.decode([last_argmax])!r}")
    for h in handles:
        h.remove()
    G4.apply_multidimensional_rope = _orig_mrope

    # ---- greedy continuation ----
    t0 = time.time()
    gen = model.generate(**dev_inputs, max_new_tokens=N_GREEDY, do_sample=False, num_beams=1, use_cache=True)
    greedy = gen[0, len(ids):].to(torch.int32).cpu()
    stash("greedy_tokens", greedy)
    print(f"[vision] greedy ({time.time() - t0:.1f}s): {greedy.tolist()}")
    print(f"[vision] greedy text: {proc.tokenizer.decode(greedy.tolist())!r}")

    # ---- serialize ----
    save_path = os.path.join(OUT_DIR, FIX_NAME + ".safetensors")
    save_file(captured, save_path)
    meta = {
        "model_id": MODEL_DIR,
        "device": DEVICE,
        "image": os.path.basename(IMAGE_PATH),
        "image_size": list(img.size),
        "resized_hw": [int(pos_ids[0, :, 1].max()) + 1, int(pos_ids[0, :, 0].max()) + 1],
        "num_patches_valid": int((pos_ids[0, :, 0] >= 0).sum()),
        "num_soft_tokens": n_soft,
        "user_text": USER_TEXT,
        "prompt_text": prompt_text,
        "seq_len": len(ids),
        "image_token_id": cfg.image_token_id,
        "boi_token_id": cfg.boi_token_id,
        "eoi_token_id": cfg.eoi_token_id,
        "use_bidirectional_attention": cfg.text_config.use_bidirectional_attention,
        "last_argmax": last_argmax,
        "n_greedy": N_GREEDY,
        "greedy_text": proc.tokenizer.decode(greedy.tolist()),
        "tensor_keys": sorted(captured.keys()),
    }
    with open(os.path.join(OUT_DIR, FIX_NAME + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    size_mb = os.path.getsize(save_path) / 1e6
    print(f"[vision] wrote {save_path} ({len(captured)} tensors, {size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
