#!/usr/bin/env python3
"""
Apertura vision fixtures — Gemma-4 UNIFIED family (12B QAT): embedder-only.

The unified architecture (Gemma4UnifiedForConditionalGeneration, model_type gemma4_unified) has
no vision tower: the processor merges 3x3 16-px patches into one 48x48x3 = 6912-vector per soft
token and Gemma4UnifiedVisionEmbedder embeds it (LN -> Dense+bias -> LN -> factorized 2-D pos-emb
-> LN -> RMSNorm(no scale) -> Linear 3840->3840). A full 12B forward does not fit a 24 GB Mac
(23.9 GB bf16), so this captures the EMBEDDER ONLY: the 10 vision tensors are loaded straight from
model.safetensors into a bare Gemma4UnifiedVisionEmbedder (bf16, CPU).

The end-to-end text gate (argmax + greedy with the image) for this family is PENDING a 128 GB
machine; on it, run Tools/generate_vision_fixtures.py with APERTURA_MODEL=<12B source>,
APERTURA_FIXTURES=fixtures_vision_unified_12b_full — it handles both families via AutoProcessor /
Gemma4UnifiedForConditionalGeneration (see the Unified note in that script's loader).

Captures (bf16 unless noted), same picture as the E2B fixture (copied beside this fixture):
  pixel_values         f32 [1, 280, 6912]   merged patches (row-major 48x48, channel-last), 0 = padding
  image_position_ids   i32 [1, 280, 2]      block (x, y), (-1, -1) = padding
  num_soft_tokens      i32 [1]              273
  u.ln1 u.dense u.ln2 u.pos_added u.pos_norm   [1, 280, 3840]   each embedder stage
  embed_vision             [1, 280, 3840]   pooler_output = the soft tokens (text hidden 3840)
"""
import json, os, struct, time
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_DIR = os.environ.get("APERTURA_MODEL", "/Volumes/Gemma 4/sources/gemma-4-12B-it-qat-q4_0-unquantized")
FIX_NAME = os.environ.get("APERTURA_FIXTURES", "fixtures_vision_unified_12b")
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aptransformerTests", "Fixtures"))
IMAGE_PATH = os.path.join(OUT_DIR, FIX_NAME + ".png")
torch.set_grad_enabled(False)

from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig
from transformers.models.gemma4_unified.image_processing_gemma4_unified import Gemma4UnifiedImageProcessor
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder

captured = {}
def stash(name, t): captured[name] = t.detach().cpu().contiguous()

def main():
    cfg = Gemma4UnifiedConfig.from_pretrained(MODEL_DIR)
    pc = json.load(open(os.path.join(MODEL_DIR, "processor_config.json")))["image_processor"]
    proc = Gemma4UnifiedImageProcessor(**{k: v for k, v in pc.items() if k != "image_processor_type"})
    img = Image.open(IMAGE_PATH).convert("RGB")
    feats = proc(images=[img], return_tensors="pt")
    pv, pos = feats["pixel_values"], feats["image_position_ids"]
    n_soft = int(feats["num_soft_tokens_per_image"][0])
    print(f"[unified] {img.size} -> pixel_values {tuple(pv.shape)} {pv.dtype}, positions {tuple(pos.shape)}, soft tokens {n_soft}")
    stash("pixel_values", pv.float()); stash("image_position_ids", pos.to(torch.int32))
    stash("num_soft_tokens", torch.tensor([n_soft], dtype=torch.int32))

    emb = Gemma4UnifiedVisionEmbedder(cfg.vision_config, cfg.text_config).to(torch.bfloat16).eval()
    t0 = time.time()
    with safe_open(os.path.join(MODEL_DIR, "model.safetensors"), "pt") as f:
        sd = {}
        for k in f.keys():
            if k.startswith("model.vision_embedder."):
                sd[k[len("model.vision_embedder."):]] = f.get_tensor(k)
            elif k == "model.embed_vision.embedding_projection.weight":
                sd["multimodal_embedder.embedding_projection.weight"] = f.get_tensor(k)
    missing, unexpected = emb.load_state_dict(sd, strict=True), None
    print(f"[unified] loaded {len(sd)} tensors in {time.time()-t0:.1f}s")

    H = []
    H.append(emb.patch_ln1.register_forward_hook(lambda m, i, o: stash("u.ln1", o)))
    H.append(emb.patch_dense.register_forward_hook(lambda m, i, o: stash("u.dense", o)))
    H.append(emb.patch_ln2.register_forward_hook(lambda m, i, o: stash("u.ln2", o)))
    H.append(emb.pos_norm.register_forward_pre_hook(lambda m, i: stash("u.pos_added", i[0])))
    H.append(emb.pos_norm.register_forward_hook(lambda m, i, o: stash("u.pos_norm", o)))
    out = emb(pixel_values=pv, image_position_ids=pos, return_dict=True)
    stash("embed_vision", out.pooler_output)
    for h in H: h.remove()

    path = os.path.join(OUT_DIR, FIX_NAME + ".safetensors")
    save_file(captured, path)
    meta = {"model_id": MODEL_DIR, "family": "gemma4_unified", "image": os.path.basename(IMAGE_PATH),
            "image_size": list(img.size), "num_soft_tokens": n_soft, "embedder_only": True,
            "pending": "full-forward argmax/greedy fixture needs a 128 GB machine (see docstring)",
            "tensor_keys": sorted(captured)}
    json.dump(meta, open(os.path.join(OUT_DIR, FIX_NAME + "_meta.json"), "w"), indent=2)
    print(f"[unified] wrote {path} ({len(captured)} tensors, {os.path.getsize(path)/1e6:.0f} MB)")

if __name__ == "__main__":
    main()
