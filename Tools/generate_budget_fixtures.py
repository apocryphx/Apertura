#!/usr/bin/env python3
"""
Visual-token-budget fixtures — the HF image processors (both families) at every supported
budget (70, 140, 280, 560, 1120), plus the E2B tower+pooler at 70 and 560. No language model is
loaded: the processors run alone, and the tower is a bare Gemma4VisionModel +
Gemma4MultimodalEmbedder fed the 659 vision tensors straight from the E2B checkpoint.

Pictures: the fixture picture (640x400), a tall synthetic (300x900) and a wide one (1600x300),
drawn here and saved beside the fixture (fixtures_budget_tall.png / fixtures_budget_wide.png).

Keys in fixtures_vision_budgets.safetensors:
  {tower|unified}.b{B}.{e2b|tall|wide}.pixel_values      f32 [1, N, D]   (N = budget / budget*9, 0 = padding)
  {tower|unified}.b{B}.{e2b|tall|wide}.image_position_ids i32 [1, N, 2]  ((-1,-1) = padding)
  {tower|unified}.b{B}.{e2b|tall|wide}.num_soft_tokens    i32 [1]
  tower.b{70|560}.e2b.last_layer     bf16 [1, N, 768]      last encoder layer output (pre-pool; padding rows included)
  tower.b{70|560}.e2b.tower_out      bf16 [tokens, 768]    Gemma4VisionModel output (padding stripped)
  tower.b{70|560}.e2b.embed_vision   bf16 [tokens, 1536]   soft tokens
Env: APERTURA_MODEL (E2B source dir), APERTURA_UNIFIED_MODEL (12B source dir, processor config only).
"""
import json, os
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import save_file

E2B = os.environ.get("APERTURA_MODEL", os.path.expanduser("~/.cache/huggingface/sources/gemma-4-E2B-it-qat-q4_0-unquantized"))
U12 = os.environ.get("APERTURA_UNIFIED_MODEL", "/Volumes/Gemma 4/sources/gemma-4-12B-it-qat-q4_0-unquantized")
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aptransformerTests", "Fixtures"))
BUDGETS = (70, 140, 280, 560, 1120)
torch.set_grad_enabled(False)

from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from transformers.models.gemma4_unified.image_processing_gemma4_unified import Gemma4UnifiedImageProcessor
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4MultimodalEmbedder

def synthetic(path, w, h):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for x in range(w):
        for y in range(h):
            px[x, y] = (int(255 * y / (h - 1)), int(255 * x / (w - 1)), (x * 7 + y * 3) % 256)
    d = ImageDraw.Draw(img)
    d.ellipse([w // 8, h // 8, w // 2, h // 2], fill=(240, 200, 20), outline=(0, 0, 0), width=3)
    d.rectangle([w // 2, h // 2, w - w // 8, h - h // 8], fill=(20, 120, 240), outline=(255, 255, 255), width=3)
    img.save(path, format="PNG")
    return img

captured = {}
def stash(k, t): captured[k] = t.detach().cpu().contiguous()

def main():
    pics = {"e2b": Image.open(os.path.join(OUT, "fixtures_vision_e2b.png")).convert("RGB"),
            "tall": synthetic(os.path.join(OUT, "fixtures_budget_tall.png"), 300, 900),
            "wide": synthetic(os.path.join(OUT, "fixtures_budget_wide.png"), 1600, 300)}
    pc_t = json.load(open(os.path.join(E2B, "processor_config.json")))["image_processor"]
    pc_u = json.load(open(os.path.join(U12, "processor_config.json")))["image_processor"]
    counts = {}
    for fam, cls, pc in (("tower", Gemma4ImageProcessor, pc_t), ("unified", Gemma4UnifiedImageProcessor, pc_u)):
        for b in BUDGETS:
            kw = {k: v for k, v in pc.items() if k != "image_processor_type"}
            kw["max_soft_tokens"] = b
            proc = cls(**kw)
            for name, img in pics.items():
                f = proc(images=[img], return_tensors="pt")
                n = int(f["num_soft_tokens_per_image"][0])
                p = f"{fam}.b{b}.{name}."
                stash(p + "pixel_values", f["pixel_values"].float())
                stash(p + "image_position_ids", f["image_position_ids"].to(torch.int32))
                stash(p + "num_soft_tokens", torch.tensor([n], dtype=torch.int32))
                counts[p[:-1]] = n
                print(f"{p[:-1]:28s} pixel_values {tuple(f['pixel_values'].shape)}  soft tokens {n}")

    # Tower + pooler at 70 and 560 (E2B), bare modules fed the checkpoint's vision tensors.
    cfg = Gemma4Config.from_pretrained(E2B)
    tower = Gemma4VisionModel(cfg.vision_config).to(torch.bfloat16).eval()
    emb = Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config).to(torch.bfloat16).eval()
    with safe_open(os.path.join(E2B, "model.safetensors"), "pt") as f:
        sd_t = {k[len("model.vision_tower."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.vision_tower.")}
        sd_e = {k[len("model.embed_vision."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.embed_vision.")}
    tower.load_state_dict(sd_t, strict=True); emb.load_state_dict(sd_e, strict=True)
    for b in (70, 560):
        kw = {k: v for k, v in pc_t.items() if k != "image_processor_type"}; kw["max_soft_tokens"] = b
        f = Gemma4ImageProcessor(**kw)(images=[pics["e2b"]], return_tensors="pt")
        h = tower.encoder.layers[-1].register_forward_hook(lambda m, i, o: stash(f"tower.b{b}.e2b.last_layer", o))
        out = tower(pixel_values=f["pixel_values"], pixel_position_ids=f["image_position_ids"], return_dict=True)
        h.remove()
        soft = emb(out.last_hidden_state)
        stash(f"tower.b{b}.e2b.tower_out", out.last_hidden_state)
        stash(f"tower.b{b}.e2b.embed_vision", soft)
        print(f"tower.b{b}.e2b: tower_out {tuple(out.last_hidden_state.shape)} embed {tuple(soft.shape)}")

    path = os.path.join(OUT, "fixtures_vision_budgets.safetensors")
    save_file(captured, path)
    json.dump({"budgets": BUDGETS, "pictures": {k: list(v.size) for k, v in pics.items()}, "soft_tokens": counts,
               "tensor_keys": sorted(captured)}, open(os.path.join(OUT, "fixtures_vision_budgets_meta.json"), "w"), indent=2)
    print(f"wrote {path} ({len(captured)} tensors, {os.path.getsize(path)/1e6:.0f} MB)")

if __name__ == "__main__":
    main()
