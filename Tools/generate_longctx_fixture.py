#!/usr/bin/env python3
"""
Long-context conformance fixture — exercises the SLIDING WINDOW boundary (>1024 tokens).

The short "The quick brown fox" fixture never crosses the 1024-token local-attention window,
so the sliding-window mask in ESGemma4TextModel::buildMask is coded-but-unverified. This builds
a >1024-token PyTorch reference so Apertura's local (sliding) layers — which must mask everything
older than 1024 positions, unlike the global layers — can be checked against ground truth.

Input: a real long document (Isolde's persona profile), tiled if needed to comfortably exceed 1024.
Captures: input_ids, last-position logits (bf16), and a short greedy continuation. Run once.
"""
import os, json, math, torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-31b-it"
DOC_PATH = "/Users/apocryphx/Downloads/Isolde Personal Profile/Isolde Personal Profile.md"
TARGET_MIN = 1408      # comfortably past the 1024 window
N_GREEDY = 8
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aptransformerTests", "Fixtures"))

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    with open(DOC_PATH) as f:
        text = f.read()

    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"[longctx] document tokenizes to {len(ids)} tokens")
    # Tile the document until we comfortably exceed the 1024 window, then trim.
    while len(ids) < TARGET_MIN:
        ids = ids + tok("\n\n", add_special_tokens=False)["input_ids"] + ids
    ids = ids[:TARGET_MIN]
    seq = len(ids)
    print(f"[longctx] using {seq} tokens (window=1024 -> last query masks the first {seq-1024} in local layers)")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": "mps"}, attn_implementation="eager")
    model.eval()

    input_ids = torch.tensor([ids], device="mps")
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    last = out.logits[0, -1]
    argmax = int(torch.argmax(last).item())
    print(f"[longctx] argmax(last) = {argmax} -> {tok.decode([argmax])!r}")

    with torch.no_grad():
        gen = model.generate(input_ids=input_ids, max_new_tokens=N_GREEDY,
                             do_sample=False, num_beams=1, use_cache=True)
    greedy = gen[0, seq:].to(torch.int32).cpu()
    print(f"[longctx] greedy: {greedy.tolist()} -> {tok.decode(greedy.tolist())!r}")

    save_file({
        "input_ids":   torch.tensor(ids, dtype=torch.int32),
        "logits_last": last.detach().to("cpu").contiguous(),
        "greedy_tokens": greedy,
    }, os.path.join(OUT, "longctx.safetensors"))
    with open(os.path.join(OUT, "longctx_meta.json"), "w") as f:
        json.dump({"seq_len": seq, "sliding_window": 1024, "last_argmax": argmax,
                   "masked_in_local": seq - 1024, "n_greedy": N_GREEDY}, f, indent=2)
    print(f"[longctx] wrote longctx.safetensors ({seq} tokens)")

if __name__ == "__main__":
    main()
