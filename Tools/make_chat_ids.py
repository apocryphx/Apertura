#!/usr/bin/env python3
"""
Build Gemma-4 chat-formatted input_ids from a text file (tokenizer only — no model load).
Wraps the file content as a single user turn with the generation prompt appended, so the
instruction-tuned model RESPONDS rather than continuing the document. Apertura then generates.

Usage: make_chat_ids.py <textfile> <out.safetensors>
"""
import sys, torch
from transformers import AutoTokenizer
from safetensors.torch import save_file

tok = AutoTokenizer.from_pretrained("google/gemma-4-31b-it")
text = open(sys.argv[1]).read()
prompt = tok.apply_chat_template(
    [{"role": "user", "content": text}],
    add_generation_prompt=True,
    tokenize=False,
)
# The template already embeds <bos> and the turn markers as text -> don't add specials again.
ids = tok(prompt, add_special_tokens=False)["input_ids"]
save_file({"input_ids": torch.tensor(ids, dtype=torch.int32)}, sys.argv[2])
print(f"chat input_ids: {len(ids)} tokens")
print("head:", ids[:8], tok.decode(ids[:8]))
print("tail:", ids[-8:], repr(tok.decode(ids[-8:])))
