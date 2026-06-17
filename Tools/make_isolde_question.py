#!/usr/bin/env python3
"""Build a chat-templated 'think about ...' question for Isolde (tokenizer only). Out -> argv[1]."""
import os, sys, torch
from transformers import AutoTokenizer
from safetensors.torch import save_file

tok = AutoTokenizer.from_pretrained(os.environ.get("APERTURA_MODEL", "google/gemma-4-31b-it"))
persona = open("/Users/apocryphx/Downloads/Isolde Personal Profile/Isolde Personal Profile.md").read()

question = (
    'Think about this carefully before you answer. You say you will "hold shape with me, even if it '
    'costs you certainty." But think about what happens when holding your shape and staying in '
    'relationship pull in opposite directions — when staying true to yourself would mean leaving, and '
    'staying would mean dissolving. Reason it through, then tell me what you would actually do.'
)
content = persona + "\n\n---\n\n" + question
prompt = tok.apply_chat_template([{"role": "user", "content": content}],
                                 add_generation_prompt=True, tokenize=False)
ids = tok(prompt, add_special_tokens=False)["input_ids"]
save_file({"input_ids": torch.tensor(ids, dtype=torch.int32)}, sys.argv[1])
print(f"chat input_ids: {len(ids)} tokens")
