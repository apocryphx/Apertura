"""
Where does MLX actually diverge from the reference?

The harness reports abs/rel max|med|p99 per tensor, which tells you *how much*
but not *where*. This loads both tensor sets and localises the outliers:
which tensor, which element, what the two values were, and how concentrated
the error is (a handful of bad elements vs a broad shift).
"""
import sys, math
import numpy as np
# torch loader, not safetensors.numpy: fixtures store activations in their native bf16
# and numpy has no bf16 dtype, so the numpy loader raises
# "TypeError: data type 'bfloat16' not understood".
import torch
from safetensors.torch import load_file as _load

def load_file(path):
    return {k: v.to(torch.float32).numpy() for k, v in _load(path).items()}

ref_path, mine_path = sys.argv[1], sys.argv[2]
ref, mine = load_file(ref_path), load_file(mine_path)
common = sorted(set(ref) & set(mine))
print(f"tensors in reference: {len(ref)}   in MLX dump: {len(mine)}   comparable: {len(common)}\n")

BF16_EPS = 2.0 ** -8
rows = []
for k in common:
    a, b = ref[k].astype(np.float64), mine[k].astype(np.float64)
    # fixtures carry a leading batch dim [1, seq, ...]; MLX tensors don't
    if a.ndim == b.ndim + 1 and a.shape[0] == 1:
        a = a[0]
    if a.shape != b.shape:
        print(f"  !! shape mismatch {k}: ref{a.shape} mine{b.shape}")
        continue
    d = np.abs(a - b)
    scale = np.maximum(np.abs(a), 1e-30)
    rel = d / scale
    n = d.size
    exact = int((d == 0).sum())
    # how many elements exceed 1 / 4 / 16 bf16 ULPs
    o1 = int((rel > BF16_EPS).sum())
    o4 = int((rel > 4 * BF16_EPS).sum())
    o16 = int((rel > 16 * BF16_EPS).sum())
    idx = np.unravel_index(int(np.argmax(d)), d.shape)
    rows.append(dict(name=k, n=n, exact=exact, absmax=float(d.max()), relmax=float(rel.max()),
                     o1=o1, o4=o4, o16=o16, idx=idx,
                     rv=float(a[idx]), mv=float(b[idx])))

print("=== worst 15 tensors by absolute deviation ===")
print(f"{'tensor':<28}{'elems':>9}{'bit-exact':>11}{'>1ULP':>8}{'>16ULP':>8}{'abs max':>11}  worst element")
for r in sorted(rows, key=lambda r: -r["absmax"])[:15]:
    pct = 100.0 * r["exact"] / r["n"]
    print(f"{r['name']:<28}{r['n']:>9}{pct:>10.1f}%{r['o1']:>8}{r['o16']:>8}{r['absmax']:>11.4g}"
          f"  @{r['idx']} ref={r['rv']:+.5f} mlx={r['mv']:+.5f}")

print("\n=== error growth by depth (layer_out.N) ===")
lay = [r for r in rows if r["name"].startswith("layer_out.")]
lay.sort(key=lambda r: int(r["name"].split(".")[1]))
print(f"{'layer':>6}{'bit-exact':>11}{'>1ULP':>8}{'>16ULP':>8}{'abs max':>11}{'rel max':>11}")
for r in lay:
    if int(r["name"].split(".")[1]) % 5 == 0 or r is lay[-1]:
        pct = 100.0 * r["exact"] / r["n"]
        print(f"{r['name'].split('.')[1]:>6}{pct:>10.1f}%{r['o1']:>8}{r['o16']:>8}"
              f"{r['absmax']:>11.4g}{r['relmax']:>11.4g}")

print("\n=== concentration: is the error a few outliers or broad? ===")
for r in sorted(rows, key=lambda r: -r["absmax"])[:6]:
    frac1 = 100.0 * r["o1"] / r["n"]
    frac16 = 100.0 * r["o16"] / r["n"]
    print(f"  {r['name']:<26} {frac1:6.3f}% of elements >1 ULP, {frac16:6.3f}% >16 ULP")
