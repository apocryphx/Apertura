# Performance Roadmap — closing the long-context gap

**Status:** design notes / backlog. Apertura is a research instrument first (bit-exact
conformance, inspectable layers); this document is for *if and when* competitive
runtime performance — especially at long context — becomes a goal.

It complements the in-code `PERFORMANCE FINDINGS` block in
[`ESModelConfig.h`](ESModelConfig.h) and the attention-path note in
[`ESAttention.mm`](ESAttention.mm). Those record what already works
(Q4/g64 weights, Q8 head, `fused` = `mx::fast` SDPA + `mx::compile`, `ESSession`
prefix caching). This document records what is **not yet done** and would close the
remaining gap.

---

## 1. Where we stand (measured 2026-07-20, Gemma-4-31B QAT Q4, Apple M4 Max)

Same model (Apertura `.apml` Q4-g64 vs the identical GGUF `Q4_0`, both 4.5 bits/weight),
same hardware. llama.cpp measured via **`llama-server` + API** (NOT `llama-bench`, which
under-measures decode ~1.7× for this model — use the server).

| Workload | Apertura (fused) | llama.cpp (stock, LM Studio ≡ brew server) |
|---|---:|---:|
| decode, short ctx (~32–512) | ~20 tok/s | ~24 tok/s |
| decode, 4096 ctx | **collapses (~single digits)** | ~22 tok/s |
| prefill 512 | ~208 tok/s | ~204 tok/s |
| prefill 4096 | ~112 tok/s | ~196 tok/s |
| prefill ~9.9K (Isolde prompt) | ~150 s to first token | fast |

**Read:** short context is competitive (~83–85% of the most-optimized production engine
for a readable from-scratch build — a strong result). The gap opens up **with context
length**, on both prefill and decode. That is the target of this roadmap.

### Root cause (from a Metal System Trace of both engines during decode)

| | Apertura | llama.cpp |
|---|---:|---:|
| GPU-busy during decode | ~87% | ~98% |
| GPU kernels dispatched / token | ~86 | ~3.5 |

Two independent problems, in priority order below:
1. **Long-context attention cost is not bounded** — the dominant long-context problem.
2. **Per-token dispatch overhead** — MLX's eager op-graph fires ~86 small kernels/token
   vs llama.cpp's ~3.5 fused ones, leaving the GPU ~13% idle at *all* context lengths
   (this is the ~15% short-context decode gap).

---

## 2. Optimizations, ranked

Each item: **what · why/evidence · expected impact · effort · risk**. "Risk" is
dominated by the bit-exact conformance gate — any change must keep greedy output
token-identical to the PyTorch reference (`ESConformance`).

### P1 — Sliding-window KV cache (evict local-layer keys) · **DONE (2026-07-20) — biggest long-context lever**

> **Implemented + validated.** `ESModelConfig::slidingWindowCache` (opt-in) →
> `ESKVCache::update(maxKeep)` trims sliding-layer buffers to the window on single-token
> decode; `ESAttention` slices the mask to the retained keys (`alignMask`). Prefill,
> global layers, and elastic/quant paths untouched. Verified via `--swa-verify`
> (eviction off vs on, same greedy decode):
> - bit-exact: **129/129** (prefill 2048) and **65/65** (prefill 4096) tokens identical.
> - decode speedup: **2.35×** @ 2048 ctx (7.8→18.3 tok/s), **3.44×** @ 4096 (3.3→11.4).
>   Grows with context. Residual degradation at 4096 is the 10 global layers (expected).
>
> Original design notes below.


- **What:** Gemma-4 is 5:1 local:global. Local (sliding) layers — 50 of 60 — are only
  supposed to attend the last `slidingWindow` (1024) keys. Today
  [`ESKVCache`](ESKVCache.h) **never evicts** ("Phase 1 … buffers simply grow"), and
  [`ESAttention::forwardFused`](ESAttention.mm) passes the **full** `Kfull/Vfull`
  (`seqK` = entire context) to SDPA, then a `maskSliding` from
  [`buildMask`](ESGemma4TextModel.mm) zeroes out everything beyond the window. So the
  result is correct but the flash kernel still *processes the whole growing cache* on
  5/6 of all layers.
- **Why it dominates:** at 13.5K context, local layers do ~13× the attention work they
  need. This is the primary reason decode collapses and long prefill degrades while
  llama.cpp (which uses a rotating SWA cache) holds steady.
- **Fix:** give local layers a fixed-capacity **rotating** K/V buffer of size
  `slidingWindow` (+ the current chunk); global layers keep the full cache. `seqK` for
  local layers becomes O(window), not O(context). Bit-exactness holds because keys
  outside the window are already masked to −∞ — evicting them changes nothing
  numerically (verify against `ESConformance` at >window context).
- **Impact:** large at long context (bounds 5/6 of layers); ~none at ctx ≤ window.
- **Effort:** medium. **Risk:** medium (touches cache + attention; conformance-gated).

### P2 — Stop materializing the O(seq²) attention mask · **INVESTIGATED → REJECTED (2026-07-20)**

> **Tried and dropped — it's a net LOSS.** Implemented `mx::fast SDPA` built-in modes
> (`"causal"` for global layers, no-mask for decode) behind `sdpaCausalMode` and measured
> vs the materialized array-mask baseline (bit-exact, 65/65 tokens). Result at 4096 ctx:
> - prefill **0.90×** (160.6 → 144.2 tok/s) — causal mode is SLOWER.
> - decode  **0.81×** (18.2 → 14.7 tok/s) — no-mask / causal are SLOWER.
>
> **Finding: MLX's ARRAY-MASK flash kernel is the most-optimized path for this model.**
> The `"causal"` and no-mask code paths in this MLX pin are less tuned (esp. the seqQ=1
> decode kernel), so avoiding the materialized mask trades ~390 MB of memory for a 10–20%
> speed loss — not worth it. The roadmap's original hypothesis (below) was wrong; the mask
> build was never the prefill bottleneck (the O(L²) sliding-layer *compute* is, and stock
> MLX SDPA has no windowed mode to fix that — see P5). Keep the array mask. Reverted.
>
> Original (rejected) design notes:

### P3 — Reduce per-token dispatch (full-layer kernel fusion) · short-context decode lever

- **What:** the ~86 kernels/token are MLX eager ops (every norm/proj/RoPE/softmax is its
  own dispatch). llama.cpp hand-fuses each layer into ~a few kernels → ~98% GPU-busy.
- **Evidence + what we already tried:** the `--bench-async` prototypes on branch
  `perf/async-compile-decode-prototype` — on-device sampled token (no `.item()` readback)
  and `mx::compile` on the *stateless* post-attention tail — were bit-exact but bought
  only ~4% and ~2%. They trimmed dispatch at the edges; they did **not** fuse the layer.
- **Fix (the real one):** `mx::compile` the **whole per-token decode step**, which
  requires making the KV cache **functional** (K/V arrays passed in/out of the compiled
  function) so the 60-layer graph is captured once and replayed. This is the structural
  change the tail-only prototype couldn't reach because the cache is stateful
  (`mx::concatenate`, mutated by pointer through the layers).
- **Impact:** targets the ~13% GPU-idle → up to ~a short-context-decode-parity win.
- **Effort:** high (cache re-architecture). **Risk:** high (bit-exact, invasive).

### P4 — q4 matmul kernel parity + optional Q4 head

- **What:** the in-code note attributes the residual short-context decode gap to
  "llama.cpp's hand-written q4_0 Metal kernels + our Q8 head." MLX's generic
  `quantized_matmul` is a hair slower per byte than llama.cpp's specialized q4_0 GEMV,
  and Apertura's **Q8** LM head costs ~+4% decode bandwidth vs a Q4 head.
- **Fix options:** (a) a custom Metal q4 GEMV kernel matched to the layout; (b) expose a
  Q4-head mode for a speed/quality trade (keep Q8 as default — it holds ~95–98% top-1).
- **Impact:** small–moderate on decode at all lengths.
- **Effort:** high (custom Metal) or low (Q4 head flag). **Risk:** medium.

### P5 — Fused quantized-flash (qSDPA) Metal kernel · only pays off > ~16K ctx

- **What:** stock MLX has no quantized SDPA, so `quantKVBits` forgoes flash entirely
  (see [`ESAttention.mm`](ESAttention.mm)) — making quant-KV a *capacity* lever, never a
  speed one. A fused quantized-flash kernel would let very-long contexts keep both a
  compressed KV cache **and** flash.
- **Impact:** enables > 16K contexts to stay fast; below that, P1 (windowing) matters more.
- **Effort:** very high (custom Metal). **Risk:** high. **Do last.**

---

## 3. Do NOT bother (proven dead ends this session)

- **Async / on-device-token decode** — bit-exact but ~4%; the sync barrier was never the
  bottleneck (decode was already GPU-bound). Kept on the prototype branch for reference.
- **`mx::compile` on the stateless tail only** — ~2%; must fuse the *whole* layer (P3).
- **`--quant-kv` for speed** — a capacity lever; ~2× *slower* at short/medium ctx
  (forgoes flash). Only for fitting a huge KV cache in RAM.
- **g32 weight bundle** — finer group = more dequant metadata; ~23% *slower* decode than
  g64 for negligible quality gain. Stay on **g64**.

---

## 4. Also worth noting (not a kernel issue)

- **Prefix caching (`ESSession`) is already the dominant real-world win** for long
  personas/multi-turn (a 13.5K persona: ~128 s re-prefill every turn → ~3.8 s primed
  once, 33.7×). If the workload is a fixed long system prompt (e.g. Isolde), *use
  `ESSession`* — it dwarfs every kernel lever. P1/P2 still matter for the *first* prime
  and for growing conversation length.
- **MLX JIT cold-start:** the first generation in a fresh process pays ~1.5–2 s of MLX
  kernel compilation (a 128-token one-shot reads ~16 tok/s vs ~20 warm). llama.cpp
  precompiles Metal shaders at load. A persistent Apertura server would amortize this;
  the one-shot CLI pays it every run.

---

## 5. Suggested order

1. **P1 (sliding-window KV cache)** — DONE. Unlocked long-context decode (2.35–3.44×).
2. ~~P2 (mask modes)~~ — REJECTED (measured net-slower; MLX array-mask kernel is fastest).
3. **P4 (Q4 head flag)** — cheap short-context decode win; defer the custom q4 kernel.
4. **P3 (full-layer fusion)** — the big structural decode lever; the main remaining decode gap.
5. **P5 (qSDPA / windowed-prefill kernel)** — the real long-*prefill* lever (needs a custom
   Metal kernel; stock MLX SDPA can't do windowed attention). Big effort; only if long-ctx
   prefill latency (112 vs 196 tok/s @4K) matters enough.

## 6. How to validate (don't repeat this session's measurement traps)

- **Bit-exactness first:** every change must pass `ESConformance` (greedy token-identical
  to the PyTorch reference) before any perf claim.
- **Context-match all comparisons** — decode speed depends strongly on KV depth; always
  state the context length. Short-ctx and long-ctx are different regimes.
- **Warm, decode-only rate** — discard a warmup pass (or measure a long enough run to
  amortize JIT); separate prefill from decode.
- **Benchmark llama.cpp via `llama-server` + API timings, never `llama-bench`** (it
  under-measured this model's decode ~1.7×).
- **Profile with** `xctrace record --template "Metal System Trace"` (`--attach <pid>` for
  a running `llama-server`); export `metal-gpu-intervals`, union the Compute-channel
  intervals for true GPU-busy %. Watch kernels-per-token and GPU-idle gaps.
