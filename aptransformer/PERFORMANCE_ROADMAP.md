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

### P0 — Preallocated `slice_update` KV storage · **DONE (2026-07-21) — kills the concat-grow append tax**

> **Implemented + validated (`ESModelConfig::preallocKVCache`, default ON; `--no-prealloc-cache`
> to A/B).** The 2026-07-21 deep profile found the cache *append* itself was an algorithmic
> flaw independent of P1: `ESKVCache` grew every layer by `mx::concatenate` each token, which
> (a) copies the whole cache per layer per token, and (b) produces monotonically growing buffer
> sizes that defeat MLX's BufferCache (its reuse window is `[size, size+2 pages)`, and a growing
> cache always requests more than it just freed) — a real Metal allocation per layer per token.
> Isolated cost (kvbench, decode shapes): **~6-7 ms/token at ALL context lengths**; the
> no-eviction variant alone is ~48 ms/token @4096 — the entire pre-P1 collapse.
>
> New design: chunk-grown (256-position) fixed-capacity buffers; appends are `mx::slice_update`
> in-place writes (buffer donation — verified ~5 µs/update); sliding eviction advances a logical
> `start` instead of trimming storage; hitting capacity compacts the live range into a fresh
> buffer (one copy per ~256 tokens, sizes repeat → BufferCache recycles). Attention consumes
> slice VIEWS — MLX's SDPA vector kernel takes strided K/V at batch 1 (+2% sliding, +9% global
> fallback — negligible), and `prepare_reshape` keeps the `[kv,seq,hd]→[1,kv,seq,hd]` reshape
> zero-copy. Isolated append cost drops to **~0.9 ms/token, context-independent**.
>
> **Bit-exact (gated via `--cache-verify`, legacy vs prealloc greedy streams):** 301/301
> (P=8/D=300, growth), 521/521 (P=1030/D=520, eviction + compaction), 522/522 (same + a
> mid-decode multi-token turn append — the ESSession transition), `--session-verify` 16/16
> byte-identical with the 9.6× per-turn speedup intact.
>
> **Measured (fresh-process pairs, fused, D=300):** decode 14.4 → **20.4 tok/s @1030 ctx
> (+42%, cool machine)**; +4-5% @512 and @4096 measured under thermal throttle (legacy's
> allocator wall is CPU-side and barely thermal-sensitive, so hot-machine ratios compress —
> pair order and thermal state matter, see §6). Long decode runs no longer degrade the
> allocator pool for everything after them.
>
> **Unblocks P3:** cache state is now fixed-capacity + `slice_update` (static shapes,
> functional-izable) — the stateful concat cache was P3's stated blocker.
>
> Residual: even with P0+P1, decode still picks up ~15 ms/token going 1030→4096 that KV-byte
> math (~+2 ms) cannot explain, in BOTH arms — the "GPU ~45%-busy at depth" CPU-serialization
> signature (per-token eager graph rebuild + dispatch). That is now the dominant long-context
> decode cost and is exactly P3's target.

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
  function) so the 60-layer graph is captured once and replayed. This was blocked by the
  stateful `mx::concatenate` cache — **P0 removed that blocker** (fixed-capacity
  `slice_update` buffers have static shapes and thread through a compiled function
  naturally). P0's residual finding makes this MORE valuable than originally scored: the
  remaining long-context decode cost (~15 ms/token @4096 beyond KV-byte math, both cache
  modes) is per-token CPU graph-rebuild/dispatch serialization — precisely what whole-step
  compile eliminates.
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
2. **P0 (prealloc `slice_update` cache)** — DONE. Removed the append copy/alloc tax
   (+42% decode @1030 ctx) and unblocked P3.
3. ~~P2 (mask modes)~~ — REJECTED (measured net-slower; MLX array-mask kernel is fastest).
4. **P3 (whole-step compile / full-layer fusion)** — NOW the main remaining decode lever at
   every context length: the residual ~15 ms/token depth cost is CPU dispatch serialization,
   and P0 made the cache compile-friendly.
5. **P4 (Q4 head flag)** — cheap short-context decode win; defer the custom q4 kernel.
6. **P5 (qSDPA / windowed-prefill kernel)** — the real long-*prefill* lever. Note the
   2026-07-21 profile: MLX's FULL flash kernel supports only head-dim {64,80,128}, so at
   prefill NO gemma-4 layer (d=256/512) uses flash — 100% composite, and sliding layers do
   the full unwindowed O(L²). A cheap first step (no custom kernel): chunked prefill with
   window-trimmed K for sliding layers — measured 2.6× less sliding-attention time @4K
   (46.7→18.1 ms/layer composite), growing with L.

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
- **Never measure an arm after a pool-polluting arm in the SAME process.** The legacy
  concat cache fills MLX's buffer pool with hundreds of odd-size buffers; everything
  measured afterwards in that process (even byte-identical code paths) reads 10-25% slow.
  This is what made P1's original "3.3→11.4 tok/s @4096" numbers artifacts — a fresh
  process measures 15.3. A/B via separate processes (`--no-prealloc-cache`,
  `--no-swa-cache`), one arm per process. Note `--bench` runs its unfused arm before the
  fused arm, so its fused absolutes are conservative (the A/B stays internally fair).
- **Mind thermals on long measurement sessions.** After ~an hour of sustained GPU load,
  GPU-bound numbers compress ~20% while CPU-bound walls (the legacy allocator churn)
  barely move — ratios taken hot understate a GPU-bound fix. Compare only same-run pairs,
  or let the machine cool.
