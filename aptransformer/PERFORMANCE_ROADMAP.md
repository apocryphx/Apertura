# Performance Roadmap — closing the long-context gap · **GOAL MET (2026-07-21)**

**Status:** the roadmap's goal — competitive runtime performance at practical context
lengths, without giving up bit-exact conformance or readability — is **achieved**, with
**no custom Metal kernels**. Apertura is a research instrument first; the levers that got
here (P0 preallocated cache, P4 Q4-head option, P5 chunked prefill) are all conformance-
gated, and the two that trade exactness for speed (P3 compiled step, Q4 head) are opt-in.

This document now serves three purposes: the **current measured standing** (below), the
**per-lever record** (§2 — what was built, what it bought, what was ruled out and why),
and the **measurement discipline** (§6 — the traps that manufactured every "collapse" this
project ever thought it had). It complements the in-code `PERFORMANCE FINDINGS` block in
[`ESModelConfig.h`](ESModelConfig.h) and the attention-path note in
[`ESAttention.mm`](ESAttention.mm).

---

## 1. Where we stand (measured 2026-07-21, Gemma-4-31B QAT Q4, Apple M4 Max)

Same model (Apertura `.apml` Q4-g64 vs the identical GGUF `Q4_0`, both 4.5 bits/weight),
same hardware. Methodology: **cold-gated (die ≤ ~48 °C, `Tools/hidtemp.m`), fresh-process,
single-arm** (`--bench-eager` / `--bench-step`); llama.cpp via `llama-server` + API (never
`llama-bench`). D=300 decode unless noted.

| Workload | Apertura | llama.cpp (stock) | standing |
|---|---:|---:|---|
| decode 512 ctx (Q8 head default) | 22.5 tok/s | ~24 | 94% |
| decode 512 ctx (`--quant-embed 4`) | **23.3** | ~24 | **97%** |
| decode 4096 ctx (Q4 head) | **21.9** | ~22 | **99.5%** |
| prefill 512 | **206** | ~204 | parity |
| prefill 4096 | **195.6** | ~196 | **parity** |
| prefill 9870 (Isolde-length) | **179.9** (54.9 s TTFT) | — | linear scaling |

The decode curve is flat-minus-KV-bandwidth (22.5 → 21.1 from 512 → 4096 ≈ exactly the
KV-read growth term): decode is **memory-bandwidth-bound at every depth**, with MLX's
quantized GEMVs sustaining ~480-500 GB/s (~90% of peak). There is no long-context cliff.

> **Historical note (2026-07-20, SUPERSEDED — kept as a methodology lesson):** this
> document originally recorded "decode collapses to single digits @4096, prefill 112
> vs 196, ~150 s Isolde TTFT" and a "GPU ~87% busy / ~86 kernels/token vs 98% / 3.5"
> trace comparison, and ranked the levers accordingly. Nearly all of the *gap* portion
> of those numbers was measurement artifact, peeled back in three layers: same-process
> arm ordering (swaVerify), thermal state (~24% compression), and `--bench`'s
> unfused-arm pool pollution (~35% fused-row tax @4096). The real structural problems
> fixed along the way were the concat-grow cache (P0) and unwindowed quadratic prefill
> (P5). Full story in §6 and the P0/P3/P5 sections.

---

## 1b. Elastic standing — E2B / E4B (measured 2026-10-05, Apple M4 Max)

The 31B numbers above are the bandwidth-bound regime. The elastic models are the other
regime this engine has to serve: at Q4 the E2B's active weights are ~1.3 GB, so a
bandwidth-bound decode would run several hundred tok/s — the measured ~120 says the
per-token cost is **dispatch**, not bytes. That is exactly where P3 was predicted to pay
("may pay on smaller models"), and it does.

Methodology as §6: fresh process per arm, `--bench-eager` / `--bench-step`, D=300, two
repeats, die gated at ≤ 53 °C (the idle floor that day — see the ambient note in §6; a
hot day, LA 37 °C, the 48 °C gate of July never releases). Snapshot = bf16 HF checkpoint,
quantized at load (`--quant 4 --quant-embed 8`, g64); bundles produce identical weights
(`--verify-bundle`).

| Model · precision · ctx | eager decode | compiled step (P3) | Δ | prefill |
|---|---:|---:|---:|---:|
| E2B · Q4 g64 + Q8 head · 512 | 118.8–125.0 | **141.5–142.0** | **+13–19%** | ~960–1000 |
| E2B · Q4 g64 + Q8 head · 4096 | 117.4–117.8 | **137.4–138.7** | **+17%** | ~2090–2160 |
| E2B · bf16 · 512 | 70.1–70.3 | 76.1–76.2 | +8% | ~1000–1050 |
| E2B · bf16 · 4096 | 67.3–67.9 | — | — | ~2230 |
| E4B · Q4 g64 + Q8 head · 512 | 80.6–81.7 | **88.1–89.8** | **+9–10%** | ~655–665 |
| E4B · bf16 · 512 | 39.7 | — | — | ~700 |

Observations (all cold pairs):

- **Decode is flat with depth** on E2B too (125 → 118 eager, 142 → 138 step from 512 →
  4096): no long-context cliff, same shape as the 31B curve.
- **P3 gain grows as the model shrinks:** 31B ≈ 0%, E4B +9-10%, E2B +13-19%. The compiled
  step removes per-token graph rebuild + dispatch, which is the dominant cost only when the
  GEMVs are tiny. It is still the documented ε mode (see P3 addendum) — not the default.
- **Q4 head helps E2B more than the 31B:** `--quant-embed 4` 131.2 vs 124.8 (+5%, two
  repeats byte-identical) — the tied 262k×1536 head is a larger share of a small model's
  per-token bytes. Quality trade as P4 (`--head-verify`).
- **Q8 weights are a loss on E2B:** 100.8 tok/s vs 125 at Q4 and 70 at bf16 — Q8 GEMVs
  are slower than Q4 and the model is not bandwidth-bound enough for the byte saving to
  matter. Stay on Q4 g64.
- **Prefill is matmul-bound and unaffected by every lever here** (≈1000 @512, ≈2200 @4096
  on E2B, bf16 and Q4 alike). First-run numbers in a fresh process can read 10-30% low
  (MLX JIT of newly needed kernels) — repeat before believing a prefill delta.
- **Context for the deployment question:** the same E2B at the same Q4 g64 + Q8 head runs
  54 tok/s under Apple's Core AI export on this Mac (static iOS graph, 8192-ctx buckets) —
  Apertura eager is 2.3× that and P3 2.6×.

### 1c. Bundle size: the PLE table · **DONE (2026-10-05, `b949f1e`) — E2B 5.9 → 3.8 GB, free**

The `.apml` exporter kept `embed_tokens_per_layer` (2.35 B params = 50% of E2B, 4.7 GB
bf16) at full precision — 75% of the published 6.28 GB E2B bundle was that one tensor.
It is read only by a per-token row gather, so `b949f1e` quantizes it (`ple_bits`,
default 8 on export; `--quant-ple N`; the table is an `ESEmbedding`, gather+dequant,
traces into the compiled step). Measured on E2B Q4 g64 + Q8 head, bundles on disk:

| PLE table | bundle | `--verify-bundle` | `--vs-bf16` 3 probes ×48 | `--vs-bf16` 3313-tok prompt ×256 | cold decode | cold first prefill 512 |
|---|---:|---|---:|---:|---:|---:|
| bf16 (old) | 5.9 GB | PASS, Δ0 | 131/144 | 201/256 | 124.2–124.9 | 0.47 s |
| **Q8** | **3.8 GB** | PASS, Δ0 | 131/144 | 202/256 | 123.4–125.5 | 0.33 s |
| Q4 g64 | 2.7 GB | PASS, Δ0 | 129/144 | 203/256 | 122.2–125.1 | 0.27 s |

- **Quality: unchanged at Q8** — identical top-1 agreement to the bf16 table, prompt by
  prompt, on both gates; Q4 costs 2/144 on the short probes and nothing on the long
  prompt. The Q4 *layers* are the whole deviation from bf16 (91% short / 78.5% long —
  the latter is a property of the existing Q4 g64 recipe on a long analytical answer and
  worth its own look, independent of this change).
- **Speed: decode neutral** at every precision. The "first prefill" column is the first
  forward paging the weights in from disk and tracks bundle bytes — a load-latency win,
  not steady-state prefill. The opposite artifact exists in memory: `--quant-ple` on an
  HF snapshot pays MLX's lazy `quantize` of the 4.7 GB table inside the first prefill
  (~60 ms; in-memory 512-prefill reads 865 vs 980 tok/s). Bench bundles, not snapshots,
  when the table precision is the variable.
- `--step-verify` with the Q8 table: 317/317 PASS, 1.10× in-process.
- **E4B, same recipe + Q8 table:** 8.2 → 5.7 GB on disk, `--verify-bundle` Δ0,
  `--vs-bf16` long prompt 240/256 = 93.8% (the larger model tolerates the Q4 layers far
  better than E2B's 78.9% — same prompt, same recipe).
- Older bundles (no `ple_bits`) load unchanged (table bf16). Published 2026-10-05:
  `apocryphx/gemma-4-E2B-it-q4-apml` 6.28 → 4.08 GB remote,
  `apocryphx/gemma-4-E4B-it-q4-apml` 8.77 → 6.13 GB remote, both Q8 table.
- Q4 g64 PLE is the next notch if 2.7 GB matters (e.g. the 8 GB-phone deployment story);
  the gates say it is nearly free, but it was not shipped by default.
### 1d. QAT fidelity: lattice-exact export · **DONE (2026-10-09) — 31B QAT bundle now matches the checkpoint to bf16 precision**

**The problem.** Every published QAT bundle (31B/12B/26B, `*-qat-q4-apml`) was `mx::quantize`
affine g64 run over `google/*-qat-q4_0-unquantized` — a *second* quantization of weights that
already sit on a trained int4 lattice (the failure Unsloth documented for naive Q4_0 GGUFs). It
was gated only with `--verify-bundle` (round-trip against the same recipe), never against bf16.

**The lattice, measured on the 31B (not what ggml q4_0 would predict):** per 32-block along the
input dim, `w = bf16(k·d)`, codes `k ∈ [-8, 7]`, positive per-block step `d` = the QAT-learned
scale — *not* `absmax/-8` (that convention fits only 62.6% of weights) and not `absmax/7`. The
extreme code present is 8 in ~60% of blocks, 7 in ~38%, lower in the rest, so `d` has to be
recovered from the lattice structure (`quantizeQ4Lattice`, BUNDLE.md). Written as ordinary
MLX affine g32 with `scale = d`, `bias = -8d`: no loader or kernel change.

**Weight-level gate (`--verify-lattice`, bundle dequantized vs source, all 30.70 B quantized weights):**

| recipe | bundle | bit-exact | within 1 bf16 ulp | max \|err\| |
|---|---:|---:|---:|---:|
| affine g64 Q4 + Q8 head (published recipe) | 17 GB | 28.32% | 33.38% | 1.66e-2 |
| **lattice g32 Q4 + exact Q4 head** | **18 GB** | **90.67%** | **100.0000%** | 3.9e-3 (= 1 ulp of the largest embed weights) |

The 9.3% that are not bit-identical are one bf16 rounding step away: the checkpoint stores
bf16 *roundings* of `k·d`, so no single bf16 step reproduces every element (a float32 LS step
reaches only 96.9%) — the bundle and the checkpoint are two equally faithful bf16 renderings
of the same trained lattice point. This is the ceiling for a 4-bit affine format with bf16 scales.

**Forward gate (`--vs-bf16`, teacher-forced top-1 vs the bf16 QAT reference, same machine):**

| recipe | 3 probes ×48 | 2176-tok War-and-Peace prompt ×256 |
|---|---:|---:|
| affine g64 (published) | 57/60 = 95.0% ("The capital of France is" → 1/2) | 243/256 = 94.9% |
| **lattice g32** | **60/60 = 100%** | **255/256 = 99.6%** |

- The exact head: `embed_tokens` is on the lattice too, so the tied head is stored exact at Q4
  (1.41 B weights, 100% within 1 ulp) — smaller *and* more faithful than the Q8 affine head
  (58% within 1 ulp). `--quant-embed` on this bundle would re-quantize it lossily; leave it off.
- **Cost, re-measured 2026-10-09 (cold-gated ≤49 °C at a 47 °C idle, fresh process per arm,
  `--bench-eager --fused`, D=300, interleaved g32/g64, 2-3 repeats; one g32 arm discarded —
  a local AI job was running concurrently and halved it, 9.1 tok/s at 87 °C):**

  | arm | decode @512 | decode @4096 | prefill @4096 |
  |---|---:|---:|---:|
  | affine g64 + Q8 head (old recipe, same source) | 22.3 / 22.3 | 20.9 / 20.9 / 20.4 | 201 / 198 / 198 |
  | **lattice g32 + exact Q4 head** | **21.2 / 21.1** | **19.8 / 19.7** | 199 / 197 |
  | penalty | **−5.2%** | **−4.6%** | parity |

  So the bundle costs ~5% decode, not the ~23% §3 recorded for g32 in July. Two effects
  net out: g32 doubles the scale/bias bytes the GEMVs read (+1 GB on disk, ≈ −8-9%
  decode on its own), and the exact Q4 head reads 0.79 GB instead of the Q8 head's 1.5 GB
  per token (+3.3-3.6%, P4). The July 23% was measured before P0/P3/P4 and is superseded.
- Bundle: `/Volumes/Gemma 4/gemma-4-31b-it-qat-q4-lattice.apml` (`quantization.json` carries
  `lattice: qat-int4-g32` + fit stats). The A/B bundle `…-q4-g64-affine.apml` sits beside it.
  Published: `apocryphx/gemma-4-31b-it-qat-q4-apml` was overwritten in place with the lattice
  bundle (2026-10-09); its earlier revisions are the 95% row.
- **12B and 26B-A4B (same day, same path):** both on the lattice, zero fallback tensors; the 26B's
  3-D expert tensors (22.8 B of its 25.2 B quantized weights) reconstruct 93.7% bit-exact. A/B
  against the previously published g64 bundles, same gates (26B forward gates with `--moe-sparse`
  on both sides):

  | model | recipe | bit-exact | ≤1 ulp | 3 probes | 2176-tok ×256 |
  |---|---|---:|---:|---:|---:|
  | 12B | affine g64 (published until 2026-10-09) | 27.7% | 33.9% | 65/67 | 231/256 = 90.2% |
  | 12B | **lattice g32** | **90.3%** | **100%** | **66/67** | **254/256 = 99.2%** |
  | 26B-A4B | affine g64 (published until 2026-10-09) | 17.7% | 20.5% | 67/69 | 232/256 = 90.6% |
  | 26B-A4B | **lattice g32** | **93.5%** | **100%** | **69/69** | **254/256 = 99.2%** |

  All three canonical repos (`apocryphx/gemma-4-{31b,12b,26b-a4b}-it-qat-q4-apml`) were
  overwritten in place with the lattice bundles; their earlier revisions are the g64 rows.

- **E2B / E4B re-sourced from Google's QAT checkpoints (same day).** The published bundles were
  post-training g64 over the plain `-it` release — the family that suffered most at long context
  (§1c: 78.5-91%). `google/gemma-4-{E2B,E4B}-it-qat-q4_0-unquantized` are on the same lattice,
  including the **per-layer embedding table** (2.35 B / 3.9 B entries, ~90% bit-exact), which is
  therefore stored exact at Q4 — E2B shrinks 4.0 → 2.9 GB and E4B 6.0 → 4.7 GB while gaining fidelity. The old rows
  are measured against their own source (plain bf16 `-it`):

  | model | recipe | bit-exact | ≤1 ulp | 3 probes | 2176-tok ×256 |
  |---|---|---:|---:|---:|---:|
  | E2B | plain source, affine g64 (published until 2026-10-09) | 14.2% | 33.2% | 131/144 = 91.0% | 240/256 = 93.8% |
  | E2B | **QAT source, lattice g32** | **90.4%** | **100%** | **116/116** | **251/256 = 98.0%** |
  | E4B | plain source, affine g64 (published until 2026-10-09) | 11.7% | 27.2% | 120/130 = 92.3% | 222/256 = 86.7% |
  | E4B | **QAT source, lattice g32** | **90.4%** | **100%** | **115/116** | **251/256 = 98.0%** |

  Engine change needed: the QAT elastic checkpoints omit the never-used `k_proj`/`v_proj`/`k_norm`
  on the 20 shared-KV layers (the plain release ships them as dead weights). `ESAttention` now
  installs weightless placeholders when a shared-KV layer lacks them; the plain-E2B PyTorch-fixture
  conformance gate is unchanged (14/14 numeric gates, argmax match). `apocryphx/gemma-4-{E2B,E4B}-it-q4-apml`
  overwritten in place (base model now the QAT checkpoint; cards say so).

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
> **Measured (fresh-process COLD pairs, both arms started at ~47-49 °C die, fused, D=300):**
> decode **21.1 vs 19.5 tok/s @512 (+8%)** and **15.7 vs 15.1 @4096 (+4%)** — consistent with
> the isolated append arithmetic. (An earlier same-day "+42% @1030" pair was mostly a THERMAL
> ORDERING artifact — its legacy arm ran on a hotter die than its prealloc arm; one bench run
> swings the die 47→84 °C and compresses decode ~24%. See §6.) The end-to-end win is modest
> because decode overlaps append copies with its GPU-idle time; the robust gains are
> structural: the no-eviction concat pathology is gone, and long decode runs no longer poison
> the buffer pool for everything after them in-process.
>
> **Unblocks P3:** cache state is now fixed-capacity + `slice_update` (static shapes,
> functional-izable) — the stateful concat cache was P3's stated blocker.
>
> **Variant coverage (2026-07-21 eve):** `--cache-verify` also PASSES on the other model
> families — **26B-A4B MoE (sparse, bf16): 522/522**, **E2B elastic (window 512, 20
> shared-KV layers, PLE, bf16): 522/522** — same P=1030/D=520 protocol incl. eviction,
> compaction, and the mid-decode turn append. The default is gated on dense-Q4, MoE-bf16,
> and elastic-bf16 alike.
>
> Residual: even with P0+P1, decode picks up ~16 ms/token going 512→4096 iso-thermal
> (47.4→63.7 ms) that KV-byte math (~+3 ms) cannot explain, in BOTH cache modes — the
> "GPU ~45%-busy at depth" CPU-serialization signature (per-token eager graph rebuild +
> dispatch; corroborated thermally: the die only reaches ~69 °C during 4096-ctx decode vs
> ~84 °C at 512). That is now the dominant long-context decode cost and is exactly P3's
> target.

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

### P3 — Whole-step compiled decode · **PROTOTYPED (2026-07-21) — measured ≈ NEUTRAL, and it exposed the real story** · **ELASTIC PORT (2026-10-05) — +13-19% on E2B, +9-10% on E4B**

> **Addendum (2026-10-05): the step now covers the elastic family** (commit `cad9561`).
> `ESCompiledStep` threw on PLE models; three changes lift that: (1) `forwardStep` builds
> the per-layer inputs on device from the int32 token-id array (gather + projection + norm,
> static shapes, so they trace into the compiled graph) and threads the shared-KV scratch as
> `forward()` does; (2) `ESKVCache` step mode keyed the scatter index on `maxKeep > 0`, which
> misfiles a *storing* sliding layer (it appends with `maxKeep == 0` to keep full length for
> the shared layers) as global — `setStepLayerTypes` installs the per-layer type and
> `update()` keys on it (empty → old rule, so dense is unchanged by construction); (3) the
> step tracks slot-owning layers (`owned_` = all non-kv-shared) and adopts/feeds/compacts/
> grows only those; shared-KV layers own no slot and read the storing layer's full-capacity
> buffer through `ESSharedKV` inside the traced forward, which the additive mask already
> covers (same capacity).
>
> **Gates (`--step-verify`, P=512, D=300+16 warm):** E2B Q4 **317/317 PASS**, E4B Q4
> **317/317 PASS**, E2B bf16 312/317 @512 (one near-tie flip that resynced), 317/317 @1030,
> 217/217 @8. `--step-lockstep` E2B bf16: mean |Δlogit| **0.129**, max **0.922**, **1/300**
> argmax flips — below the 31B's 0.31 / 2.5 / 0.5% above, i.e. inside the documented ε.
> Dense not re-gated (no dense snapshot on the machine); unchanged by construction.
>
> **Cold pairs (§1b):** E2B Q4 decode 125 → 142 @512, 118 → 138 @4096; E4B Q4 81 → 89.
> The prediction in the original verdict — "may pay on smaller models (dispatch-bound
> regimes)" — holds, and the gain scales inversely with model size.

> **Implemented as an opt-in prototype (`ESCompiledStep`, `--bench-step` / `--step-verify` /
> `--step-lockstep`; default paths untouched).** The entire per-token step — embed, RoPE
> (computed on-device from the position input), both masks (computed on-device over slot
> indices), 60 layers with `mx::scatter` cache appends at a position ARRAY, LM head + softcap —
> is recorded once by `mx::compile` and replayed; per-token host work is three int32 [1] uploads
> + the argmax readback. Fixed-capacity slot buffers (sliding: window+256, compaction every 256
> tokens, no re-trace; global: 1024-chunks, one re-trace per chunk).
>
> **Correctness: ε-equivalent, NOT bit-exact.** Token-exact at depth in gates (537/537 @1030 ctx
> incl. compactions) but the fixed-capacity kernels reduce in a different order than
> length-exact ones, and 60 bf16 layers amplify: lockstep numerics (forced identical stream)
> measure mean |Δlogit| 0.31 / max 2.5 with **0.5% argmax flips at shallow ctx** (1/200 @P=8;
> 0/537 @P=1030). Inherent to fixed-shape compiled attention vs a length-exact reference —
> so this can never be the conformance-gated default; it is a documented ε speed mode.
>
> **Performance: ≈ ZERO gain over CLEAN eager decode** (cold, structure-matched pairs, D=300):
> step 22.2 vs eager 22.5 tok/s @512; **21.1 vs 21.1 @4096**. Building the clean comparison arm
> (`--bench-eager`) revealed why: the "~16 ms/token depth residual / GPU half-idle at 4K" that
> motivated P3 was yet another measurement artifact — `--bench` always runs its UNFUSED arm
> first in-process, and its pool pollution taxes the fused arm ~5% @512 and **~35% @4096**
> (15.6 tok/s polluted vs 21.1 clean). Clean eager decode after P0 is simply bandwidth-bound at
> every depth: 22.5→21.1 from 512→4096 is exactly the KV-read growth term (+2.9 ms/token
> measured, +3 predicted). There is no CPU-serialization residual to recover.
>
> **Corrected standing vs llama.cpp (clean, cold, fresh-process):** decode 22.5 vs ~24 @512
> (−6%), 21.1 vs ~22 @4096 (−4%); prefill 206 vs ~204 @512 (parity), 184 vs ~196 @4096 (−6%).
> The historic "long-context collapse" was measurement methodology end to end (llama.cpp was
> always benched in its own clean server process). Keep the prototype: it may pay on smaller
> models (dispatch-bound regimes), stacked with on-device sampling, or as the base for a
> compiled chunked prefill.
>
> Original design notes below.

### P3 (original notes) — Reduce per-token dispatch (full-layer kernel fusion) · short-context decode lever

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

### P4 — q4 matmul kernel parity + optional Q4 head · **DONE (2026-07-21) — kernel half DEAD by measurement, Q4 head LANDED**

> **(a) Custom q4 GEMV kernel: NOT WORTH IT.** Batched-eval microbench of every decode-shape
> `quantized_matmul` exactly as ESLinear issues them: MLX's qmv sustains **~480-500 GB/s at
> DRAM-bound shapes** (65 MB MLP, 1.5 GB head) — equal to the bf16 matmul ceiling (477) and
> ~90% of the M4 Max's ~546 GB/s peak. (Small per-layer shapes read 580-745 GB/s in the bench —
> SLC cache inflation from same-weight batching, not real DRAM headroom.) The premise that
> "MLX's generic quantized_matmul is a hair slower per byte than llama.cpp's q4_0 GEMV" is
> false in this MLX pin at these shapes. Custom Metal would chase ≤5% on the matmul share.
> Microbench trap for the record: eval-per-call adds ~100 µs sync latency and made small GEMVs
> read 3-6× slow — batch ~30 calls per eval for true throughput.
>
> **(b) Q4 head: LANDED as `--quant-embed 4` on bundles.** `esMakeEmbedding` re-quantizes the
> bundle's packed Q8 embedding at load when the requested bits differ (dequant→requant; Q8's
> error is tiny against a Q4 bin, so ≈ quantizing from bf16). Head GEMV bytes 1.50 → 0.79 GB
> per decode token. Cold-gated pairs (D=300): decode **22.5 → 23.3 tok/s @512 (+3.6%)**,
> **21.2 → 21.9 @4096 (+3.3%)** — exactly the microbench's 1.46 ms/token prediction. Quality
> (`--head-verify`, 500 teacher-forced steps vs the Q8 head): **top-1 agreement 99.40%**,
> mean |Δlogit| 0.89. Q6 also tested (llama.cpp's q4_0 GGUFs ship ~Q6_K heads): SAME 99.40%
> agreement but only +1.3% speed — **Q4 strictly dominates Q6 here; Q8 stays the
> quality-first default.**
>
> **Standing with `--quant-embed 4`: decode 23.3 @512 / 21.9 @4096 vs stock llama.cpp ~24 /
> ~22 → 97-99.5%. Decode is at practical parity.**
>
> Original notes below.

- **What:** the in-code note attributes the residual short-context decode gap to
  "llama.cpp's hand-written q4_0 Metal kernels + our Q8 head." MLX's generic
  `quantized_matmul` is a hair slower per byte than llama.cpp's specialized q4_0 GEMV,
  and Apertura's **Q8** LM head costs ~+4% decode bandwidth vs a Q4 head.
- **Fix options:** (a) a custom Metal q4 GEMV kernel matched to the layout; (b) expose a
  Q4-head mode for a speed/quality trade (keep Q8 as default — it holds ~95–98% top-1).
- **Impact:** small–moderate on decode at all lengths.
- **Effort:** high (custom Metal) or low (Q4 head flag). **Risk:** medium.

### P5 — Long-prefill lever · **DONE via CHUNKED PREFILL (2026-07-21) — no custom kernel needed**

> **Implemented + default ON (`ESModelConfig::prefillChunk = 512`; `--prefill-chunk 0` to
> disable, `N` to tune).** `lastLogits` runs prompts longer than the chunk through the stack
> in chunk-sized forwards (evaluated per chunk), and `ESAttention` extends the sliding-window
> trim to multi-token appends (`maxKeep = window + seq`): every dropped key is outside every
> current AND future query's window — softmax weight exactly 0 — so the kept computation is
> identical. Sliding-layer prefill attention drops from O(L²) to O(L·(window+chunk)) on 50/60
> layers, and the composite-path score/mask transients are bounded at O(chunk·ctx) instead of
> O(L²) — which also stops long prefills from polluting the buffer pool for the decode that
> follows (measured: decode-after-9.9K-prefill 18.2 → 19.9 tok/s).
>
> **Measured (cold fresh-process pairs, chunk 512):** prefill **180.3 → 195.6 tok/s @4096
> (+8.5% — llama.cpp parity, ~196)** and **139.0 → 179.9 @9870 (+29%, TTFT 71.0 s → 54.9 s)**;
> the win grows with L. Decode unchanged (21.2 both @4096).
>
> **Gates (all PASS):** `--chunk-verify` greedy-token match vs the whole-prompt forward —
> 301/301 @4096/D=300, 49/49 @2048 and @4096; the `--longctx` PyTorch-oracle fixture (1408
> tokens, crosses the window) passes chunked with the identical argmax and greedy tokens; and
> `--session-verify` stays byte-identical (16/16) with the 9.9× turn speedup intact. The
> tiled-GEMM reassociation risk (the P3 lesson) did not materialize — the trimmed keys'
> contributions are exact zeros and the gates confirm token-exactness in practice.
>
> **Variant coverage (2026-07-21 eve):** `--chunk-verify` also PASSES on **26B-A4B MoE
> (sparse, bf16, P=2560): 49/49** and **E2B elastic (window 512, PLE, P=2048): 49/49** —
> the sliding trim, per-chunk PLE inputs, and the shared-KV storing-layer exemption are
> token-exact on both. The default is gated on all three model families.
>
> The ORIGINAL P5 (fused quantized-flash / windowed SDPA Metal kernel, below) is now only
> relevant for >16K contexts where quant-KV + flash would need to coexist; the chunked
> approach covers the practical range without custom Metal.

### P5 (original notes) — Fused quantized-flash (qSDPA) Metal kernel · only pays off > ~16K ctx

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
- **g32 weight bundle (for plain post-training quantization)** — finer group = more dequant
  metadata; measured ~23% slower decode than g64 in July (pre-P0/P3/P4) for negligible quality
  gain on a non-QAT source. **Superseded for QAT sources (§1d, 2026-10-09):** the lattice-exact
  g32 bundle costs ~5% decode vs g64 affine and is exact to bf16 precision, so QAT bundles are
  g32 by construction. Plain `-it` exports stay on g64.
- **Metal fast math (measured 2026-10-05, E2B)** — rebuilt the pinned MLX with
  `-ffast-math` in the precompiled metallib *and* `MathMode::Fast` as the default for
  runtime-generated kernels, linked a second driver against it. **Speed: zero** — 16 cold
  fresh-process arms (bf16/Q4 × 512/4096 × 2 repeats), every delta inside noise (e.g. Q4
  decode 124.8 vs 125.2 @512, 118.1 vs 118.2 @4096); the one "slower fast-math prefill" per
  pair is the JIT of the new generated kernels and vanishes on repeat. Neither regime is
  ALU-bound, and MLX's kernels already pin rsqrt/sqrt/log/tanh to `metal::precise`.
  **Numerics: measurable at depth** — 4-token per-op conformance bit-identical to the safe
  build (bf16 output rounding hides fp32-ULP differences), but the 1408-token PyTorch
  oracle reads max |Δlogit| 1.250 vs 1.125 safe, and free-running greedy on a real 70-token
  prompt diverges from the safe build after ~40 tokens. An ε mode that buys nothing. Pinned
  build untouched; the fast-math library lives in `../mlx/build-fastmath` for reference.

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
2. **P0 (prealloc `slice_update` cache)** — DONE. Removed the append copy/alloc tax and the
   allocator-churn pathologies; unblocked P3.
3. ~~P2 (mask modes)~~ — REJECTED (measured net-slower; MLX array-mask kernel is fastest).
4. **P3 (whole-step compile)** — PROTOTYPED, measured ≈ neutral: clean eager decode after P0
   is already bandwidth-bound at every depth (the "depth residual" was `--bench` arm
   pollution). Kept as an opt-in ε mode; not the default (0.5% shallow-ctx argmax flips).
   **2026-10-05:** ported to the elastic family — on E2B/E4B, where decode is dispatch-bound,
   it is a real lever (+13-19% / +9-10% cold, §1b) at the same ε.
5. **P4** — DONE: custom-kernel half measured dead (MLX qmv ≈ 480-500 GB/s ≈ ceiling); Q4
   head landed (`--quant-embed 4`): decode 23.3 @512 / 21.9 @4096 (+3.3-3.6%), 99.40% top-1
   vs Q8. **Decode is at practical parity with llama.cpp (97-99.5%).**
6. **P5** — DONE via chunked prefill (default ON, chunk 512): prefill 195.6 @4096
   (**llama.cpp parity**) and 179.9 @9870 (+29%, −16 s TTFT), token-exact through every
   gate incl. the PyTorch long-context oracle. Custom windowed/qSDPA Metal deferred to
   >16K-ctx territory.
7. **Remaining:** a **persistent server process** (amortizes the ~1.5-2 s MLX JIT
   cold-start the one-shot CLI pays every run — llama.cpp precompiles at load; ESSession
   already holds the state story), and only then exotic territory (Q3/Q2 bundles, >16K
   quant-KV flash). **With P0+P4+P5 the engine is at llama.cpp parity on both axes at
   practical context lengths — the roadmap's original goal is met.**

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
  `--no-swa-cache`), one arm per process.
- **`--bench`'s fused row is NOT a clean number — use `--bench-eager` / `--bench-step`.**
  `--bench` runs the unfused arm first in-process; its pollution taxes the fused row ~5%
  @512 and **~35% @4096** (decode 15.6 vs 21.1 clean; prefill 130 vs 184). This
  single-handedly manufactured the "decode collapses with context depth" narrative and the
  "GPU half-idle at 4K" trace signature (the traced runs were multi-arm too). The clean
  eager decode curve after P0 is flat-minus-KV-bandwidth: 22.5 @512 → 21.1 @4096.
  Cross-engine comparisons must use single-arm fresh processes on BOTH sides — llama.cpp
  was always measured in its own clean server process, so every polluted Apertura number
  overstated the gap.
- **Mind thermals — they are a first-order confound, now measured.** A single `--bench`
  run swings the max die temp 47→84 °C (M4 Max), and decode @512 reads 21.1 tok/s cold vs
  16.2 hot (~24% compression). Consecutive A/B arms are NOT iso-thermal (the second arm
  starts hotter): this manufactured a fake "+42%" pair reading where the true iso-thermal
  delta was +8%. Gate every arm on a cold start (die ≤ ~48 °C) and annotate temps. **The
  gate is relative to the idle floor, not absolute:** on a 37 °C LA day (2026-10-05) the die
  idles at 50-53 °C and a 48 °C gate never releases — gate at idle + ~1-2 °C, annotate
  ambient, and treat cross-day absolutes as ±5% (today's arms start ~5 °C warmer than
  July's, which is a few % of the 24% compression over the 47→84 °C swing). Die
  temps are readable WITHOUT root via the HID sensor services (usage page 0xff00, usage 5 —
  `IOHIDEventSystemClientCreate` + temperature events; ~40-line tool), or via
  `sudo powermetrics -s gpu_power,thermal` for GPU frequency + pressure. Corroborating
  detail: after a 4096-ctx decode run the die reaches only ~69 °C vs ~84 °C at 512 — the
  GPU is literally too CPU-starved at depth to get hot (independent confirmation of the
  dispatch-serialization residual).
