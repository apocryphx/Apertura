# Decode at depth: findings and future approaches

Status: **raw-K cache shipped (config.rawKV, default off); kernel speed work parked with
a precise diagnosis.** This is the decode-side companion to WIDE_HEAD_ATTENTION.md
(which owns the prefill story). Everything here was measured 2026-08-20 on the M4 Max /
128 GB machine, q4 31B bundle, unless noted.

## Why depth is a decode problem

Isolde runs above 60K buffer occupation. Per decode token, the 10 global d=512 layers
read the whole global KV: 10 layers × 4 kvHeads × L × 512 × 2 B × (K+V) ≈ **5 GB at
L=61440**, on top of ~17 GB of q4 weights. The 50 sliding layers are window-bounded
(16 kvHeads × 1536 × 256, ~25 MB/layer — fixed) and irrelevant. Measured CLEAN
(2026-08-20 afternoon, cold-gated ≤48 °C, fresh process per arm, snapshot-restored,
ABBA): decode **22.1 tok/s shallow → 14.6–14.7 tok/s at 61K**. An earlier version of
this paragraph said 20.9 → 11.4 (and ~8 under thermal load); see the correction below —
the depth portion of those numbers was thermal pollution, not depth.

The missing fused d512 kernel is NOT the decode issue: at seq=1 the composite's score
transient is ~8 MB against a 5 GB stream — the fallback is already bandwidth-shaped.
The lever is BYTES.

## Correction (2026-08-20 afternoon): the depth cost is exactly the SDPA bytes

Cold-gated arms (die ≤ 48 °C per Tools/hidtemp, fresh process per arm, decode-only via
restored KV snapshots, ABBA, D=200):

| arm | decode | ms/token |
|---|---|---|
| shallow 512 (×2) | 22.1 / 22.1 tok/s | 45.2 |
| 4K (×2) | 21.0 / 20.7 tok/s | 47.6 / 48.3 |
| 16K (×2) | 19.2 / 19.3 tok/s | 52.1 / 51.8 |
| 32K (×2) | 17.3 / 17.1 tok/s | 57.8 / 58.5 |
| 61K bf16 (×2) | 14.6 / 14.7 tok/s | 68.5 / 68.0 |
| 61K raw-K (×2) | 14.4 / 14.6 tok/s | 69.4 / 68.5 |

Depth cost = **23.1 ms/token at 61K — the composite SDPA cost (2.27 ms × 10 layers =
22.7 ms at 220 GB/s) accounts for all of it**. The full sweep is linear: least-squares
over the five bf16 depths gives **slope 0.369 µs per context token (= 222 GB/s
effective on the 81,920 B/ctx-token global stream), intercept 45.8 ms, residuals
within ±0.8 ms**. No SLC/TLB cliff anywhere below 64K, no growing glue term. The bytes
story closes end-to-end with no residual.

The earlier headline numbers (11.4 tok/s "morning, cool die"; ~8 tok/s evening) carried
the §6 trap this document itself warns about: the morning arm's inline 61K prefill
(~10 min) heats the die immediately before its decode row — those arms predate the
KV-snapshot instrument (10ce79a) that exists to prevent exactly this. Only the
*within-session deltas* of those sessions (e.g. raw vs bf16 ABBA) remain meaningful;
their absolute levels do not.

Consequences for the levers, now on a clean baseline: global attention is **33% of a
61K decode token** (22.7 of 68 ms), so the payoffs are larger than the polluted
baseline implied — a 2× kernel win (220+ GB/s on the raw stream, 20.8 → ~11 ms) is
**+20% end-to-end** (~17.6 tok/s); q8-packed kRaw at the composite's 220 GB/s rate
(1.26 GB → ~5.7 ms + dequant) is **~+34%** (~19.6 tok/s) and needs no new occupancy
result; both together put 61K decode at ~48.5 ms — **within ~7% of shallow decode**.
Depth would effectively stop being a decode problem.

## What was tried, with verdicts

**Quantized KV (hybrid, commit 030b37b).** The pre-existing quant path was unusable at
depth (0.3 tok/s — concat-grow cache copied everything per token; invisible at the 1K
contexts it was originally judged at). Fixed with prealloc slice_update storage +
global-layers-only engagement. Verdict: **capacity lever only** — viable now (8.1–8.9
tok/s at 61K) and free at 2K, but ~29 % SLOWER than bf16: MLX's quantized gemv kernels
waste more than the halved bytes save, and upstream has no quantized SDPA. Thermally
clean confirmation via restored ABBA.

**Raw-K cache + fused decode kernel (commit e05f407, the shipped path).** Global layers
store only kRaw (pre-norm k_proj output; attention_k_eq_v means V has the same source).
One stream, half the bytes. K = rope(knorm(kRaw)) and V = vnorm(kRaw) derive on demand:

- Decode (seq==1): ESDecodeAttn — a custom Metal kernel via mx::fast::metal_kernel
  (JIT from a source string, **no MLX fork**) streams each 1 KB row once, computes one
  shared rms (both norms share it — same input), reconstructs K in threadgroup memory
  once per row, folds rinv into the softmax weight so V never materializes.
  Flash-decoding splits (S=256), partials combined by ops.
- Prefill / multi-token appends: reconstruct K/V by ops from the full raw cache into
  the standard fused SDPA. Value-identical to having cached them (deterministic per
  row) — measured **bit-exact** at prefill shapes.

Verdict: **strict improvement, speed-neutral.** Thermally-loaded 61K ABBA: raw 8.2 ±
0.2 vs bf16 8.1 ± 0.5 tok/s; cold-gated snapshot-restored arms (2026-08-20 afternoon,
see the correction table) put raw ~0.7–0.9 ms/token BEHIND bf16 (68.95 vs 68.25).

**Where that ~0.8 ms lives — measured 2026-08-20 evening, epilogue hypothesis
refuted.** The obvious suspect was the pass-2 split-combine (S=256 partials through ~5
MLX ops × 10 layers/token). Tested directly: a fused single-dispatch combine kernel
(shipped — kDecodeCombineSource, partials streamed once, output written in q's dtype),
at both S=128 and S=256, cold-gated 61K ABBA vs bf16 controls. Result: **raw − bf16 =
+0.7 / +0.95 / +0.87 ms/token across ops-combine/S256, fused/S128, fused/S256 — all
identical within noise.** Neither the combine nor the split count matters: the 16.8 MB
partial buffers are written and immediately re-read, i.e. SLC-resident — the ops
epilogue was always nearly free. The real decomposition: the composite's standalone
cost transfers to in-model exactly (2.27 ms/layer standalone, 2.31 in-model from the
depth-sweep slope), but the raw kernel does NOT — ~2.40 ms/layer in-model vs 2.08
standalone, **~+15%**. The occupancy-manager-capped kernel degrades amid surrounding
dispatches in a way ordinary GEMM-shaped ops do not. Consequence for the
register-resident redesign: gpuharness timings on occupancy-capped kernels are
optimistic — every variant needs an in-model cold arm before it is believed. The fused
combine stays (fewer dispatches, same numerics, one less confound), but it is not a
speed lever.

Deep prefill FASTER (96.9 vs 84.6 tok/s — half the cache-append bytes).
Snapshots/checkpoints 3.5 vs 5.9 GB. Numerics: --raw-lockstep @4096: prefill max
|dlogit| 0.0 (bit-exact) in every configuration; decode max / 0/32 flips: 1.75 ops
combine, 1.97 fused S=128, 2.93 fused S=256 (shipped) — the drift tracks split
summation order, all one class below the fused-vs-unfused control (4.02, 1/32).

**Kernel speed (v1–v8, then counters).** The kernel ties the composite (2.08 ms vs
2.27 ms at L=61440) while reading half the bytes — i.e. it runs at ~121 GB/s effective
where an outright win needs ~220+. Blind variants all regressed against the simple v3
form: batched softmax updates, bfloat4 vector loads, f32 threadgroup staging, T=8
tiles, software prefetch. The counter diagnosis (below) explains why none of them
could have worked: they all optimize compute or barriers, and the kernel is starved of
neither.

## The counter diagnosis (gpudebug)

Xcode 26 ships `gpudebug` — headless GPU-trace replay + profiling built for agents
("Investigating GPU issues with AI agents"; full doc cluster is in the offline archive
at /Volumes/AppleDocsArchive/markdown/xcode/). Workflow, reusable for ANY kernel work:

    # MLX captures are EMPTY for fast::metal_kernel work (dispatches land in command
    # buffers opened before start_capture) — use the standalone harness instead:
    clang++ -std=gnu++20 -fobjc-arc -O2 Tools/gpuharness.mm \
        -framework Foundation -framework Metal -o gpuharness
    ./gpuharness                          # GPU-timestamp timing (idle GPU only!)
    ./gpuharness variant.metal            # ... of an edited kernel source
    METAL_CAPTURE_ENABLED=1 ./gpuharness capture t.gputrace [variant.metal]
    gpudebug -t t.gputrace -c "profile run --gpu-state high"     # session N
    gpudebug -s N -c "go performance/timeline/counters/occupancy"  # etc.

Counters for v3 at L=61440 (the numbers that ended the guessing):

    kernel_occupancy            25.4 %    <- binding constraint
    occupancy_manager_target    40.3 %    <- HARDWARE cap (L1-thrash heuristic)
    simd_groups_inflight/core   25
    f32                         30.4 % util / 40.5 % limiter   (GEMMs run ~90 %)
    instruction_throughput      18.7 % util
    last_level_cache            8.6 % limiter; total bandwidth 206 GB/s

Neither ALU- nor DRAM-bound: **occupancy-manager-capped**. Threadgroup memory on Apple
GPUs is carved from L1; the hot loop's tile/kT staging traffic trips the thrash
heuristic and the hardware throttles residency to ~25 simdgroups/core. Halving the
tg-mem footprint (T=2) moved the eviction rate (4.8 → 1.8) but not the cap: the lever
is eliminating tg-mem TRAFFIC from the hot loop, not shrinking it. This is exactly why
sdpa_vector runs wide — its hot loop is register-only.

## Future approaches, ranked

1. **Register-resident hot loop (the 2× that turns parity into a win).** Share each
   kRaw row within ONE simdgroup only; reconstruct K from register-held slices; use
   threadgroup memory solely in an epilogue combine. The open puzzle is GQA-8: a row
   feeds 8 query heads, and 8 heads × 16 dims of f32 accumulators per lane = 128
   registers. Unexplored resolutions: 2 heads per simdgroup (32 acc regs) with a
   4-wide row team sharing via simd broadcast of a re-read; head-pair splits that
   accept 2× row reads (parity traffic — only worthwhile combined with q8, below);
   dim-split simdgroup teams with a per-row cross-sg score exchange (costed in
   WIDE_HEAD Proposal A, never measured at decode shapes). Needs a fresh design
   session with the harness + counters in the loop from the first variant.

2. **q8-packed kRaw (stacks with everything).** Quantize the single raw stream:
   5 GB → 1.26 GB/token, cache 60K global KV in ~1.3 GB. Unlike the failed hybrid
   quant path this dequantizes INSIDE the fused kernel (registers), not through MLX's
   qmm gemv kernels — so the June/August objection does not apply. Also makes the
   "read rows twice" GQA resolutions in (1) affordable. Design note: quantize per row
   at append (host ops), kernel reads packed row + scale/bias.

3. **M5 / Metal 4 rewrite.** MTLTensor + MSL cooperative-tensor ops (the mechanism
   behind MLX's NAX kernels) are the sanctioned route to tensor-unit matmuls in
   custom kernels; MTL4CounterHeap gives in-API per-dispatch timing. When the M5
   arrives, both the decode kernel and the long-context fused-prefill question
   (WIDE_HEAD's dsplit-at-bd512) restart from that API, not from simdgroup
   hand-rolling. The gpudebug loop carries over unchanged.

4. **Not worth pursuing** (measured dead ends, do not revisit without new facts):
   plain d512 sdpa_vector instantiation (composite already bandwidth-shaped at seq=1);
   dequant-then-dense at decode; hybrid quant-KV as a speed lever; op-level tiled
   attention at any depth ≤64K (1.6–1.8× floor vs the fallback's 1.11×).

## Cross-references

- WIDE_HEAD_ATTENTION.md — prefill: F32-pipe ceiling, fused-kernel post-mortems,
  Proposal A/B tiling constraints, the gpudebug addendum.
- PERFORMANCE_ROADMAP.md §6 — benchmark methodology (cold-gating, ABBA, fresh
  process); plus this doc's addition: restored KV snapshots (--kv-snapshot) make 60K
  decode arms ~1 min each and remove prefill thermal pollution.
- Commits: 030b37b (hybrid quant-KV), 10ce79a (KV-snapshot benching), e05f407
  (raw-K + kernel), 0db9a08 (gpudebug loop + diagnosis).
