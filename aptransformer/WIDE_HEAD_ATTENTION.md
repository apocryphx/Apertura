# Fused attention for wide head dims (256 / 512)

Status: **proposal, deprioritized after measurement correction**. Two prerequisite
experiments are done and negative — see "What was already tried" below. Do not start by
instantiating the existing kernel; both feasible tile configurations have been measured
and both are slower. More importantly, the prize is ~5× smaller than the first draft of
this document claimed — read "Why this matters" before spending time here.

## Why this matters (corrected 2026-08-11)

Apertura runs `head_dim` 256 on sliding layers and 512 on global layers. MLX's fused
full-attention kernel instantiates only `{64, 80, 96, 128}`, so `use_fallback` returns true
and **all prefill attention runs unfused** — the full `[heads, seqQ, seqK]` score matrix is
materialised, written and re-read across QK^T, mask, softmax and AV.

**Correction.** An earlier version of this section fit per-token prefill cost as
`t(L) = 4.46 + 0.00119·L` ms and attributed the whole linear term to attention — 12% of
prefill at 512 context rising to 52% at 4096, "the largest unclaimed performance item in
the stack". That fit was measured with `--bench --fused`, whose in-process unfused-arm-first
ordering pollutes the fused row (the same instrument artifact documented in
PERFORMANCE_ROADMAP.md §6 — its "stock" 158.5 tok/s at 2048 measures 195.9 clean).
Re-measured on the clean single-arm instrument (`--bench-eager`, fresh process per run,
order-balanced, n=3):

| prefill | stock fallback (clean) |
|---|---|
| 2048 | 195.9 tok/s |
| 4096 | 187.6 tok/s |

That degradation fits a per-token slope of ~0.00022 ms per context token — an attention
share of roughly **8% at 4096**, in agreement with the July direct profile (composite
attention 2.33s of a 30s unchunked 4096 prefill). Two structural facts keep it small:
chunked prefill (P5) trims sliding-layer K/V to window+chunk, so 50 of 60 layers do
**bounded** attention work regardless of depth; the quadratic term lives entirely in the
10 global `head_dim` 512 layers. It still grows without bound on a 256K-context model —
this becomes worth claiming at very long context — but at practical depths the ceiling is
single-digit percent, which changes the cost/benefit against carrying an MLX fork patch.

## What was already tried (2026-08-11) — and why it failed

`head_dim` is already a template parameter (`bd`) in `steel_attention.h`, so the obvious move
is to add instantiations. That was done, with routing, at **both** feasible K-block widths
for bd256, and both lose to the fallback. Order-balanced Latin square, fresh process per
run, `--bench-eager`, n=3 per arm, q4 bundle:

| prefill | stock (MLX fallback) | fused bd256 BK8 + bd512 | fused bd256 BK16 + bd512 |
|---|---|---|---|
| 2048 | **195.9** tok/s | 171.2 | 172.7 |
| 4096 | **187.6** tok/s | 146.3 | 147.4 |
| 16384 | **164.2** tok/s | 88.6 | 84.9 |

Decode is flat (~19.5 tok/s) across all arms, as expected — the full kernel only runs at
`seqQ > 8`. Within-arm spread 0.8–2.3% at 2048, 4.5–10.8% at 4096.

Two separate findings:

1. **BK is not the lever.** The first experiment ran bd256 at BQ32/BK8/WM4 and blamed the
   loss on K-block reuse ("MLX's own tuning shrinks BK as BD grows; at bd256 BK must fall
   to 8"). That inference was wrong twice over: bd256/BQ32/**BK16** fits threadgroup memory
   (Q 16896 + KV 12288 = 29184 of 32768 bytes) and was simply never tested — and when
   tested, doubling BK moved prefill by **less than 1%**, inside the within-arm spread.
   The fused deficit is not K reuse.

2. **The remaining suspect is register-limited occupancy.** The pipeline probe reports
   `maxTotalThreadsPerThreadgroup = 128` for both bd256 kernels — exactly the 128 threads
   the dispatch needs, zero headroom. `Otile` at bd256 is 64 floats per lane (TD=32); the
   register budget caps how many threadgroups a core can hold, and the kernel cannot hide
   its own latency. This is the constraint Proposal A actually addresses.

3. **The d512 instantiation is a depth-scaling liability — this attribution is now
   settled.** The fused-arm deficit diverges with depth: −12% at 2048, −21% at 4096,
   **−47% at 16384** (~5.7ms/token extra at 16K). The bd256 layers cannot be the cause —
   their K is window-trimmed and stops growing at ~1.5K context — so the divergence is the
   d512 kernel (BQ8/BK8/WM1, 32 threads per group) processing ever-longer unwindowed K at
   register-crippled occupancy. Plain instantiation at d512 is dead at exactly the depths
   where a fused kernel would matter; any future d512 work starts from the WN occupancy
   rework (Proposal A) or split-K (Proposal B), not from tiles.

The 16K sweep also softened the stock curve further: per-token cost 5.11/5.33/6.09 ms at
2K/4K/16K fits a slope of ~0.00014 ms per context token — attention share ~18% at 16K,
extrapolating to roughly 40% at 50K and ~58% at 100K. (Thermal note: decode drifted
17.5→14.5 tok/s across the nine-run session; the Latin square absorbs this in the means,
but per-arm spread at 16K is 2.6–11.3%.)

Correctness was fine throughout — argmax and greedy match, conformance passes, pipelines
fit. One notable numerics observation from `--fused-lockstep` (prefill 512, 16 forced
steps): the BK16 build sits at max |dlogit| 3.5 with **0/16 argmax flips** against the
unfused reference, versus 10.5 and 6/16 flips for the stock fused path (composite fallback)
and 10.1 / 3/16 for BK8. Wider K blocks mean fewer online-softmax rescales per row, so the
fused kernel is numerically *closer* to the length-exact reference than the shipping
fallback is. The large stock-path divergence is pre-existing fused-path behavior worth its
own investigation (ESModelConfig.h says the fused path "must stay argmax bit-stable").

## Root cause, measured (2026-08-12) — supersedes the occupancy framing below

A synthetic decomposition (isolated `mx.fast` SDPA, fused vs composite, fresh process
per cell, order-balanced; branch `cliff-probe` = PR #4185 + bd512 instantiation,
harness in the session scratchpad `cliffgrid/`) localized and explained the cliff:

| isolated SDPA, seqQ=512 | ratio fused/composite |
|---|---|
| d256 (BQ32/WM4) | **0.81×** @512 (fused wins) → 1.17× @16K |
| d512 (BQ8/WM1) | 4.9× @512 → **17.4×** @16K |

**d256 is healthy.** The entire cliff is the d512 configuration, and it is a
**bandwidth story, not an occupancy story**:

1. **Fused d512 time is a pure function of K-stream traffic**: qH=16 @4096 and
   qH=32 @2048 (equal `heads × seqK`) time within 0.5% of each other (73.3 vs
   73.6 ms). At the 16K asymptote the kernel streams 68.7 GB in 758 ms —
   **~90 GB/s effective, ≈18% of peak**.
2. **BQ collapse is the traffic multiplier**: at BQ=8 every threadgroup streams the
   full K/V for only 8 query rows — 4× the traffic of BQ=32.
3. **The SLC hides this at d128 and not at d512**: a BQ ladder at bd128
   (BQ 32/16/8 via `MLX_SDPA_BQ`) costs only 1.12×/1.33× @2048 (1.17×/1.48× @8192) —
   the re-streamed K hits cache. d512 working sets (8.4→33.6 MB across the heads
   axis) overflow it, which is exactly where the grid's superlinear heads-scaling
   onsets. Once past cache, the 4× is real DRAM traffic.
4. **One-warp threadgroups can't hide DRAM latency**, which is the remaining
   ~5× between 4×-inflated-traffic-at-peak and what is measured.
5. **Refuted**: per-core residency as a standalone cause. A bd128 build with the
   threadgroup footprint inflated 15→29 KB (1 threadgroup/core instead of 2;
   branch `pad-probe`) is timing-identical to stock (1.439 vs 1.444 ms @2048).
   Register/threadgroup occupancy per se was the wrong frame.

**Consequences for the proposals:**

- The BK null result is explained: BK never touches the traffic term.
- **Proposal A as sketched does not fix this.** WN-partitioning relieves registers
  but leaves `Q_smem = BQ·BD` and the K tile unchanged, so BQ stays pinned at 8–16
  at bd512 and the 4× traffic term survives. A only helps if it is extended to
  remove Q from threadgroup memory entirely (Q direct-to-registers, each warp
  holding its BD/WN slice) so BQ can grow on the warp grid — that redesign, not
  the reduction sketch below, is the actual work item.
- **Proposal B gains standing**: split-K doesn't reduce traffic, but the deficit is
  latency-bound delivery (18% of peak), and more concurrent streams attack exactly
  that. It may be the cheaper large lever.
- Any fused d512 must be benchmarked against the SLC boundary explicitly:
  wins below it do not extrapolate above it.

## Constraints (all `static_assert`ed or budget-forced)

```
BQ == kNWarps * kFragSize  and  TQ == 1     ->  BQ is pinned to 8 * WM * WN
TK = BK / kFragSize >= 1                    ->  BK >= 8
Q_smem[BQ*(BD+pad)] + KV_smem <= 32 KB      ->  see below
```

Threadgroup memory is the binding constraint at bd512: WM=4 forces BQ=32 and 32 KB for `Q_smem`
alone; WM=2/BQ=16 overshoots by **exactly 256 bytes** (33024 vs 32768); only WM=1/BQ=8 fits,
i.e. 32 threads per group. K and V share one buffer (`Ks = Vs = KV_smem`), sized
`max((BK+pad)·BD, BK·(BD+pad))`.

## Root cause

From `steel_attention.h`:

```cpp
constexpr int TQ = BQ / (kNWarps * kFragSize);  // Q seq frags per warp
constexpr int TK = BK / kFragSize;              // KV frags (all warps load the same frags)
constexpr int TD = BD / kFragSize;              // HeadDim frags (all warps load the same frags)
MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;
```

**Parallelism is only along Q.** Every warp holds the entire head dimension, so the output
accumulator is `TQ × TD` fragments per warp and register pressure scales linearly with BD:

| BD | TD | Otile floats/lane |
|---|---|---|
| 128 | 16 | 32 |
| 256 | 32 | 64 |
| 512 | 64 | **128** |

`WN` exists in the signature but only ever multiplies into `kNWarps` to split Q further; it is
1 in every instantiation. Nothing currently partitions BD. The BK8-vs-BK16 null result says
the damage is done by this register pressure (occupancy), not by K-tile reuse.

## Proposal A (primary): partition head_dim across warps

Give `WN` its natural meaning — warp columns over the head dimension.

- Warp `(m, n)` owns Q rows `m` and head-dim slice `n`, i.e. `Otile` becomes `TQ × (TD / WN)`.
- **AV needs no cross-warp communication**: `O[q,d] = Σ_k P[q,k]·V[k,d]` contracts over K, and
  each warp owns a disjoint `d` range. It needs all of `P[q,:]`, which is shared.
- **QK^T does**: `S[q,k] = Σ_d Q[q,d]·K[k,d]` contracts over the split axis, so each warp
  produces a partial score tile that must be summed across the `WN` warps before softmax.

Cost: one threadgroup reduction of a `BQ × BK` float tile per K block, plus barriers. The
partial-S exchange needs staging memory: either a dedicated `BQ·BK·4`-byte buffer, or reuse
of `KV_smem` in the window after QK^T consumes K and before the V load overwrites it — the
aliasing already forces a barrier there, but the reduce serializes the V load behind it.
This cost is per K block and unmodelled; it is the main thing the bd256 prototype must
measure.

**The benefit is register relief and the occupancy it buys — stated per head_dim, because
they differ:**

- **bd256, WN=2** (WM=2, BQ=16, 128 threads): `Otile` drops 64 → 32 floats/lane, back to
  bd128 pressure. Threadgroup memory allows BK=32 here (Q 8448 + KV 20480 + S-staging 2048
  ≈ 31 KB) — though the BK null result above means the win, if any, must come from the
  register relief itself, not the wider K block.
- **bd512, WN=4** (WM=1, BQ=8, 128 threads): `Otile` drops 128 → 32 floats/lane, and the
  threadgroup grows 32 → 128 threads — a 4× occupancy improvement over the measured
  BQ8/BK8/WM1 configuration. **BK cannot "return to 32" at bd512**: `KV_smem` at BK=32 is
  `(32+8)·512·2 = 40960` bytes, over the entire 32 KB budget by itself, and even BK=16
  overshoots by 128 bytes at standard padding (24576 + 8320). BK stays 8 (or 16 with
  reduced padding, at the cost of the loader's 16-byte alignment). An earlier draft of this
  section claimed BK=32 at bd512; the arithmetic does not allow it.

## Proposal B (complementary): split-K across threadgroups

FlashDecoding-style, and MLX already has the machinery in `sdpa_vector_2pass`. Partition the K
axis across threadgroups; each emits partial `(m, l, O)`; a second pass combines them with the
standard online-softmax merge. Does not address register pressure, so it does not replace A —
but it adds parallelism at long context, which is exactly where the quadratic term dominates,
and the quadratic term is exactly the 10 global bd512 layers where A is weakest. If anything
here is attempted for the long-context regime, B at bd512 may matter more than A.

## Proposal C (fallback): stream head_dim in chunks

Process BD in chunks of 128 with `O` accumulated in threadgroup memory rather than registers.
Simpler than A and needs no cross-warp reduction, but trades register pressure for threadgroup
traffic on every K block. Expect it to be slower than A; keep it only if A's reduction proves
more expensive than modelled.

## Validation plan

The infrastructure exists as of 2026-08-11; none of this needs building.

1. **Pipeline fit** — `pipe_probe_attn.m` (session scratchpad) reports
   `maxTotalThreadsPerThreadgroup` per specialised kernel. Compiling is not launching;
   check this before anything else. The bd256 kernels sit at exactly 128 — any rework must
   raise this ceiling, or it has not relieved the registers it claims to.
2. **Correctness** — `AperturaResearch --fused-lockstep` (cheap, self-contained: fused vs
   unfused on a forced stream, reports max |dlogit| and argmax flips), then
   `--ulp --probe-all --fused` with the `attn_oproj` gate (47bfb40) for op-level
   attribution. The BK16 kernel's 0/16-flip baseline is the number to beat.
3. **Performance** — order-balanced interleaved benchmark, fresh process per run,
   `--bench-eager` only. **Never `--bench` for cross-arm comparisons** — its in-process
   unfused-arm-first ordering taxes the fused row (this is what corrupted the first version
   of this document's motivation table). A gap that does not flip sign with ordering is
   real; one that does is thermal position.
4. **End to end** — `testSessionEndToEnd` with `APERTURAKIT_TEST_MODEL`, plus the 1408-context
   PyTorch oracle (`--longctx --fused`).

## Risks and unknowns

- **The prize at practical context is ~8% of prefill, concentrated in 10 layers.** Carrying
  a fork patch in MLX's most-churned file (four SDPA commits between the July and August
  pins) for a single-digit win is a poor trade today. The calculus changes at very long
  context, or if the kernel lands upstream: ml-explore/mlx#3885 declined head_dim expansion
  without evidence; a working kernel plus these benchmarks is the evidence that was missing.
- **The cross-warp reduction may cost more than modelled.** It is per K block; prototype A
  at bd256 first, where the register relief is 2× and the answer arrives sooner — but note
  that a bd256-only win claims the *bounded* part of attention cost, not the growing part.
  The growing part is bd512, where A's relief is occupancy-only and BK stays pinned.
- **NAX.** `is_nax_available()` requires GPU gen >= 17 (18 for 'p'-class parts). M4 Max is
  `applegpu_g16s`, so the non-NAX `steel_attention` is live here. On M5 the NAX variant
  takes over and needs the same work separately — verify which kernel is live before
  benchmarking on new hardware.

## Starting point

The working (if slow) instantiations, routing, constraint-solving and both measured
configurations are preserved on the **`sdpa-wide-head-wip` branch of code/mlx**:

- `836de3f` — bd256 (BQ32/BK8/WM4) and bd512 (BQ8/BK8/WM1) instantiations +
  `wm`/`bq`/`bk` selection per head_dim + `sdpa_full_supported_head_dim` extended to 256/512
- `f3e28b6` — bd256 BK 8 → 16 (the null result above)

Begin from the branch rather than from scratch; the constraint-solving is the part that took
the longest and it is already done. Build with the same cmake options as the pin build and
point Apertura at it by overriding `LIBRARY_SEARCH_PATHS`, `MLX_METALLIB` and
`HEADER_SEARCH_PATHS` on the xcodebuild command line (the session worktrees under
/private/tmp are disposable; the branch is the durable copy).

## Prefill is closed on M4 (2026-08-20) — counter-level attribution

The op-level exploration (tiled/online-softmax prefill attention, `config.tiledKChunk`,
gates `--tiled-verify` / `--tiled-lockstep`) ended in a decisive negative with full
attribution. Synthetic decomposition at the global-layer shape (d512, nKV4, groups8,
seqQ 512):

| variant | @16K vs GEMM floor |
|---|---|
| bare (Q@K^T)@V | 1.00× (39.0 ms) |
| chunked GEMMs only | 1.00× |
| softmax(precise) on bf16, no f32 round-trips (≈ MLX SDPA fallback) | 1.11× |
| full f32-glue composite (Apertura unfused) | 1.42× |
| op-level tiled online-softmax (C=1024) | 1.76× — **loses; merge glue > softmax saved** |

Numerics of the tiled path were fine (forced-stream lockstep drift below the shipping
fused-vs-unfused control), it is just not faster: the shipping fallback already sits at
1.11× of a GEMM floor that Xcode GPU-capture counters show is **ALU-bound at the F32
pipe** — F32 Limiter 95.3–95.8%, F32 Utilization 89–92%, F16 pipe 0.00%, memory
limiters negligible, on both the qmm/dense-GEMM and attention-GEMM encoders (94.8% of
GPU time). Occupancy 35–40% is irrelevant when the F32 pipe is saturated.

Corollaries, all measured:
- `quantized_matmul` q4 at M=512 is within 5–8% of dense bf16; dequant-then-GEMM buys
  nothing. No matmul-strategy lever at prefill.
- fp16 GEMM is SLOWER than bf16 in MLX (9.7 vs 12.1 TFLOP/s at 512×5376×21504), so the
  idle F16 pipe is not reachable by a dtype switch.
- Compute-channel timeline (xctrace): 91.5% busy during a GEMM loop — dispatch gaps ~8.5%.

Bottom line: MLX steel GEMM at ~12–14.5 TFLOP/s ≈ 90% of the M4 Max F32-pipe rate is
the prefill ceiling; every op family (attention, dense, qmm) sits on the same wall.
Prefill speedup on M4 is closed at the op level and nearly closed at the kernel level.
The ceiling moves only with hardware (M5 NAX tensor units — where MLX already has the
kernel path, including the dsplit head-split variant this document proposed).

## Decode kernel: counter-guided diagnosis via gpudebug (2026-08-20, evening)

Apple's `gpudebug` CLI (Xcode 26, "Investigating GPU issues with AI agents") reopened
the counter loop that MTLCaptureManager+MLX had foreclosed (fast::metal_kernel work
lands in command buffers opened before start_capture — the capture is empty; a
standalone harness, Tools/gpuharness.mm, sidesteps MLX entirely: compile the kernel
source raw, METAL_CAPTURE_ENABLED=1 capture, `gpudebug ... "profile run --gpu-state
high"`, then read `performance/timeline/counters/*`. Fully headless.)

Counters for the shipping raw-K decode kernel (v3, 2.08 ms / 121 GB/s at L=61440):

    kernel_occupancy            25.4%     <- the binding constraint
    occupancy_manager_target    40.3%     <- HARDWARE cap (L1-thrash heuristic)
    simd_groups_inflight/core   25
    instruction_throughput      18.7% util / 56.2% limiter
    f32                         30.4% util / 40.5% limiter
    last_level_cache            8.6% limiter;  bandwidth 206 GB/s total

NOT ALU-bound (cf. the GEMMs' 90%+ f32), NOT DRAM-bound — occupancy-manager-capped.
Threadgroup memory is carved from L1 on Apple GPUs; the hot loop's tile/kT staging
traffic trips the manager's thrash heuristic and it throttles residency to ~25
simdgroups/core. Halving threadgroup memory (T=2) left occupancy at 26% (eviction
rate fell 4.8 -> 1.8; the cap did not move) — the static footprint is not the lever,
the L1 TRAFFIC is. sdpa_vector runs wide because its hot loop is register-only.

Next design (fresh session): register-resident hot loop — share rows within a
simdgroup only (reconstruct K per-sg from a register-held row slice), threadgroup
memory in the epilogue combine alone. The 2x to composite-parity-at-half-bytes ->
outright-win lives behind that redesign.
