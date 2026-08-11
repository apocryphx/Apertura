# Fused attention for wide head dims (256 / 512)

Status: **proposal**. The prerequisite experiment is done and negative — see "What was
already tried" below. Do not start by instantiating the existing kernel; that has been
measured and it is slower.

## Why this matters

Apertura runs `head_dim` 256 on sliding layers and 512 on global layers. MLX's fused
full-attention kernel instantiates only `{64, 80, 96, 128}`, so `use_fallback` returns true
and **all prefill attention runs unfused** — the full `[heads, seqQ, seqK]` score matrix is
materialised, written and re-read across QK^T, mask, softmax and AV.

Measured on the unfused path, per-token prefill cost fits `t(L) = 4.46 + 0.00119·L` ms
almost exactly, splitting as:

| context | linear term (MLP etc.) | quadratic term (attention) |
|---|---|---|
| 512 | 88% | **12%** |
| 1024 | 79% | **21%** |
| 2048 | 65% | **35%** |
| 4096 | 48% | **52%** |

The quadratic share grows linearly with context, on a model that supports 256K. This is the
largest unclaimed performance item in the stack.

## What was already tried (2026-08-11) — and why it failed

`head_dim` is already a template parameter (`bd`) in `steel_attention.h`, so the obvious move
is to add instantiations. That was done, with routing, and it **works but is 17.7% slower**.

Order-balanced benchmark, n=4 per arm, q4 bundle, prefill 2048:

| | prefill | decode | within-arm spread |
|---|---|---|---|
| stock (MLX fallback) | **158.5** tok/s | 18.80 | 7.4% |
| fused bd256/bd512 | 130.5 tok/s | 18.93 | 6.7% |

Correctness was fine — argmax and greedy match, conformance passes, pipelines fit. The
problem is tile shape, and the constraints that force it are all `static_assert`ed:

```
BQ == kNWarps * kFragSize  and  TQ == 1     ->  BQ is pinned to 8 * WM * WN
TK = BK / kFragSize >= 1                    ->  BK >= 8
Q_smem[BQ*BD] + KV_smem <= 32 KB            ->  see below
```

MLX's own tuning shrinks BK as BD grows (32 at bd64, 16 at bd128). At bd256 BK must fall to
**8, the hard minimum** — many K iterations, poor reuse — while the fallback being replaced
runs heavily-tuned steel GEMMs. The bandwidth saved on the score matrix does not pay for the
GEMM efficiency lost.

Threadgroup memory is the binding constraint at bd512: WM=4 forces BQ=32 and 32 KB for `Q_smem`
alone; WM=2/BQ=16 overshoots by **exactly 256 bytes** (33024 vs 32768); only WM=1/BQ=8 fits,
i.e. 32 threads per group.

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
1 in every instantiation. Nothing currently partitions BD.

## Proposal A (primary): partition head_dim across warps

Give `WN` its natural meaning — warp columns over the head dimension.

- Warp `(m, n)` owns Q rows `m` and head-dim slice `n`, i.e. `Otile` becomes `TQ × (TD / WN)`.
- **AV needs no cross-warp communication**: `O[q,d] = Σ_k P[q,k]·V[k,d]` contracts over K, and
  each warp owns a disjoint `d` range. It needs all of `P[q,:]`, which is shared.
- **QK^T does**: `S[q,k] = Σ_d Q[q,d]·K[k,d]` contracts over the split axis, so each warp
  produces a partial score tile that must be summed across the `WN` warps before softmax.

Cost: one threadgroup reduction of a `BQ × BK` float tile per K block, plus a barrier.
At BQ=32, BK=32 that is 4 KB reduced across 4 warps — small against the MMA work it enables.

Benefit at bd512, WN=4: `Otile` drops from 64 to 16 fragments per warp — back to bd128
pressure — which frees the registers that currently force `BK = 8`. **BK can return to 32**,
restoring the K reuse that makes the fused form win in the first place.

Sketch:

```
for each K block:
    each warp computes partial S over its BD slice          (MMA, no change)
    threadgroup-reduce partial S across WN warps            (NEW: barrier + 4KB reduce)
    all warps read full S, apply mask/softcap, online softmax update
    each warp accumulates its own O slice from P and its V slice   (MMA, no change)
```

`Q_smem` and `KV_smem` are unchanged in size and still shared, so threadgroup memory does not
grow; the freed budget goes to a larger BK.

## Proposal B (complementary): split-K across threadgroups

FlashDecoding-style, and MLX already has the machinery in `sdpa_vector_2pass`. Partition the K
axis across threadgroups; each emits partial `(m, l, O)`; a second pass combines them with the
standard online-softmax merge. Does not address register pressure, so it does not replace A —
but it adds parallelism at long context, which is exactly where the quadratic term dominates.
Worth doing after A, guarded on `kL` above some threshold.

## Proposal C (fallback): stream head_dim in chunks

Process BD in chunks of 128 with `O` accumulated in threadgroup memory rather than registers.
Simpler than A and needs no cross-warp reduction, but trades register pressure for threadgroup
traffic on every K block. Expect it to be slower than A; keep it only if A's reduction proves
more expensive than modelled.

## Validation plan

The infrastructure exists as of 2026-08-11; none of this needs building.

1. **Pipeline fit** — `scratchpad/pipe_probe_attn.m` reports `maxTotalThreadsPerThreadgroup`
   per specialised kernel. Compiling is not launching; check this before anything else.
2. **Correctness** — `AperturaResearch --ulp --probe-all --fused`. The `attn_oproj` gate added
   in 47bfb40 measures attention output directly rather than inferring it from `layer_out`.
   Expect fused to sit near the unfused figures (98.8-100% within 1 ULP) if the rework is
   numerically faithful; today's fused path sits at 38-81%.
3. **Performance** — order-balanced interleaved benchmark, both orderings, per-rep values
   reported. A gap that does not flip sign with ordering is real; one that does is thermal
   position. Blocked runs on this machine give the first-measured arm a 12-20% advantage.
4. **End to end** — `testSessionEndToEnd` with `APERTURAKIT_TEST_MODEL`, plus the 1408-context
   PyTorch oracle (`--longctx --fused`).

## Risks and unknowns

- **The cross-warp reduction may cost more than modelled.** It is per K block, so its relative
  cost falls as BK grows — but that is an argument, not a measurement. Prototype A at bd256
  first, where the register relief is 2x rather than 4x and the answer arrives sooner.
- **This is MLX's source, not Apertura's.** It means carrying a fork patch in the file upstream
  changes most often (four SDPA commits in the 187 between the July and August pins), or
  landing it upstream. ml-explore/mlx#3885 declined head_dim expansion without evidence; a
  working kernel plus these benchmarks is the evidence that was missing.
- **NAX.** `is_nax_available()` requires GPU gen >= 17. M4 Max is `applegpu_g16s` (gen 16), so
  the non-NAX `steel_attention` is live here. On M5 the NAX variant takes over and needs the
  same work separately — verify which kernel is live before benchmarking on new hardware.
- **head_dim 512 is the harder half** and only 10 of 60 layers. If A works at 256 and not at
  512, shipping 256-only is still 5/6 of the layers.

## Starting point

A working (if slow) instantiation with routing, constraints solved and correctness verified,
is preserved in the session worktree `scratchpad/mlx_attn`:

- `steel_attention.metal` — bd256 (BQ32/BK8/WM4) and bd512 (BQ8/BK8/WM1) instantiations
- `scaled_dot_product_attention.cpp` — `wm`/`bq`/`bk` selection per head_dim, and
  `sdpa_full_supported_head_dim` extended to 256/512

Begin from there rather than from scratch; the constraint-solving is the part that took the
longest and it is already done.
