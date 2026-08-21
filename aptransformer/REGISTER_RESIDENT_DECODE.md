# Register-resident decode kernel: session brief

Status: **not started — this is the handoff for the design session.** Written
2026-08-20 at the end of the measurement day that produced the corrected baseline in
DECODE_AT_DEPTH.md (commits 282430c, 7984e07, ce3fdbd). Read that doc's correction
section and q8 verdict first; this file adds only what the fresh session needs to
start cold.

## Objective and the numbers to beat

Make the global-layer (d512, GQA 8:1) decode kernel's hot loop register-resident so
the occupancy manager stops throttling it. Everything below is same-session cold-gated
(die ≤ 48 °C, fresh process, snapshot-restored, D=200) on the M4 Max / q4 31B bundle:

| arm @61K | ms/token | note |
|---|---|---|
| bf16 composite (baseline) | 68.25–70.4 | drifts ~2 ms with ambient across sessions |
| raw bf16 kernel (shipped v3) | +0.7–0.95 vs same-session bf16 | kernel −1.9 ms standalone, +15% in-model |
| raw q8 kernel | +2.6 vs same-session bf16 | quarter bytes buys nothing at 25% occupancy |

Success = raw arm BEATS the same-session bf16 composite arm at 61K. The bytes ceiling:
a bandwidth-bound kernel at ~220 GB/s reads 2.5 GB (bf16 raw) in ~11 ms/token (+20%
end-to-end, ~17.6 tok/s) or 0.63 GB (q8) in ~3 ms (**~48.5 ms/token total, within ~7%
of shallow — depth stops being a decode problem**). The q8 stream (config.rawKVQ8,
shipped in ce3fdbd) becomes a speed lever exactly when this redesign lands.

## The diagnosis this must answer (do not re-derive)

gpudebug counters on v3 at 61K: kernel_occupancy 25.4% vs occupancy_manager_target
40.3% — the HARDWARE caps residency on an L1-thrash heuristic; threadgroup memory is
carved from L1 and the hot loop's tile/kT staging TRAFFIC (not footprint — T=2 halved
the footprint, eviction rate 4.8→1.8, cap unmoved) trips it. Not ALU-bound (f32 30%),
not DRAM-bound (121 GB/s effective). sdpa_vector runs wide because its hot loop is
register-only. Full workflow + counter table: DECODE_AT_DEPTH.md, WIDE_HEAD addendum.

## Design space (from the review session, 2026-08-20)

Rows live in ONE simdgroup's registers (16 elems/lane); threadgroup memory only in an
epilogue combine. The row math is register-only:

- **rms**: per-lane sum of squares, `simd_sum`.
- **rope**: pairs are (d, d+256); with the lane-owns-16-dims layout the partner of
  lane l sits at lane l^16 — one `simd_shuffle_xor(kn, 16)` delivers it, no memory.
  Each lane's kn already carries its own kw[d], matching the shipped kernel's pairing.
  CAVEAT: `__is_valid_simdgroup_type` (Metal toolchain 32023 header) admits
  half/float/double/int types, NOT bfloat — shuffle the f32 `kn` values (the kernel
  already computes them in f32) or bitcast bf16 via `as_type<ushort>`.
- **cos/sin**: 64 values/row; compute in the row-owning simdgroup (precise::cos —
  ALU has headroom), or shuffle-share.

The open puzzle is GQA-8: one row feeds 8 query heads; 8 heads × 16 dims of f32
accumulators/lane = 128 registers. Candidate resolutions (unmeasured, from
DECODE_AT_DEPTH future-approaches §1):

1. 2 heads per simdgroup (32 acc regs), 4-wide row team sharing via broadcast of a
   re-read;
2. head-pair splits accepting 2× row reads — parity traffic with bf16, HALF traffic
   with the q8 stream (this is where rawKVQ8 stacks);
3. dim-split simdgroup teams with per-row cross-sg score exchange. Note: a striped
   mod-4 chunk assignment keeps rope pairs simdgroup-local (chunk c pairs with c+16;
   c+16 ≡ c mod 4), so dim-split needs cross-sg exchange only for scores, never rope.

Apple's own Metal 4 cooperative tensors are documented register-resident precisely
"to avoid the latency of writing to device or threadgroup memory" — the direction is
sanctioned; on M5 that API replaces hand-rolling (DECODE_AT_DEPTH §3).

## Validation protocol (hard rules, all measured this session)

1. **Counters from variant #1**: Tools/gpuharness.mm + gpudebug (recipe in
   DECODE_AT_DEPTH). Watch kernel_occupancy vs occupancy_manager_target — if the cap
   doesn't lift toward ~40%+, the variant hasn't removed the L1 traffic it claims to.
2. **In-model cold arms are MANDATORY per variant**: standalone gpuharness timings
   flatter occupancy-capped kernels by ~15% (v3: 2.08 standalone vs 2.40 in-model
   ms/layer) while composite ops transfer exactly. A standalone win is a hypothesis,
   not a result.
3. **Same-session deltas only**: cold-gated arms drift ~2 ms/token across sessions
   with ambient (bf16 control: 68.25 → 70.4 over one afternoon). Every claim needs a
   contemporaneous control arm. ABBA within one script run.
4. **Numerics**: `--raw-lockstep --prefill 4096 --decode 32` (add `--raw-q8` for the
   q8 stream). Classes to stay inside: bf16 raw ≤ ~3 max |dlogit| / 0 flips; q8 ≤ ~4.3
   / 1–2 of 32 (shipping fused-vs-unfused control: 4.02, 1/32).
5. **Do not re-test** (all refuted 2026-08-20, fusions shipped): S=128 vs 256 splits;
   op-level vs fused split-combine; op-level vs fused append quantizer; q8 as a speed
   lever on the current kernel; dispatch-overhead attributions priced without
   measurement (MLX dispatch cost at tens/token is negligible — two hypotheses died).

## Open attribution (cheap, do first)

A contemporaneous shallow triplet — bf16 / raw-bf16 / raw-q8 at prefill 512, one
cold-gated session — to pin the raw path's small per-token constant that cross-session
drift contaminated today. ~15 minutes, closes the last bookkeeping gap.

## Assets

- **Bundle**: `~/Documents/Github.nosync/code/APML Models/gemma-4-31b-it-qat-q4.apml`
- **KV snapshots** (durable copies, APFS-cloned 2026-08-20):
  `~/Documents/Github.nosync/code/Apertura-snapshots/` — kv60k_bf16 (5.9 GB),
  kv60k_raw (3.5 GB, bf16 raw: q8 arms lazy-migrate it on first append), kv4k/16k/32k
  bf16. Fingerprint = bundle path + P + qb=0 + qkv=0g + chunk=512 (rawKV modes are NOT
  in the fingerprint — pair file to flag manually).
- **Bench arm**: `AperturaResearch <bundle> --bench-eager --prefill <P> --decode 200
  [--raw-kv|--raw-q8] --kv-snapshot <file>` — fresh process, gate on
  `Tools/hidtemp.m` ≤ 48 °C (build: clang -fobjc-arc -framework Foundation -framework
  IOKit). Restored 61K arms run ~1 min.
- **Kernels**: aptransformer/ESDecodeAttn.mm — v3 (`kDecodeAttnSource`), q8 variant,
  fused combine, fused row quantizer. All mx::fast::metal_kernel JIT, no MLX fork.
- **Build via the WORKSPACE**: `xcodebuild -workspace Apertura.xcworkspace -scheme
  AperturaResearch -configuration Debug build`. The workspace includes the
  ObjCTokenizer project; `-project Apertura.xcodeproj` links only while a
  workspace-built ObjCTokenizer.framework already sits in the shared products dir —
  it fails on a clean DerivedData.
