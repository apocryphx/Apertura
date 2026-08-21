#pragma once
//  ESDecodeAttn — fused decode attention for the GLOBAL (d=512, GQA 8:1) layers over a
//  raw-K cache (config.rawKV). One custom Metal kernel (mx::fast::metal_kernel — JIT,
//  no MLX fork) streams each cached kRaw row ONCE and recomputes in registers:
//      rinv = rsqrt(mean(kraw^2) + eps)      (kNorm and vNorm share it — same input)
//      K    = rope(kraw * rinv * kw)         (p-RoPE pairs (d, d+256) for d < 64)
//      V    = kraw * rinv                    (weightless vNorm; rinv folds into the
//                                             softmax weight, V never materializes)
//  then runs online-softmax attention for the 8 query heads of its KV head.
//  Flash-decoding: S sequence splits in the kernel, (m, l, acc) partials combined with
//  ops. Halves the depth-growing decode read (no V stream) at time parity with the
//  composite at 60K context (probe: 2.30 vs 2.27 ms; ~1e-3 drift vs the ops reference).
//
//  Gemma-4-global shapes are hardcoded (headDim 512, 8 query heads per KV head); the
//  wrapper asserts them.
#include "mlx/mlx.h"
#include <array>
#include <vector>

namespace es {
namespace mx = mlx::core;

// q:      [numQ, headDim] bf16, query heads kv-major (numQ = numKV * 8), post qNorm+RoPE.
// rawBuf: the UNSLICED prealloc cache buffer [numKV, pitch, headDim] bf16 (kRaw rows).
// len:    live rows. kw: kNorm scale weights [headDim]. invFreq: the rotary table
//         (length >= 64; zero tail ignored). Returns O [numQ, headDim] bf16.
mx::array esDecodeAttnGlobal(const mx::array & q, const mx::array & rawBuf, int len, int pitch,
                             const mx::array & kw, float eps, const std::vector<float> & invFreq);

// q8 variant (config.rawKVQ8): the cache rows are per-row affine u8 — qBuf [numKV, pitch,
// headDim] u8 plus scale/bias [numKV, pitch, 1] f32 (unsliced prealloc buffers). The kernel
// dequantizes in registers at every use (stage rms, K reconstruction, V accumulation);
// device traffic per row drops to 512 B + 8 B. Same splits, same fused combine.
mx::array esDecodeAttnGlobalQ8(const mx::array & q, const mx::array & qBuf,
                               const mx::array & scBuf, const mx::array & bsBuf,
                               int len, int pitch, const mx::array & kw, float eps,
                               const std::vector<float> & invFreq);

// Per-row affine u8 quantizer as ONE fused dispatch: kNew [kvH, n, hd] bf16 ->
// {q u8 [kvH, n, hd], scale f32 [kvH, n, 1], bias f32 [kvH, n, 1]}. The op-level
// version (~6 MLX dispatches) costs ~3 ms/token at decode across the 10 global
// layers — pure dispatch overhead on a [4, 1, 512] row. One simdgroup per row.
std::array<mx::array, 3> esQuantizeRawRows(const mx::array & kNew);

}  // namespace es
