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
#include <vector>

namespace es {
namespace mx = mlx::core;

// q:      [numQ, headDim] bf16, query heads kv-major (numQ = numKV * 8), post qNorm+RoPE.
// rawBuf: the UNSLICED prealloc cache buffer [numKV, pitch, headDim] bf16 (kRaw rows).
// len:    live rows. kw: kNorm scale weights [headDim]. invFreq: the rotary table
//         (length >= 64; zero tail ignored). Returns O [numQ, headDim] bf16.
mx::array esDecodeAttnGlobal(const mx::array & q, const mx::array & rawBuf, int len, int pitch,
                             const mx::array & kw, float eps, const std::vector<float> & invFreq);

}  // namespace es
