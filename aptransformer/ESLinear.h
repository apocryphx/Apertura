#pragma once
//  ESLinear — bias-free linear, optionally weight-quantized.
//
//   y = x @ W^T   (HF weight shape [out, in]).
//   quantBits == 0 -> bf16 matmul (canonical).
//   quantBits 4/8  -> affine group quantization at construction; forward uses
//                     mx::quantized_matmul (reads ~quantBits/16 of the bf16 bandwidth).
//  This is the bandwidth lever for DECODE on the 31B (weights dominate the per-token read):
//  measured ~2.5x at Q4/g64. Note the asymmetry — DECODE is bandwidth-bound so fewer weight
//  bytes wins big, but PREFILL is compute-bound, so quantized_matmul's dequant overhead makes
//  prefill ~11% SLOWER. 4-bit is a decode win, a small prefill cost. (See ESModelConfig.h.)
#include "mlx/mlx.h"

namespace es {
namespace mx = mlx::core;

class ESLinear {
public:
    // weight: [out, in] in computeDtype. quantBits 0 -> store as-is; 4/8 -> quantize now.
    ESLinear(mx::array weight, int quantBits, int groupSize);

    // Already-quantized: adopt packed weight + scales + biases verbatim (reload from an .apml
    // bundle — no re-quantization). bits>0 always (this ctor is the quantized path).
    ESLinear(mx::array packedWeight, mx::array scales, mx::array biases, int bits, int groupSize);

    mx::array forward(const mx::array & x) const;  // [.., in] -> [.., out]

private:
    bool      quant_;
    int       bits_, groupSize_;
    mx::array w_;            // bf16 weight (quant_ == false) OR packed quantized weight
    mx::array scales_, biases_;  // valid iff quant_
};

}  // namespace es
