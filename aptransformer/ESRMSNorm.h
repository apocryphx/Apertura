#pragma once
//  ESRMSNorm — Gemma4RMSNorm. Computes in float32, casts back to input dtype.
//
//  normed = x * rsqrt(mean(x^2, -1) + eps);  if with_scale: normed *= weight.
//  NOTE: weight is used DIRECTLY (no 1+weight); HF safetensors store it centered near 1.
//  with_scale=false (e.g. attention v_norm) => pure normalization, no weight.
#include "mlx/mlx.h"

namespace es {
namespace mx = mlx::core;

class ESRMSNorm {
public:
    ESRMSNorm(mx::array weight, float eps, bool fused = false);  // with scale
    explicit ESRMSNorm(float eps, bool fused = false);           // weightless (with_scale=false)

    mx::array forward(const mx::array & x) const;

private:
    bool      withScale_;
    mx::array weight_;  // valid iff withScale_
    float     eps_;
    bool      fused_;   // true -> mx::fast::rms_norm (f32 accum, fused Metal kernel)
};

}  // namespace es
