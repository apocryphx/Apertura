#pragma once
//  ESVisionPooler — Gemma4VisionPooler: k x k average pooling by patch position, then the
//  sqrt(hidden) scale in float32.
//
//  Each pooled token gathers the patches whose (x/k, y/k) coincide: kernel index =
//  x/k + (maxX/k) * (y/k), realised as a one-hot [n, tokens] weight matrix /k^2 applied in
//  float32 (weights.T @ h.float()) — the reference then casts the pool back to the activation
//  dtype BEFORE scaling by sqrt(hidden) in float32, and returns float32. Standardization
//  (std_bias/std_scale) and the final cast back to the working dtype are the vision model's.
#include "mlx/mlx.h"
#include <cmath>

namespace es {
namespace mx = mlx::core;

class ESVisionPooler {
public:
    ESVisionPooler(int hiddenSize, int kernel) : k_(kernel), rootHidden_((float) std::sqrt((double) hiddenSize)) {}

    // h: [n, hidden] (compute dtype); positionIds: int32 [n, 2]; returns float32 [tokens, hidden].
    mx::array forward(const mx::array & h, const mx::array & positionIds, int tokens) const;

private:
    int   k_;
    float rootHidden_;
};

}  // namespace es
