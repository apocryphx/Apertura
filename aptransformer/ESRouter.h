#pragma once
//  ESRouter — Gemma4TextRouter. Decides which experts each token goes to.
//
//   h = rmsnorm(x)                      (weightless)
//   h = h * scale * hidden^-0.5
//   scores = h @ proj^T                 -> [seq, num_experts]
//   probs  = softmax(scores)
//   top-k by prob; renormalize the k weights to sum 1; multiply by per_expert_scale
//
//  Returns a DENSE weight matrix W [seq, num_experts]: the (renormalized, scaled) weight at each
//  token's chosen experts, 0 elsewhere. ESExperts then computes out = sum_e W[:,e] * expert_e(x).
#include "mlx/mlx.h"

namespace es {
namespace mx = mlx::core;

class ESRouter {
public:
    ESRouter(mx::array projWeight, mx::array scale, mx::array perExpertScale,
             int numExperts, int topK, int hiddenSize, float eps, mx::Dtype computeDtype);

    // Dense weight matrix [seq, numExperts] (0 for unchosen) — for the dense expert path.
    mx::array routeWeights(const mx::array & x) const;

    // Sparse routing for gather_mm: top-k expert indices [seq, k] (int32) and their weights [seq, k].
    struct TopK { mx::array idx; mx::array w; };
    TopK routeTopK(const mx::array & x) const;

private:
    mx::array proj_, scale_, perExpertScale_;
    int       numExperts_, topK_;
    float     eps_, scalarRoot_;   // scalarRoot_ = hidden^-0.5
    mx::Dtype computeDtype_;
};

}  // namespace es
