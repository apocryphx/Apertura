#include "ESRouter.h"

#include <cmath>
#include <utility>

namespace es {

ESRouter::ESRouter(mx::array projWeight, mx::array scale, mx::array perExpertScale,
                   int numExperts, int topK, int hiddenSize, float eps, mx::Dtype computeDtype)
    : proj_(std::move(projWeight)), scale_(std::move(scale)), perExpertScale_(std::move(perExpertScale)),
      numExperts_(numExperts), topK_(topK), eps_(eps),
      scalarRoot_((float) (1.0 / std::sqrt((double) hiddenSize))), computeDtype_(computeDtype) {}

ESRouter::TopK ESRouter::routeTopK(const mx::array & x) const {
    const int seq = x.shape(0);

    // Weightless RMSNorm (f32), then * scale * hidden^-0.5.
    mx::array xf  = mx::astype(x, mx::float32);
    mx::array ms  = mx::mean(mx::multiply(xf, xf), -1, /*keepdims=*/true);
    mx::array h   = mx::multiply(xf, mx::rsqrt(mx::add(ms, mx::array(eps_, mx::float32))));
    h = mx::multiply(mx::multiply(h, mx::astype(scale_, mx::float32)), mx::array(scalarRoot_, mx::float32));

    // Expert logits -> probabilities -> top-k (descending), renormalized, per-expert-scaled.
    mx::array scores = mx::matmul(h, mx::transpose(mx::astype(proj_, mx::float32)));  // [seq, E]
    mx::array probs  = mx::softmax(scores, -1, /*precise=*/true);
    mx::array order  = mx::argsort(mx::negative(probs), -1);
    mx::array topIdx = mx::astype(mx::slice(order, {0, 0}, {seq, topK_}), mx::int32);  // [seq, k]
    mx::array topP   = mx::take_along_axis(probs, topIdx, -1);
    mx::array topW   = mx::divide(topP, mx::sum(topP, -1, /*keepdims=*/true));
    topW = mx::multiply(topW, mx::take(mx::astype(perExpertScale_, mx::float32), topIdx, 0));
    return {topIdx, mx::astype(topW, computeDtype_)};
}

mx::array ESRouter::routeWeights(const mx::array & x) const {
    const int seq = x.shape(0);
    TopK tk = routeTopK(x);
    mx::array W = mx::zeros({seq, numExperts_}, mx::float32);
    W = mx::put_along_axis(W, tk.idx, mx::astype(tk.w, mx::float32), -1);
    return mx::astype(W, computeDtype_);
}

}  // namespace es
