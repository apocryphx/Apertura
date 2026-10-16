#include "ESVisionPooler.h"

namespace es {

mx::array ESVisionPooler::forward(const mx::array & h, const mx::array & positionIds, int tokens) const {
    const int n = h.shape(0);
    mx::array xs = mx::reshape(mx::slice(positionIds, {0, 0}, {n, 1}), {n});
    mx::array ys = mx::reshape(mx::slice(positionIds, {0, 1}, {n, 2}), {n});
    mx::array maxX = mx::add(mx::max(xs), mx::array(1, mx::int32));
    mx::array kernelIdx = mx::add(mx::floor_divide(xs, mx::array(k_, mx::int32)),
                                  mx::multiply(mx::floor_divide(maxX, mx::array(k_, mx::int32)),
                                               mx::floor_divide(ys, mx::array(k_, mx::int32))));  // [n]
    // one_hot(kernel_idx, tokens).float() / k^2  ->  [n, tokens]
    mx::array onehot = mx::astype(mx::equal(mx::expand_dims(kernelIdx, 1), mx::arange(tokens, mx::int32)),
                                  mx::float32);
    mx::array weights = mx::divide(onehot, mx::array((float) (k_ * k_)));
    mx::array pooled = mx::matmul(mx::transpose(weights), mx::astype(h, mx::float32));   // [tokens, hidden]
    pooled = mx::astype(pooled, h.dtype());                      // output.to(hidden_states.dtype)
    return mx::multiply(mx::astype(pooled, mx::float32), mx::array(rootHidden_));   // f32 * sqrt(hidden)
}

}  // namespace es
