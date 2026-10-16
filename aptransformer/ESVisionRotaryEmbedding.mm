#include "ESVisionRotaryEmbedding.h"
#include "ESOps.h"

#include <cmath>

namespace es {

ESVisionRotaryEmbedding::ESVisionRotaryEmbedding(int headDim, float thetaBase, mx::Dtype computeDtype)
    : headDim_(headDim), computeDtype_(computeDtype) {
    // compute_axial_rope_parameters: spatial_dim = headDim/2; inv_freq over arange(0, spatial_dim, 2).
    const int spatial = headDim / 2;
    for (int i = 0; i < spatial; i += 2) {
        invFreq_.push_back((float) (1.0 / std::pow((double) thetaBase, (double) i / (double) spatial)));
    }
}

std::pair<mx::array, mx::array> ESVisionRotaryEmbedding::cosSin(const mx::array & positionIds) const {
    const int n = positionIds.shape(0);
    const int q = (int) invFreq_.size();   // headDim/4
    mx::array pos  = mx::astype(positionIds, mx::float32);                      // [n, 2]
    mx::array invf = mx::array(invFreq_.data(), {1, 1, q}, mx::float32);
    mx::array freqs = mx::multiply(mx::expand_dims(pos, -1), invf);             // [n, 2, q]
    mx::array cos = mx::cos(freqs), sin = mx::sin(freqs);
    // recomposition_frequencies: cat[f_x, f_x, f_y, f_y] along the last axis.
    auto recompose = [&](const mx::array & f) {
        mx::array fx = mx::reshape(mx::slice(f, {0, 0, 0}, {n, 1, q}), {n, q});
        mx::array fy = mx::reshape(mx::slice(f, {0, 1, 0}, {n, 2, q}), {n, q});
        return mx::astype(mx::concatenate({fx, fx, fy, fy}, -1), computeDtype_);  // [n, headDim]
    };
    return {recompose(cos), recompose(sin)};
}

mx::array ESVisionRotaryEmbedding::apply(const mx::array & x, const mx::array & cos, const mx::array & sin) {
    // apply_multidimensional_rope, ndim 2: split head into two halves, rotate each with its block.
    auto xs = mx::split(x, 2, -1);
    auto cs = mx::split(mx::expand_dims(cos, 1), 2, -1);   // [n, 1, headDim/2] each
    auto ss = mx::split(mx::expand_dims(sin, 1), 2, -1);
    mx::array y0 = mx::add(mx::multiply(xs[0], cs[0]), mx::multiply(rotateHalf(xs[0]), ss[0]));
    mx::array y1 = mx::add(mx::multiply(xs[1], cs[1]), mx::multiply(rotateHalf(xs[1]), ss[1]));
    return mx::concatenate({y0, y1}, -1);
}

}  // namespace es
