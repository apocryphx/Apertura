#include "ESRotaryEmbedding.h"
#include "ESOps.h"

#include <cmath>

namespace es {

ESRotaryEmbedding::ESRotaryEmbedding(int headDim, float thetaBase, float partialRotaryFactor,
                                     mx::Dtype computeDtype)
    : headDim_(headDim), computeDtype_(computeDtype) {
    const int half = headDim / 2;
    invFreq_.assign(half, 0.0f);

    // rope_angles = int(partial * headDim // 2). For full rotation (factor 1.0) this is `half`.
    int ropeAngles = (int) ((double) partialRotaryFactor * headDim / 2.0);
    if (ropeAngles > half) ropeAngles = half;

    // inv_freq_rotated[i] = 1 / base^(2i / headDim), i in [0, ropeAngles); rest stay 0.
    for (int i = 0; i < ropeAngles; ++i) {
        double exponent = (double) (2 * i) / (double) headDim;
        invFreq_[i] = (float) (1.0 / std::pow((double) thetaBase, exponent));
    }
}

std::pair<mx::array, mx::array> ESRotaryEmbedding::cosSin(int seqLen, int offset) const {
    const int half = headDim_ / 2;

    // positions [seqLen, 1], invFreq [1, half]  ->  freqs [seqLen, half]  (float32)
    std::vector<float> pos(seqLen);
    for (int i = 0; i < seqLen; ++i) pos[i] = (float) (offset + i);
    mx::array posArr  = mx::array(pos.data(), {seqLen, 1}, mx::float32);
    mx::array freqArr = mx::array(invFreq_.data(), {1, half}, mx::float32);
    mx::array freqs   = mx::matmul(posArr, freqArr);              // [seqLen, half]
    mx::array emb     = mx::concatenate({freqs, freqs}, -1);      // [seqLen, headDim]

    mx::array cos = mx::astype(mx::cos(emb), computeDtype_);
    mx::array sin = mx::astype(mx::sin(emb), computeDtype_);
    return {cos, sin};
}

mx::array ESRotaryEmbedding::apply(const mx::array & x, const mx::array & cos, const mx::array & sin) {
    // x: [seq, heads, headDim]; cos/sin: [seq, headDim] -> [seq, 1, headDim] to broadcast over heads.
    mx::array c = mx::expand_dims(cos, 1);
    mx::array s = mx::expand_dims(sin, 1);
    return mx::add(mx::multiply(x, c), mx::multiply(rotateHalf(x), s));
}

}  // namespace es
