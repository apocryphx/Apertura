#include "ESEmbedding.h"

#include <utility>

namespace es {

ESEmbedding::ESEmbedding(mx::array weight, int quantBits, int groupSize)
    : quant_(quantBits > 0),
      bits_(quantBits),
      groupSize_(groupSize),
      w_(std::move(weight)),
      scales_(mx::array(0.0f)),
      biases_(mx::array(0.0f)) {
    if (quant_) {
        auto parts = mx::quantize(w_, groupSize_, bits_);  // {w_q, scales, biases}
        w_      = parts[0];
        scales_ = parts[1];
        biases_ = parts[2];
    }
}

ESEmbedding::ESEmbedding(mx::array packedWeight, mx::array scales, mx::array biases, int bits, int groupSize)
    : quant_(true), bits_(bits), groupSize_(groupSize),
      w_(std::move(packedWeight)), scales_(std::move(scales)), biases_(std::move(biases)) {}

mx::array ESEmbedding::lookup(const std::vector<int> & ids) const {
    mx::array idx = mx::array(ids.data(), {(int) ids.size()}, mx::int32);
    return lookup(idx);
}

// On-device gather: `idx` is an int32 [n] token-id array that already lives on the GPU
// (e.g. the argmax of the previous step's logits). Feeding it back without a host readback
// keeps the whole decode chain lazy so consecutive token forwards can overlap. Identical math
// to the host-vector overload — that one just builds `idx` from a std::vector first.
mx::array ESEmbedding::lookup(const mx::array & idx) const {
    if (!quant_) {
        return mx::take(w_, idx, 0);  // [n, hidden]
    }
    // Gather the quantized rows + their scales/biases, then dequantize.
    mx::array wq = mx::take(w_, idx, 0);       // [n, packed_in]
    mx::array sc = mx::take(scales_, idx, 0);  // [n, in/group]
    mx::array bi = mx::take(biases_, idx, 0);
    return mx::dequantize(wq, sc, bi, groupSize_, bits_);  // [n, hidden]
}

mx::array ESEmbedding::logits(const mx::array & hidden) const {
    if (!quant_) {
        return mx::matmul(hidden, mx::transpose(w_));  // [seq, vocab]
    }
    return mx::quantized_matmul(hidden, w_, scales_, biases_, /*transpose=*/true, groupSize_, bits_);
}

}  // namespace es
