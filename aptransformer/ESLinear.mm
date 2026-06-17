#include "ESLinear.h"

#include <utility>

namespace es {

ESLinear::ESLinear(mx::array weight, int quantBits, int groupSize)
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

mx::array ESLinear::forward(const mx::array & x) const {
    if (quant_) {
        return mx::quantized_matmul(x, w_, scales_, biases_, /*transpose=*/true, groupSize_, bits_);
    }
    return mx::matmul(x, mx::transpose(w_));
}

}  // namespace es
