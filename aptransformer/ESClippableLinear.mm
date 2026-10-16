#include "ESClippableLinear.h"

#include <utility>

namespace es {

ESClippableLinear::ESClippableLinear(ESLinear linear)
    : linear_(std::move(linear)), clipped_(false),
      inMin_(mx::array(0.0f)), inMax_(mx::array(0.0f)), outMin_(mx::array(0.0f)), outMax_(mx::array(0.0f)) {}

ESClippableLinear::ESClippableLinear(ESLinear linear, mx::array inputMin, mx::array inputMax,
                                     mx::array outputMin, mx::array outputMax)
    : linear_(std::move(linear)), clipped_(true),
      inMin_(std::move(inputMin)), inMax_(std::move(inputMax)),
      outMin_(std::move(outputMin)), outMax_(std::move(outputMax)) {}

mx::array ESClippableLinear::forward(const mx::array & x) const {
    if (!clipped_) return linear_.forward(x);
    // torch.clamp(x, min, max) with 0-d bf16 bounds: elementwise in the activation dtype.
    mx::array h = mx::clip(x, inMin_, inMax_);
    h = linear_.forward(h);
    return mx::clip(h, outMin_, outMax_);
}

}  // namespace es
