#include "ESMLPBlock.h"
#include "ESOps.h"

#include <utility>

namespace es {

ESMLPBlock::ESMLPBlock(mx::array gate, mx::array up, mx::array down,
                       bool fused, int quantBits, int groupSize)
    : gate_(std::move(gate), quantBits, groupSize),
      up_(std::move(up), quantBits, groupSize),
      down_(std::move(down), quantBits, groupSize),
      fused_(fused) {}

mx::array ESMLPBlock::forward(const mx::array & x) const {
    mx::array g = (fused_ ? geluTanhFused : geluTanh)(gate_.forward(x));
    mx::array u = up_.forward(x);
    return down_.forward(mx::multiply(g, u));
}

}  // namespace es
