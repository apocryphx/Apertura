#include "ESMLPBlock.h"
#include "ESOps.h"

#include <utility>

namespace es {

ESMLPBlock::ESMLPBlock(ESLinear gate, ESLinear up, ESLinear down, bool fused)
    : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)), fused_(fused) {}

mx::array ESMLPBlock::forward(const mx::array & x) const {
    mx::array g = (fused_ ? geluTanhFused : geluTanh)(gate_.forward(x));
    mx::array u = up_.forward(x);
    return down_.forward(mx::multiply(g, u));
}

}  // namespace es
