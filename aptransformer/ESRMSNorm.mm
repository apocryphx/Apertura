#include "ESRMSNorm.h"
#include "ESOps.h"

#include <utility>

namespace es {

ESRMSNorm::ESRMSNorm(mx::array weight, float eps, bool fused)
    : withScale_(true), weight_(std::move(weight)), eps_(eps), fused_(fused) {}

ESRMSNorm::ESRMSNorm(float eps, bool fused)
    : withScale_(false), weight_(mx::array(0.0f)), eps_(eps), fused_(fused) {}

mx::array ESRMSNorm::forward(const mx::array & x) const {
    if (fused_) {
        // Fused Metal kernel; f32 accumulation internally, same math as the manual path.
        std::optional<mx::array> w = withScale_ ? std::optional<mx::array>(weight_) : std::nullopt;
        return mx::fast::rms_norm(x, w, eps_);
    }
    mx::array xf  = mx::astype(x, mx::float32);
    mx::array ms  = mx::mean(mx::multiply(xf, xf), -1, /*keepdims=*/true);  // mean(x^2, last)
    mx::array inv = mx::rsqrt(mx::add(ms, mx::array(eps_, mx::float32)));
    mx::array out = mx::multiply(xf, inv);
    if (withScale_) {
        out = mx::multiply(out, mx::astype(weight_, mx::float32));
    }
    return mx::astype(out, x.dtype());
}

}  // namespace es
