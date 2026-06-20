#include "ESExperts.h"
#include "ESOps.h"

#include <utility>

namespace es {

ESExperts::ESExperts(mx::array gateUp, mx::array down, int quantBits, int groupSize)
    : gateUp_(std::move(gateUp)), down_(std::move(down)),
      quant_(quantBits > 0), bits_(quantBits), gs_(groupSize),
      gateUpQ_(mx::array(0.0f)), gateUpS_(mx::array(0.0f)), gateUpB_(mx::array(0.0f)),
      downQ_(mx::array(0.0f)), downS_(mx::array(0.0f)), downB_(mx::array(0.0f)) {
    if (quant_) {
        auto gq = mx::quantize(gateUp_, gs_, bits_);  // quantize along last axis (in dim)
        gateUpQ_ = gq[0]; gateUpS_ = gq[1]; gateUpB_ = gq[2];
        auto dq = mx::quantize(down_, gs_, bits_);
        downQ_ = dq[0]; downS_ = dq[1]; downB_ = dq[2];
    }
}

ESExperts::ESExperts(mx::array gateUpQ, mx::array gateUpS, mx::array gateUpB,
                     mx::array downQ, mx::array downS, mx::array downB, int bits, int groupSize)
    : gateUp_(mx::array(0.0f)), down_(mx::array(0.0f)),  // bf16 dense path unavailable from a bundle
      quant_(true), bits_(bits), gs_(groupSize),
      gateUpQ_(std::move(gateUpQ)), gateUpS_(std::move(gateUpS)), gateUpB_(std::move(gateUpB)),
      downQ_(std::move(downQ)), downS_(std::move(downS)), downB_(std::move(downB)) {}

mx::array ESExperts::forward(const mx::array & x, const mx::array & W) const {
    const int seq = x.shape(0);

    // Evaluate every expert: [1,seq,hidden] @ [E,hidden,2I] -> [E,seq,2I] (batched over experts).
    mx::array xb = mx::expand_dims(x, 0);                              // [1, seq, hidden]
    mx::array gu = mx::matmul(xb, mx::transpose(gateUp_, {0, 2, 1}));  // [E, seq, 2I]
    auto halves  = mx::split(gu, 2, -1);                               // gate, up : [E, seq, I]
    mx::array y  = mx::multiply(geluTanh(halves[0]), halves[1]);       // SwiGLU [E, seq, I]
    y = mx::matmul(y, mx::transpose(down_, {0, 2, 1}));                // [E, seq, hidden]

    // Combine by the router weights: out[s] = sum_e W[s,e] * y[e,s,:].
    mx::array yt  = mx::transpose(y, {1, 0, 2});                       // [seq, E, hidden]
    mx::array out = mx::sum(mx::multiply(mx::expand_dims(W, -1), yt), 1);  // [seq, hidden]
    return out;
}

mx::array ESExperts::sparseForward(const mx::array & x, const mx::array & idx, const mx::array & w) const {
    // Per-token gather of only the top-k experts (mlx-lm SwitchGLU pattern). x [seq, hidden].
    mx::array xe = mx::expand_dims(x, std::vector<int>{-2, -3});       // [seq, 1, 1, hidden]

    // gate_up then down, gathering only the selected experts. Quantized via gather_qmm (transpose=true
    // expects [E, out, in] quantized along in), else bf16 gather_mm on the swapaxed weight.
    mx::array gu = quant_
        ? mx::gather_qmm(xe, gateUpQ_, gateUpS_, gateUpB_, std::nullopt, idx, /*transpose=*/true, gs_, bits_)
        : mx::gather_mm(xe, mx::swapaxes(gateUp_, -1, -2), std::nullopt, idx);   // -> [seq,k,1,2I]
    auto halves  = mx::split(gu, 2, -1);                              // gate, up : [seq,k,1,I]
    mx::array y  = mx::multiply(geluTanh(halves[0]), halves[1]);      // SwiGLU [seq,k,1,I]
    mx::array out = quant_
        ? mx::gather_qmm(y, downQ_, downS_, downB_, std::nullopt, idx, /*transpose=*/true, gs_, bits_)
        : mx::gather_mm(y, mx::swapaxes(down_, -1, -2), std::nullopt, idx);      // -> [seq,k,1,hidden]
    out = mx::squeeze(out, -2);                                       // [seq, k, hidden]
    return mx::sum(mx::multiply(mx::expand_dims(w, -1), out), 1);     // weighted sum -> [seq, hidden]
}

}  // namespace es
