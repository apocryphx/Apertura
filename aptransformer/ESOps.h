#pragma once
//  ESOps — small shared MLX helpers (dtype-safe literals, gelu, rotate_half, GQA repeat).
//  Inspectable/unfused: plain elementwise ops, values materialize between calls.
#include "mlx/mlx.h"

namespace es {
namespace mx = mlx::core;

// A scalar literal carrying `like`'s dtype, so it never promotes a bf16 expression to f32.
inline mx::array lit(float v, const mx::array & like) { return mx::array(v, like.dtype()); }

// gelu_pytorch_tanh: 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) )).
//
// Computed in f32 and rounded once on the way out. This is not gold-plating: bf16 carries 7
// mantissa bits, so evaluating the expression in bf16 quantises the CONSTANTS themselves —
// 0.7978845608 -> 0.796875 (1.3e-3 relative) and 0.044715 -> 0.0446777 (8.3e-4). That is not
// rounding error, it is a different function, biased the same way for every element of every
// MLP in every layer, so it compounds instead of averaging out.
//
// Measured against F.gelu(approximate='tanh') on bf16 activations:
//     all-bf16 (constants quantised)  55.2% bit-exact, max|d| 1.6e-2
//     f32 intermediates (this)        99.3% bit-exact, max|d| 1.5e-5
// Per-op conformance put the MLP at 43% bit-exact while every norm, RoPE and matmul was 100%:
// this was the single remaining source of divergence from the HF reference.
//
// Cost: on the fused path below the whole chain is one Metal kernel, so the f32 intermediates
// live in registers and never materialise — same bytes in, same bytes out. Unfused it does
// widen the temporaries, which is why the fused path is the shipping one.
inline mx::array geluTanh(const mx::array & x) {
    mx::array xf    = mx::astype(x, mx::float32);
    mx::array x3    = mx::multiply(mx::multiply(xf, xf), xf);
    mx::array inner = mx::multiply(mx::array(0.7978845608028654f),
                                   mx::add(xf, mx::multiply(mx::array(0.044715f), x3)));
    mx::array y     = mx::multiply(mx::multiply(mx::array(0.5f), xf),
                                   mx::add(mx::array(1.0f), mx::tanh(inner)));
    return mx::astype(y, x.dtype());
}

// Fused (performance-path) gelu: same math as above, the elementwise chain collapses into one
// Metal kernel via mx::compile(shapeless) — one trace per dtype, reused across shapes. The f32
// intermediates stay in registers inside the fused kernel. Bit-identical to geluTanh.
inline mx::array geluTanhFused(const mx::array & x) {
    static auto fn = mx::compile(
        [](const std::vector<mx::array> & in) -> std::vector<mx::array> {
            const mx::array & x = in[0];
            mx::array xf    = mx::astype(x, mx::float32);
            mx::array x3    = mx::multiply(mx::multiply(xf, xf), xf);
            mx::array inner = mx::multiply(mx::array(0.7978845608028654f),
                                           mx::add(xf, mx::multiply(mx::array(0.044715f), x3)));
            mx::array y     = mx::multiply(mx::multiply(mx::array(0.5f), xf),
                                           mx::add(mx::array(1.0f), mx::tanh(inner)));
            return {mx::astype(y, x.dtype())};
        },
        /*shapeless=*/true);
    return fn({x})[0];
}

// rotate_half: [.., d] -> cat(-x[.., d/2:], x[.., :d/2]).
inline mx::array rotateHalf(const mx::array & x) {
    auto halves = mx::split(x, 2, -1);  // {x1, x2}
    return mx::concatenate({mx::negative(halves[1]), halves[0]}, -1);
}

// repeat_kv: [nkv, s, d] -> [nkv*nrep, s, d], each kv head repeated nrep times contiguously.
inline mx::array repeatKV(const mx::array & x, int nrep) {
    if (nrep == 1) return x;
    return mx::repeat(x, nrep, 0);
}

}  // namespace es
