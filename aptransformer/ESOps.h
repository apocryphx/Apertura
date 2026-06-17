#pragma once
//  ESOps — small shared MLX helpers (dtype-safe literals, gelu, rotate_half, GQA repeat).
//  Inspectable/unfused: plain elementwise ops, values materialize between calls.
#include "mlx/mlx.h"

namespace es {
namespace mx = mlx::core;

// A scalar literal carrying `like`'s dtype, so it never promotes a bf16 expression to f32.
inline mx::array lit(float v, const mx::array & like) { return mx::array(v, like.dtype()); }

// gelu_pytorch_tanh: 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) )). Exact HF match.
inline mx::array geluTanh(const mx::array & x) {
    mx::array x3    = mx::multiply(mx::multiply(x, x), x);
    mx::array inner = mx::multiply(lit(0.7978845608028654f, x),
                                   mx::add(x, mx::multiply(lit(0.044715f, x), x3)));
    return mx::multiply(mx::multiply(lit(0.5f, x), x), mx::add(lit(1.0f, x), mx::tanh(inner)));
}

// Fused (performance-path) gelu: same math, the ~8 elementwise ops collapse into one Metal
// kernel via mx::compile(shapeless) — one trace per dtype, reused across shapes. Bit-identical.
inline mx::array geluTanhFused(const mx::array & x) {
    static auto fn = mx::compile(
        [](const std::vector<mx::array> & in) -> std::vector<mx::array> {
            const mx::array & x = in[0];
            mx::array x3    = mx::multiply(mx::multiply(x, x), x);
            mx::array inner = mx::multiply(lit(0.7978845608028654f, x),
                                           mx::add(x, mx::multiply(lit(0.044715f, x), x3)));
            return {mx::multiply(mx::multiply(lit(0.5f, x), x), mx::add(lit(1.0f, x), mx::tanh(inner)))};
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
