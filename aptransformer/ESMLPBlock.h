#pragma once
//  ESMLPBlock — Gemma4TextMLP, SwiGLU:  down( gelu_pytorch_tanh(gate(x)) * up(x) ).
//  Linears are bias-free:  y = x @ W^T  (HF weight shape [out, in]).
#include "mlx/mlx.h"
#include "ESLinear.h"

namespace es {
namespace mx = mlx::core;

class ESMLPBlock {
public:
    ESMLPBlock(mx::array gate, mx::array up, mx::array down,
               bool fused = false, int quantBits = 0, int groupSize = 64);
    mx::array forward(const mx::array & x) const;

private:
    ESLinear gate_, up_, down_;
    bool     fused_;
};

}  // namespace es
