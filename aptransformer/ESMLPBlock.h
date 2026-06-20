#pragma once
//  ESMLPBlock — Gemma4TextMLP, SwiGLU:  down( gelu_pytorch_tanh(gate(x)) * up(x) ).
//  Linears are bias-free:  y = x @ W^T  (HF weight shape [out, in]).
#include "mlx/mlx.h"
#include "ESLinear.h"

namespace es {
namespace mx = mlx::core;

class ESMLPBlock {
public:
    // Takes pre-built linears so the caller (ESDecoderLayer) can choose the quantize-now or
    // reload-from-bundle path per projection via esMakeLinear().
    ESMLPBlock(ESLinear gate, ESLinear up, ESLinear down, bool fused = false);
    mx::array forward(const mx::array & x) const;

private:
    ESLinear gate_, up_, down_;
    bool     fused_;
};

}  // namespace es
