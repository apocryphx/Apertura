#pragma once
//  ESVisionMLP — Gemma4VisionMLP: down( gelu_tanh(gate(x)) * up(x) ) with QAT-clipped linears.
#include "mlx/mlx.h"
#include "ESClippableLinear.h"
#include <string>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

class ESVisionMLP {
public:
    // prefix: "vision_tower.encoder.layers.N.mlp"
    ESVisionMLP(const std::string & prefix, const ESWeightLoader & w);
    mx::array forward(const mx::array & x) const;

private:
    ESClippableLinear gate_, up_, down_;
};

}  // namespace es
