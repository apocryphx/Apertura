#pragma once
//  ESVisionEncoderLayer — Gemma4VisionEncoderLayer: the same 4-norm sandwich as the text
//  decoder, without the per-layer scalar and without PLE.
//
//   r = x; h = input_layernorm(x); h = attn(h); h = post_attention_layernorm(h); x = r + h
//   r = x; h = pre_feedforward_layernorm(x); h = mlp(h); h = post_feedforward_layernorm(h); x = r + h
#include "mlx/mlx.h"
#include "ESVisionConfig.h"
#include "ESRMSNorm.h"
#include "ESVisionAttention.h"
#include "ESVisionMLP.h"

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

class ESVisionEncoderLayer {
public:
    ESVisionEncoderLayer(const ESVisionConfig & cfg, int layerIdx, const ESWeightLoader & w);

    mx::array forward(const mx::array & x, const mx::array & cos, const mx::array & sin) const;

    const ESVisionAttention & attention() const { return attn_; }
    const ESVisionMLP & mlp() const { return mlp_; }

private:
    ESRMSNorm inputLN_, postAttnLN_, preFFLN_, postFFLN_;
    ESVisionAttention attn_;
    ESVisionMLP mlp_;
};

}  // namespace es
