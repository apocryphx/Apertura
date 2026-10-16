#pragma once
//  ESVisionAttention — Gemma4VisionAttention: bidirectional MHA over the patch sequence.
//
//  12 heads x 64, no GQA (kv heads == heads), scaling 1.0 (QK-norm absorbs 1/sqrt(d)):
//    q = rope2d( q_norm( q_proj(x) ) ),  k = rope2d( k_norm( k_proj(x) ) ),  v = v_norm( v_proj(x) )
//    out = o_proj( softmax_f32(q k^T) v )
//  All four projections are QAT-clipped (ESClippableLinear); v_norm is weightless. No mask:
//  every valid patch attends to every other (the reference masks only padding patches, which
//  the engine does not carry).
#include "mlx/mlx.h"
#include "ESVisionConfig.h"
#include "ESClippableLinear.h"
#include "ESRMSNorm.h"
#include <string>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

class ESVisionAttention {
public:
    // prefix: "vision_tower.encoder.layers.N.self_attn"
    ESVisionAttention(const ESVisionConfig & cfg, const std::string & prefix, const ESWeightLoader & w);

    // x: [n, hidden]; cos/sin: [n, headDim]. Returns [n, hidden].
    mx::array forward(const mx::array & x, const mx::array & cos, const mx::array & sin) const;

private:
    int numHeads_, numKVHeads_, headDim_;
    ESClippableLinear qProj_, kProj_, vProj_, oProj_;
    ESRMSNorm qNorm_, kNorm_, vNorm_;
};

}  // namespace es
