#pragma once
//  ESDecoderLayer — Gemma4TextDecoderLayer. Sandwich norms + per-layer scalar.
//
//  Dense (31B):
//   r = x; h = input_layernorm(x); h = attn(h); h = post_attention_layernorm(h); x = r + h
//   r = x; h = pre_feedforward_layernorm(x); h = mlp(h); h = post_feedforward_layernorm(h); x = r + h
//   x = x * layer_scalar
//
//  MoE (26B, enable_moe_block): the feedforward stage runs the dense MLP AND a sparse expert block
//  in parallel and sums them before the post norm:
//   dense = post_ff_ln_1(mlp(pre_ff_ln(r)))
//   moe   = post_ff_ln_2(experts(pre_ff_ln_2(r), router(r)))
//   x = r + post_feedforward_layernorm(dense + moe)
#include "mlx/mlx.h"
#include "ESModelConfig.h"
#include "ESRMSNorm.h"
#include "ESMLPBlock.h"
#include "ESAttention.h"
#include "ESKVCache.h"
#include "ESRouter.h"
#include "ESExperts.h"
#include <memory>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

class ESDecoderLayer {
public:
    ESDecoderLayer(const ESModelConfig & config, int layerIdx, const ESWeightLoader & weights);

    // perLayerInput (PLE, elastic models) and sharedKV (shared-KV) are null for plain models.
    mx::array forward(const mx::array & x,
                      const mx::array & cos,
                      const mx::array & sin,
                      const mx::array & maskF32,
                      ESKVCache *       cache,
                      int               pastLen,
                      const mx::array * perLayerInput = nullptr,
                      ESSharedKV *      sharedKV = nullptr) const;

private:
    int       layerIdx_;
    ESRMSNorm inputLN_, postAttnLN_, preFFLN_, postFFLN_;
    ESAttention attn_;
    ESMLPBlock  mlp_;
    mx::array   layerScalar_;  // [1]

    // MoE block (valid iff enableMoe_).
    bool                      enableMoe_, moeSparse_;
    std::unique_ptr<ESRMSNorm> preFFLN2_, postFFLN1_, postFFLN2_;
    std::unique_ptr<ESRouter>  router_;
    std::unique_ptr<ESExperts> experts_;

    // Per-Layer Embeddings gate (valid iff hasPLE_): a per-layer residual driven by per_layer_input.
    bool                       hasPLE_;
    mx::array                  perLayerInputGate_, perLayerProjection_;  // bias-free Linears
    std::unique_ptr<ESRMSNorm> postPerLayerInputNorm_;
};

}  // namespace es
