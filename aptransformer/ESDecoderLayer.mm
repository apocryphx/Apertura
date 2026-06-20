#include "ESDecoderLayer.h"
#include "ESWeightLoader.h"
#include "ESOps.h"

namespace es {

ESDecoderLayer::ESDecoderLayer(const ESModelConfig & config, int layerIdx, const ESWeightLoader & weights)
    : layerIdx_(layerIdx),
      inputLN_(weights.layer(layerIdx, "input_layernorm.weight"), config.rmsNormEps, config.fused),
      postAttnLN_(weights.layer(layerIdx, "post_attention_layernorm.weight"), config.rmsNormEps, config.fused),
      preFFLN_(weights.layer(layerIdx, "pre_feedforward_layernorm.weight"), config.rmsNormEps, config.fused),
      postFFLN_(weights.layer(layerIdx, "post_feedforward_layernorm.weight"), config.rmsNormEps, config.fused),
      attn_(config, layerIdx, weights),
      mlp_(esMakeLinear(weights, weights.layerKey(layerIdx, "mlp.gate_proj.weight"), config.quantBits, config.quantGroupSize),
           esMakeLinear(weights, weights.layerKey(layerIdx, "mlp.up_proj.weight"),   config.quantBits, config.quantGroupSize),
           esMakeLinear(weights, weights.layerKey(layerIdx, "mlp.down_proj.weight"), config.quantBits, config.quantGroupSize),
           config.fused),
      layerScalar_(weights.layer(layerIdx, "layer_scalar")),
      enableMoe_(config.enableMoeBlock), moeSparse_(config.moeSparse),
      hasPLE_(config.hasPLE()),
      perLayerInputGate_(mx::array(0.0f)), perLayerProjection_(mx::array(0.0f)) {
    if (hasPLE_) {
        perLayerInputGate_  = weights.layer(layerIdx, "per_layer_input_gate.weight");   // [ple_dim, hidden]
        perLayerProjection_ = weights.layer(layerIdx, "per_layer_projection.weight");   // [hidden, ple_dim]
        postPerLayerInputNorm_ = std::make_unique<ESRMSNorm>(
            weights.layer(layerIdx, "post_per_layer_input_norm.weight"), config.rmsNormEps, config.fused);
    }
    if (enableMoe_) {
        preFFLN2_  = std::make_unique<ESRMSNorm>(weights.layer(layerIdx, "pre_feedforward_layernorm_2.weight"),
                                                 config.rmsNormEps, config.fused);
        postFFLN1_ = std::make_unique<ESRMSNorm>(weights.layer(layerIdx, "post_feedforward_layernorm_1.weight"),
                                                 config.rmsNormEps, config.fused);
        postFFLN2_ = std::make_unique<ESRMSNorm>(weights.layer(layerIdx, "post_feedforward_layernorm_2.weight"),
                                                 config.rmsNormEps, config.fused);
        router_  = std::make_unique<ESRouter>(weights.layer(layerIdx, "router.proj.weight"),
                                              weights.layer(layerIdx, "router.scale"),
                                              weights.layer(layerIdx, "router.per_expert_scale"),
                                              config.numExperts, config.topKExperts, config.hiddenSize,
                                              config.rmsNormEps, config.computeDtype);
        experts_ = std::make_unique<ESExperts>(esMakeExperts(weights,
                                               weights.layerKey(layerIdx, "experts.gate_up_proj"),
                                               weights.layerKey(layerIdx, "experts.down_proj"),
                                               config.quantBits, config.quantGroupSize));
    }
}

mx::array ESDecoderLayer::forward(const mx::array & x,
                                  const mx::array & cos,
                                  const mx::array & sin,
                                  const mx::array & maskF32,
                                  ESKVCache *       cache,
                                  int               pastLen,
                                  const mx::array * perLayerInput,
                                  ESSharedKV *      sharedKV) const {
    mx::array residual = x;
    mx::array h = inputLN_.forward(x);
    h = attn_.forward(h, cos, sin, maskF32, cache, pastLen, sharedKV);
    h = postAttnLN_.forward(h);
    mx::array x1 = mx::add(residual, h);

    residual = x1;
    mx::array hmlp = mlp_.forward(preFFLN_.forward(x1));  // dense MLP on pre_ff_ln(residual)

    // MoE layers sum the dense MLP path with the expert path; both read the PRE-norm residual x1.
    // moeSparse_ touches only the router's top-k experts (gather_mm); else evaluate all experts.
    mx::array ff = hmlp;
    if (enableMoe_) {
        mx::array dense = postFFLN1_->forward(hmlp);
        mx::array expIn = preFFLN2_->forward(x1);
        mx::array moe = moeSparse_
            ? [&] { ESRouter::TopK tk = router_->routeTopK(x1);
                    return experts_->sparseForward(expIn, tk.idx, tk.w); }()
            : experts_->forward(expIn, router_->routeWeights(x1));
        ff = mx::add(dense, postFFLN2_->forward(moe));
    }
    h = postFFLN_.forward(ff);
    mx::array x2 = mx::add(residual, h);

    // Per-Layer Embeddings gate (elastic models): a per-layer residual driven by per_layer_input.
    //   r = x2; g = gelu(x2 @ gate^T) * per_layer_input; x2 = r + post_norm(g @ proj^T)
    if (hasPLE_ && perLayerInput) {
        mx::array g = geluTanh(mx::matmul(x2, mx::transpose(perLayerInputGate_)));  // [seq, ple_dim]
        g = mx::multiply(g, *perLayerInput);                                        // [seq, ple_dim]
        g = mx::matmul(g, mx::transpose(perLayerProjection_));                      // [seq, hidden]
        x2 = mx::add(x2, postPerLayerInputNorm_->forward(g));
    }

    return mx::multiply(x2, layerScalar_);  // broadcast [1]
}

}  // namespace es
