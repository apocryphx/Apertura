#include "ESGemma4TextModel.h"
#include "ESWeightLoader.h"

#include <cmath>
#include <vector>

namespace es {

//  THE DECODER STACK — the body of the LLM beneath the LM head.
//
//  Construction wires the model from loaded weights once: the (optionally quantized) token
//  embedding, a constant embedding-scale factor (sqrt(hidden_size), bf16-rounded to reproduce
//  Gemma's quirk), 60 ESDecoderLayer objects (each one self-attention + MLP with its own weights),
//  the final RMSNorm, and the two rotary-embedding generators — one for local (sliding) layers
//  (full RoPE, theta 1e4) and one for global (full-attention) layers (p-RoPE, theta 1e6, partial
//  0.25). Which generator/mask a layer uses is decided per layer_idx by the hybrid 5:1 schedule.

ESGemma4TextModel::ESGemma4TextModel(const ESModelConfig & config, const ESWeightLoader & weights)
    : config_(config),
      embed_(esMakeEmbedding(weights, "embed_tokens.weight", config.quantEmbedBits, config.quantGroupSize)),
      embedScaleArr_(mx::astype(mx::array(config.embedScale()), config.computeDtype)),
      finalNorm_(weights.get("norm.weight"), config.rmsNormEps, config.fused),
      hasPLE_(config.hasPLE()),
      embedPerLayer_(mx::array(0.0f)), embedPerLayerScaleArr_(mx::array(0.0f)),
      perLayerModelProjection_(mx::array(0.0f)), perLayerProjScaleArr_(mx::array(0.0f)),
      perLayerInputScaleArr_(mx::array(0.0f)) {
    layers_.reserve(config.numHiddenLayers);
    for (int i = 0; i < config.numHiddenLayers; ++i) {
        layers_.push_back(std::make_unique<ESDecoderLayer>(config, i, weights));
    }
    localRope_ = std::make_unique<ESRotaryEmbedding>(
        config.headDim, config.ropeThetaLocal, /*partial=*/1.0f, config.computeDtype);
    globalRope_ = std::make_unique<ESRotaryEmbedding>(
        config.globalHeadDim, config.ropeThetaGlobal, config.globalPartialRotaryFactor, config.computeDtype);

    if (hasPLE_) {
        auto cd = [&](float v) { return mx::astype(mx::array(v), config.computeDtype); };
        embedPerLayer_           = weights.get("embed_tokens_per_layer.weight");
        embedPerLayerScaleArr_   = cd(std::sqrt((float) config.hiddenSizePerLayerInput));
        perLayerModelProjection_ = weights.get("per_layer_model_projection.weight");
        perLayerProjScaleArr_    = cd(1.0f / std::sqrt((float) config.hiddenSize));
        perLayerInputScaleArr_   = cd(1.0f / std::sqrt(2.0f));   // per_layer_input_scale = 2^-0.5
        perLayerProjectionNorm_  = std::make_unique<ESRMSNorm>(
            weights.get("per_layer_projection_norm.weight"), config.rmsNormEps, config.fused);
    }
}

// PLE: combine the token-identity component (embed_tokens_per_layer) with the context projection
// (per_layer_model_projection of the scaled embedding), normalized, scaled by 1/sqrt(2).
mx::array ESGemma4TextModel::computePerLayerInputs(const std::vector<int> & tokens,
                                                   const mx::array & scaledEmbed) const {
    const int seq = (int) tokens.size();
    const int L = config_.numHiddenLayers, ple = config_.hiddenSizePerLayerInput;
    mx::array ids = mx::array(tokens.data(), {seq}, mx::int32);

    // token-identity: per-layer embedding lookup × sqrt(ple), reshaped to [seq, L, ple].
    mx::array tokId = mx::multiply(mx::take(embedPerLayer_, ids, 0), embedPerLayerScaleArr_);
    tokId = mx::reshape(tokId, {seq, L, ple});

    // context: project the main scaled embedding × 1/sqrt(hidden), reshape, RMSNorm over ple.
    mx::array proj = mx::multiply(mx::matmul(scaledEmbed, mx::transpose(perLayerModelProjection_)),
                                  perLayerProjScaleArr_);
    proj = perLayerProjectionNorm_->forward(mx::reshape(proj, {seq, L, ple}));

    return mx::multiply(mx::add(proj, tokId), perLayerInputScaleArr_);  // [seq, L, ple]
}

mx::array ESGemma4TextModel::buildMask(int seqQ, int pastLen, bool sliding) const {
    const int seqK = pastLen + seqQ;
    const float NEG = -1e30f;
    std::vector<float> m((size_t) seqQ * seqK, 0.0f);
    for (int qi = 0; qi < seqQ; ++qi) {
        int qAbs = pastLen + qi;
        for (int kj = 0; kj < seqK; ++kj) {
            bool allowed = (kj <= qAbs);
            if (allowed && sliding) {
                allowed = (kj > qAbs - config_.slidingWindow);
            }
            if (!allowed) m[(size_t) qi * seqK + kj] = NEG;
        }
    }
    return mx::array(m.data(), {seqQ, seqK}, mx::float32);
}

// Forward: token ids -> final-norm hidden states [seq, hidden]. The residual stream `h` enters as
// the scaled embedding and is transformed in place by each decoder layer; nothing else is mutated
// (state lives in `cache`). pastLen = positions already in the cache (0 for a fresh prefill).
mx::array ESGemma4TextModel::forward(const std::vector<int> & tokens, ESKVCache * cache, int pastLen) const {
    const int seq = (int) tokens.size();

    // Embed each token id and scale by sqrt(hidden_size) -> the initial residual stream.
    mx::array h = mx::multiply(embed_.lookup(tokens), embedScaleArr_);

    // Positional rotations and attention masks depend only on (seq, pastLen, layer type), so build
    // each variant once and reuse across the 60 layers. Local layers get a sliding-window mask
    // (only the last `sliding_window` keys are visible) + local RoPE; global layers get a plain
    // causal mask + p-RoPE. (head_dim differs by type, so the two RoPEs produce different cos/sin.)
    auto localCS  = localRope_->cosSin(seq, pastLen);
    auto globalCS = globalRope_->cosSin(seq, pastLen);
    mx::array maskSliding = buildMask(seq, pastLen, /*sliding=*/true);
    mx::array maskFull    = buildMask(seq, pastLen, /*sliding=*/false);

    // Elastic-model state: shared-KV scratch (per forward) and the per-layer input embeddings.
    ESSharedKV shared;
    const int ple = config_.hiddenSizePerLayerInput;
    mx::array pleInputs = hasPLE_ ? computePerLayerInputs(tokens, h) : mx::array(0.0f);  // [seq, L, ple]

    // The decoder stack: each layer reads the residual stream, mixes information across positions
    // (attention, dispatched to the right RoPE+mask for its type) and across features (MLP), and
    // writes the result back. Depth is where the model's expressivity comes from.
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        bool sliding = config_.isSliding(i);                 // hybrid local:global schedule
        const auto & cs   = sliding ? localCS : globalCS;
        const auto & mask = sliding ? maskSliding : maskFull;
        mx::array pli = mx::array(0.0f);
        const mx::array * pliPtr = nullptr;
        if (hasPLE_) { pli = mx::reshape(mx::slice(pleInputs, {0, i, 0}, {seq, i + 1, ple}), {seq, ple}); pliPtr = &pli; }
        h = layers_[i]->forward(h, cs.first, cs.second, mask, cache, pastLen, pliPtr, &shared);
    }

    // Final RMSNorm before the LM head reads it.
    return finalNorm_.forward(h);
}

// On-device decode forward: same math as forward(), but the initial embedding is gathered from an
// on-device token-id array (`tokenIds`, int32 [seq]) instead of a host std::vector. This keeps the
// step lazy so the async decode loop can overlap consecutive tokens without a host readback.
// PLE (elastic) models need the host token ids to build per-layer inputs, so this fast path is
// restricted to non-PLE configs; callers must fall back to forward() otherwise.
mx::array ESGemma4TextModel::forwardDev(const mx::array & tokenIds, ESKVCache * cache, int pastLen,
                                        bool compiledTail) const {
    if (hasPLE_) throw std::runtime_error("forwardDev: not supported for PLE (elastic) models");
    const int seq = tokenIds.shape(0);

    mx::array h = mx::multiply(embed_.lookup(tokenIds), embedScaleArr_);

    auto localCS  = localRope_->cosSin(seq, pastLen);
    auto globalCS = globalRope_->cosSin(seq, pastLen);
    mx::array maskSliding = buildMask(seq, pastLen, /*sliding=*/true);
    mx::array maskFull    = buildMask(seq, pastLen, /*sliding=*/false);

    ESSharedKV shared;
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        bool sliding = config_.isSliding(i);
        const auto & cs   = sliding ? localCS : globalCS;
        const auto & mask = sliding ? maskSliding : maskFull;
        h = compiledTail
            ? layers_[i]->forwardDecodeCompiled(h, cs.first, cs.second, mask, cache, pastLen)
            : layers_[i]->forward(h, cs.first, cs.second, mask, cache, pastLen, nullptr, &shared);
    }
    return finalNorm_.forward(h);
}

mx::array ESGemma4TextModel::isolatedLayer(int layerIdx, const mx::array & xIn) const {
    const int seq = xIn.shape(0);
    bool sliding = config_.isSliding(layerIdx);
    auto cs = sliding ? localRope_->cosSin(seq, 0) : globalRope_->cosSin(seq, 0);
    mx::array mask = buildMask(seq, 0, sliding);
    return layers_[layerIdx]->forward(xIn, cs.first, cs.second, mask, nullptr, 0);
}

ESGemma4TextModel::Trace ESGemma4TextModel::forwardTrace(const std::vector<int> & tokens,
                                                         ESKVCache * cache, int pastLen) const {
    const int seq = (int) tokens.size();
    mx::array h = mx::multiply(embed_.lookup(tokens), embedScaleArr_);

    Trace tr{h, {}, h};
    tr.layerOut.reserve(config_.numHiddenLayers);

    auto localCS  = localRope_->cosSin(seq, pastLen);
    auto globalCS = globalRope_->cosSin(seq, pastLen);
    mx::array maskSliding = buildMask(seq, pastLen, true);
    mx::array maskFull    = buildMask(seq, pastLen, false);

    ESSharedKV shared;
    const int ple = config_.hiddenSizePerLayerInput;
    mx::array pleInputs = hasPLE_ ? computePerLayerInputs(tokens, h) : mx::array(0.0f);

    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        bool sliding = config_.isSliding(i);
        const auto & cs   = sliding ? localCS : globalCS;
        const auto & mask = sliding ? maskSliding : maskFull;
        mx::array pli = mx::array(0.0f);
        const mx::array * pliPtr = nullptr;
        if (hasPLE_) { pli = mx::reshape(mx::slice(pleInputs, {0, i, 0}, {seq, i + 1, ple}), {seq, ple}); pliPtr = &pli; }
        h = layers_[i]->forward(h, cs.first, cs.second, mask, cache, pastLen, pliPtr, &shared);
        tr.layerOut.push_back(h);
    }
    tr.finalNorm = finalNorm_.forward(h);
    return tr;
}

}  // namespace es
