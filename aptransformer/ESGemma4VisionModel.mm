#include "ESGemma4VisionModel.h"
#include "ESWeightLoader.h"

#include <stdexcept>

namespace es {

ESGemma4VisionModel::ESGemma4VisionModel(const ESModelConfig & config, const ESWeightLoader & weights)
    : cfg_(config.vision), stdBias_(mx::array(0.0f)), stdScale_(mx::array(1.0f)) {
    if (!weights.hasVisionTower())
        throw std::runtime_error("ESGemma4VisionModel: the loaded weights carry no vision tower");
    if (cfg_.kind == ESVisionKind::unified) {
        unified_ = std::make_unique<ESUnifiedVisionEmbedder>(cfg_, weights);
        return;
    }
    patchEmbedder_ = std::make_unique<ESVisionPatchEmbedder>(
        esMakeLinear(weights, "vision_tower.patch_embedder.input_proj.weight", 0, 64),
        weights.get("vision_tower.patch_embedder.position_embedding_table"));
    rope_   = std::make_unique<ESVisionRotaryEmbedding>(cfg_.headDim, cfg_.ropeTheta, config.computeDtype);
    pooler_ = std::make_unique<ESVisionPooler>(cfg_.hiddenSize, cfg_.poolingKernelSize);
    embedder_ = std::make_unique<ESMultimodalEmbedder>(
        esMakeLinear(weights, "embed_vision.embedding_projection.weight", 0, 64), cfg_.rmsNormEps);
    standardize_ = cfg_.standardize;
    layers_.reserve(cfg_.numHiddenLayers);
    for (int i = 0; i < cfg_.numHiddenLayers; ++i)
        layers_.push_back(std::make_unique<ESVisionEncoderLayer>(cfg_, i, weights));
    if (standardize_) {
        stdBias_  = mx::astype(weights.get("vision_tower.std_bias"), mx::float32);
        stdScale_ = mx::astype(weights.get("vision_tower.std_scale"), mx::float32);
    }
}

mx::array ESGemma4VisionModel::towerOutput(const ESImagePatches & patches, Trace * trace) const {
    if (unified_) throw std::runtime_error("ESGemma4VisionModel: towerOutput on a unified-family model");
    mx::array h = patchEmbedder_->forward(patches.pixelValues, patches.positionIds);
    auto cs = rope_->cosSin(patches.positionIds);
    if (trace) { trace->patchEmbed = h; trace->cos = cs.first; trace->sin = cs.second; }
    for (int i = 0; i < cfg_.numHiddenLayers; ++i) {
        h = layers_[i]->forward(h, cs.first, cs.second);
        if (trace) trace->layerOut.push_back(h);
    }
    const int tokens = patches.numSoftTokens();
    mx::array pooled = pooler_->forward(h, patches.positionIds, tokens);  // f32
    if (trace) trace->pooled = pooled;
    if (standardize_) pooled = mx::multiply(mx::subtract(pooled, stdBias_), stdScale_);
    return mx::astype(pooled, h.dtype());
}

mx::array ESGemma4VisionModel::softTokens(const ESImagePatches & patches) const {
    if (unified_) return unified_->forward(patches.pixelValues, patches.positionIds);
    return embedder_->forward(towerOutput(patches, nullptr));
}

ESUnifiedVisionEmbedder::Trace ESGemma4VisionModel::forwardTraceUnified(const ESImagePatches & patches) const {
    if (!unified_) throw std::runtime_error("ESGemma4VisionModel: forwardTraceUnified on a tower-family model");
    return unified_->forwardTrace(patches.pixelValues, patches.positionIds);
}

ESGemma4VisionModel::Trace ESGemma4VisionModel::forwardTrace(const ESImagePatches & patches) const {
    Trace tr{mx::array(0.0f), mx::array(0.0f), mx::array(0.0f), {}, mx::array(0.0f), mx::array(0.0f), mx::array(0.0f)};
    tr.towerOut = towerOutput(patches, &tr);
    tr.softTokens = embedder_->forward(tr.towerOut);
    return tr;
}

}  // namespace es
