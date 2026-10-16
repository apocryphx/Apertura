#pragma once
//  ESGemma4VisionModel — Gemma4VisionModel + Gemma4MultimodalEmbedder: an image's patches ->
//  the soft tokens that replace its <|image|> placeholders in the text embedding stream.
//
//    patch_embedder -> 16 encoder layers (axial 2-D RoPE) -> pooler (3x3, f32, sqrt(hidden))
//    -> [standardize] -> cast bf16 = tower output [tokens, 768]
//    -> embed_vision: rmsnorm_noscale + projection -> [tokens, text hidden]
//
//  The engine runs the valid patches only (no padding), so the pooled tokens are exactly the
//  image's (nh/3)*(nw/3) soft tokens — the count the chat template must expand <|image|> to.
//
//  Unified family (config.vision.kind == unified, 12B QAT): there is no tower; softTokens()
//  dispatches to ESUnifiedVisionEmbedder over the preprocessor's merged 48-px patches. Same
//  placeholder grammar, same scatter.
#include "mlx/mlx.h"
#include "ESModelConfig.h"
#include "ESImagePreprocessor.h"
#include "ESVisionPatchEmbedder.h"
#include "ESVisionRotaryEmbedding.h"
#include "ESVisionEncoderLayer.h"
#include "ESVisionPooler.h"
#include "ESMultimodalEmbedder.h"
#include "ESUnifiedVisionEmbedder.h"
#include <memory>
#include <vector>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

class ESGemma4VisionModel {
public:
    // Throws if the loader has no vision tower (ESWeightLoader::hasVisionTower()).
    ESGemma4VisionModel(const ESModelConfig & config, const ESWeightLoader & weights);

    // The soft tokens for one image, in text-embedding space: [tokens, textHidden] compute dtype.
    mx::array softTokens(const ESImagePatches & patches) const;

    // Conformance trace of every stage.
    struct Trace {
        mx::array patchEmbed;              // [n, 768]
        mx::array cos, sin;                // [n, 64]
        std::vector<mx::array> layerOut;   // 16 x [n, 768]
        mx::array pooled;                  // f32 [tokens, 768] (pooler output, pre-standardize)
        mx::array towerOut;                // [tokens, 768] compute dtype
        mx::array softTokens;              // [tokens, textHidden]
    };
    Trace forwardTrace(const ESImagePatches & patches) const;   // tower family only
    ESUnifiedVisionEmbedder::Trace forwardTraceUnified(const ESImagePatches & patches) const;  // unified only
    ESVisionKind kind() const { return cfg_.kind; }

    const ESVisionEncoderLayer & layer(int i) const { return *layers_[i]; }
    const ESVisionRotaryEmbedding & rope() const { return *rope_; }
    const ESUnifiedVisionEmbedder & unified() const { return *unified_; }
    const ESVisionConfig & config() const { return cfg_; }

private:
    mx::array towerOutput(const ESImagePatches & patches, Trace * trace) const;

    ESVisionConfig          cfg_;
    // Tower family (null for unified).
    std::unique_ptr<ESVisionPatchEmbedder>   patchEmbedder_;
    std::unique_ptr<ESVisionRotaryEmbedding> rope_;
    std::vector<std::unique_ptr<ESVisionEncoderLayer>> layers_;
    std::unique_ptr<ESVisionPooler>          pooler_;
    bool                    standardize_ = false;
    mx::array               stdBias_, stdScale_;
    std::unique_ptr<ESMultimodalEmbedder>    embedder_;
    // Unified family (null for tower).
    std::unique_ptr<ESUnifiedVisionEmbedder> unified_;
};

}  // namespace es
