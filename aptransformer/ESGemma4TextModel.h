#pragma once
//  ESGemma4TextModel — Gemma4TextModel. Scaled embedding -> 60 decoder layers -> final norm.
//
//  Per-layer-type RoPE (local full / global p-RoPE) and masks (causal / sliding) are built
//  once per forward and dispatched by layer_idx.
#include "mlx/mlx.h"
#include "ESModelConfig.h"
#include "ESRMSNorm.h"
#include "ESDecoderLayer.h"
#include "ESRotaryEmbedding.h"
#include "ESEmbedding.h"
#include "ESKVCache.h"
#include <memory>
#include <vector>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

class ESGemma4TextModel {
public:
    ESGemma4TextModel(const ESModelConfig & config, const ESWeightLoader & weights);

    // tokens: input ids. cache may be null (prefill-only). pastLen = positions already cached.
    // Returns final-norm hidden states [seq, hidden].
    mx::array forward(const std::vector<int> & tokens, ESKVCache * cache, int pastLen) const;

    // Conformance trace: scaled embedding, every decoder-layer output, and final norm.
    struct Trace {
        mx::array              embed;
        std::vector<mx::array> layerOut;  // size numHiddenLayers
        mx::array              finalNorm;
    };
    Trace forwardTrace(const std::vector<int> & tokens, ESKVCache * cache, int pastLen) const;

    // Run a single decoder layer in isolation on a given input (no cache, pastLen 0).
    // Used for per-layer conformance: feed the golden previous-layer output, compare the output
    // to that layer's golden — isolates real bugs from accumulated bf16 drift.
    mx::array isolatedLayer(int layerIdx, const mx::array & xIn) const;

    const ESEmbedding & embedding() const { return embed_; }
    const ESModelConfig & config() const { return config_; }

private:
    ESModelConfig config_;
    ESEmbedding   embed_;
    mx::array     embedScaleArr_;  // computeDtype scalar (bf16-rounded embed scale)
    ESRMSNorm     finalNorm_;
    std::vector<std::unique_ptr<ESDecoderLayer>> layers_;
    std::unique_ptr<ESRotaryEmbedding> localRope_, globalRope_;

    // Per-Layer Embeddings (elastic models). per_layer_inputs[seq, num_layers, ple] is built once
    // per forward and a [seq, ple] slice fed to each layer.
    bool      hasPLE_;
    mx::array embedPerLayer_;            // [vocab_per_layer, num_layers*ple]
    mx::array embedPerLayerScaleArr_;    // bf16 sqrt(ple)
    mx::array perLayerModelProjection_;  // [num_layers*ple, hidden]
    mx::array perLayerProjScaleArr_;     // bf16 1/sqrt(hidden)
    mx::array perLayerInputScaleArr_;    // bf16 1/sqrt(2)
    std::unique_ptr<ESRMSNorm> perLayerProjectionNorm_;
    mx::array computePerLayerInputs(const std::vector<int> & tokens, const mx::array & scaledEmbed) const;

    // Additive f32 mask [seqQ, seqK]; sliding adds the window lower bound.
    mx::array buildMask(int seqQ, int pastLen, bool sliding) const;
};

}  // namespace es
