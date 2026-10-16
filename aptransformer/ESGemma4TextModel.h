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

// ESSoftTokens — embedding substitutions for one forward: the image soft tokens that replace the
// <|image|> placeholders (Gemma4Model.forward's masked_scatter). `positions` index this forward's
// token vector (ascending); `embeds` [positions.size(), hidden] are already in text-embedding
// space (ESGemma4VisionModel::softTokens), in the compute dtype. The placeholder ids stay in the
// token vector — they drive nothing but the PLE lookup, where the reference substitutes the pad
// row (per_layer_inputs are formed from `where(image_mask, pad_token_id, input_ids)` while the
// context projection sees the scattered embeddings). A null/empty ESSoftTokens leaves every
// text-only path byte-identical.
struct ESSoftTokens {
    std::vector<int> positions;
    mx::array        embeds;
    ESSoftTokens() : embeds(mx::array(0.0f)) {}
    ESSoftTokens(std::vector<int> pos, mx::array emb) : positions(std::move(pos)), embeds(std::move(emb)) {}
    bool empty() const { return positions.empty(); }
    // Substitute the rows of h [seq, hidden] at `positions`.
    mx::array apply(const mx::array & h) const;
    // The entries in [begin, end), re-based to `begin` (chunked prefill).
    ESSoftTokens slice(int begin, int end) const;
    // Per position of a seq-long forward: the image index for soft-token positions (a maximal
    // run of consecutive positions is one image), -1 for text. Drives the bidirectional overlay.
    std::vector<int> blockIds(int seq) const;
};

class ESGemma4TextModel {
public:
    ESGemma4TextModel(const ESModelConfig & config, const ESWeightLoader & weights);

    // tokens: input ids. cache may be null (prefill-only). pastLen = positions already cached.
    // Returns final-norm hidden states [seq, hidden].
    // lastOnly (elastic models): the caller reads only the last row, so the shared-KV layers —
    // which write no cache — run on the last position only and the result is [1, hidden].
    // cacheOnly (elastic models): the caller only needs the cache filled (a middle prefill chunk),
    // so the stack stops before the shared-KV layers; the returned array is just something to eval.
    // soft (optional): image soft tokens scattered over the placeholder positions (see ESSoftTokens).
    mx::array forward(const std::vector<int> & tokens, ESKVCache * cache, int pastLen,
                      bool lastOnly = false, bool cacheOnly = false,
                      const ESSoftTokens * soft = nullptr) const;

    // On-device decode forward (non-PLE only): initial embedding gathered from an int32 [seq]
    // token-id array living on the GPU, so the step stays lazy for async overlap. compiledTail=true
    // routes each layer through its per-instance compiled stateless tail (graph encoded once).
    mx::array forwardDev(const mx::array & tokenIds, ESKVCache * cache, int pastLen,
                         bool compiledTail = false) const;

    // Compiled-step decode forward (P3, non-PLE): identical layer math to forwardDev, but the
    // position-dependent pieces (RoPE cos/sin and both additive masks) are PARAMETERS instead of
    // being built internally from host ints — so the whole call is traceable by mx::compile and
    // one recorded graph replays for every decode position. The cache must be in step mode
    // (ESKVCache::beginStep): appends scatter at a position array, attention reads full-capacity
    // buffers, and the masks kill invalid/expired slots (softmax weight exactly 0).
    mx::array forwardStep(const mx::array & tokenIds,
                          const std::pair<mx::array, mx::array> & localCS,
                          const std::pair<mx::array, mx::array> & globalCS,
                          const mx::array & maskSliding,
                          const mx::array & maskFull,
                          ESKVCache * cache) const;

    const ESRotaryEmbedding & localRope() const { return *localRope_; }
    const ESRotaryEmbedding & globalRope() const { return *globalRope_; }

    // Conformance trace: scaled embedding, every decoder-layer output, and final norm.
    struct Trace {
        mx::array              embed;
        std::vector<mx::array> layerOut;  // size numHiddenLayers
        mx::array              finalNorm;
    };
    Trace forwardTrace(const std::vector<int> & tokens, ESKVCache * cache, int pastLen,
                       const ESSoftTokens * soft = nullptr) const;

    // Run a single decoder layer in isolation on a given input (no cache, pastLen 0).
    // Used for per-layer conformance: feed the golden previous-layer output, compare the output
    // to that layer's golden — isolates real bugs from accumulated bf16 drift.
    mx::array isolatedLayer(int layerIdx, const mx::array & xIn) const;

    const ESEmbedding & embedding() const { return embed_; }
    const ESModelConfig & config() const { return config_; }

    // Additive f32 mask [seqQ, seqK]; sliding adds the window lower bound.
    // Public so conformance probes can build the SAME mask the forward pass uses — a probe
    // that reconstructs its own can silently diverge and then gates on the wrong thing.
    // blockIds (optional, sliding layers, config.useBidirectionalVisionAttention): per query/key
    // position of THIS forward, the image block index or -1 — keys in the same block as the query
    // are allowed regardless of causality (create_masks_for_vision_model: AND(sliding, OR(causal,
    // blockwise))). Global layers stay causal. Keys from earlier forwards (pastLen) carry no block
    // id, so an image block split across chunked-prefill boundaries loses its overlay there.
    mx::array buildMask(int seqQ, int pastLen, bool sliding, const std::vector<int> * blockIds = nullptr) const;

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
    // [vocab_per_layer, num_layers*ple] — an ESEmbedding so the table can be bundle-quantized
    // (config.quantPleBits / bundle ple_bits); lookup() gathers rows and dequantizes.
    std::unique_ptr<ESEmbedding> pleTable_;
    mx::array embedPerLayerScaleArr_;    // bf16 sqrt(ple)
    mx::array perLayerModelProjection_;  // [num_layers*ple, hidden]
    mx::array perLayerProjScaleArr_;     // bf16 1/sqrt(hidden)
    mx::array perLayerInputScaleArr_;    // bf16 1/sqrt(2)
    std::unique_ptr<ESRMSNorm> perLayerProjectionNorm_;
    mx::array computePerLayerInputs(const std::vector<int> & tokens, const mx::array & scaledEmbed) const;
    // Same math from an on-device int32 [seq] id array (compiled-step / lazy paths).
    mx::array computePerLayerInputs(const mx::array & ids, const mx::array & scaledEmbed) const;

};

}  // namespace es
