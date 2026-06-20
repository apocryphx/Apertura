#pragma once
//  ESWeightLoader — HF safetensors (sharded) -> mx::array, cast once to computeDtype.
//
//  Reads model.safetensors.index.json, loads each referenced shard via
//  mx::load_safetensors, strips the `model.language_model.` text-decoder prefix,
//  and casts every tensor to config.computeDtype. Vision/audio weights are ignored.
//  Tied embeddings: there is no lm_head weight; the LM head reuses embed_tokens.
#include "mlx/mlx.h"
#include "ESModelConfig.h"
#include "ESLinear.h"
#include "ESEmbedding.h"
#include "ESExperts.h"
#include <string>
#include <unordered_map>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader {
public:
    // modelDir: the HF snapshot directory (contains config.json, the shards, index json).
    ESWeightLoader(const std::string & modelDir, const ESModelConfig & config);

    bool has(const std::string & name) const { return weights_.count(name) > 0; }

    // Throws if missing. Names are the text-decoder-relative keys, e.g.
    //   "embed_tokens.weight", "norm.weight",
    //   "layers.7.self_attn.q_proj.weight", "layers.7.layer_scalar", ...
    const mx::array & get(const std::string & name) const;

    // Convenience for per-layer weights.
    const mx::array & layer(int idx, const std::string & suffix) const;

    size_t count() const { return weights_.size(); }

    // Read-only view of every loaded (text-decoder-relative) tensor, for bulk
    // operations like quantized-bundle export.
    const std::unordered_map<std::string, mx::array> & all() const { return weights_; }

    // --- .apml bundle (reload) mode ---------------------------------------
    // True when this loader was built from an .apml package (pre-quantized variant)
    // rather than an HF snapshot. In bundle mode tensors are stored verbatim
    // (packed weights stay uint32 — never cast to computeDtype).
    bool isBundle() const { return isBundle_; }
    // A tensor is pre-quantized iff its companion `<name>.scales` is present.
    bool hasQuantized(const std::string & name) const { return weights_.count(name + ".scales") > 0; }
    struct QuantTriple { mx::array weight, scales, biases; };
    QuantTriple quantized(const std::string & name) const;
    int bundleBits() const { return bundleBits_; }
    int bundleGroupSize() const { return bundleGroupSize_; }
    int bundleEmbedBits() const { return bundleEmbedBits_; }

    std::string layerKey(int idx, const std::string & suffix) const {
        return "layers." + std::to_string(idx) + "." + suffix;
    }

private:
    void loadHF(const std::string & modelDir, const ESModelConfig & config);
    void loadBundle(const std::string & packageDir, const ESModelConfig & config);

    std::unordered_map<std::string, mx::array> weights_;
    bool isBundle_         = false;
    int  bundleBits_       = 0;
    int  bundleGroupSize_  = 64;
    int  bundleEmbedBits_  = 0;
};

// --- Layer factories --------------------------------------------------------
// Build a layer for `name`: pre-quantized from the bundle when present, else the
// bf16 path (quantize-now iff the config bits > 0). These are the single point
// that routes the reload-vs-quantize-now decision, so construction sites stay simple.
ESLinear    esMakeLinear   (const ESWeightLoader & w, const std::string & name,
                            int quantBits, int groupSize);
ESEmbedding esMakeEmbedding (const ESWeightLoader & w, const std::string & name,
                            int quantEmbedBits, int groupSize);
ESExperts   esMakeExperts  (const ESWeightLoader & w, const std::string & gateUpName,
                            const std::string & downName, int quantBits, int groupSize);

// --- Quantized .apml bundle export -----------------------------------------
//
// Quantize an HF model snapshot and write a self-describing `.apml` package (a
// macOS document package — see BUNDLE.md). Quantizes exactly the projections the
// runtime quantizes (q/k/v/o, gate/up/down, MoE experts) at `bits`, the token
// embedding at `embedBits`, and leaves norms/scalars/router/PLE weights bf16.
// The package is assembled in a temp directory and moved into place atomically.
struct ESBundleExportOptions {
    int bits       = 4;          // layer-projection quant bits (0 = keep bf16)
    int groupSize  = 64;         // affine group size
    int embedBits  = 8;          // token-embedding / tied-head bits (0 = keep bf16)
    std::string variantId      = "mlx-q4";
    std::string sourceModelId;   // provenance (optional)
    std::string sourceRevision;  // provenance (optional)
};

// Returns true on success. On failure returns false and, if `error` is non-null,
// sets it to a human-readable message.
bool exportQuantizedBundle(const std::string & modelDir,
                           const std::string & outPackagePath,
                           const ESBundleExportOptions & opts,
                           std::string * error = nullptr);

}  // namespace es
