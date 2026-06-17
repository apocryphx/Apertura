#pragma once
//  ESWeightLoader — HF safetensors (sharded) -> mx::array, cast once to computeDtype.
//
//  Reads model.safetensors.index.json, loads each referenced shard via
//  mx::load_safetensors, strips the `model.language_model.` text-decoder prefix,
//  and casts every tensor to config.computeDtype. Vision/audio weights are ignored.
//  Tied embeddings: there is no lm_head weight; the LM head reuses embed_tokens.
#include "mlx/mlx.h"
#include "ESModelConfig.h"
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

private:
    std::unordered_map<std::string, mx::array> weights_;
};

}  // namespace es
