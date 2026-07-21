#pragma once
//  ESGemma4TextForCausalLM — text model + tied LM head + final-logit soft-capping.
//
//   hidden = model(tokens)
//   logits = hidden @ embed_tokens.weight^T          (tie_word_embeddings)
//   logits = tanh(logits / cap) * cap                (final_logit_softcapping = 30)
#include "mlx/mlx.h"
#include "ESModelConfig.h"
#include "ESGemma4TextModel.h"
#include "ESKVCache.h"
#include <memory>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

class ESGemma4TextForCausalLM {
public:
    ESGemma4TextForCausalLM(const ESModelConfig & config, const ESWeightLoader & weights);

    // Returns logits [seq, vocab].
    mx::array forward(const std::vector<int> & tokens, ESKVCache * cache, int pastLen) const;

    // Logits for the last position only [vocab].
    mx::array lastLogits(const std::vector<int> & tokens, ESKVCache * cache, int pastLen) const;

    // On-device single-token decode: previous sampled id as int32 [1] device array -> last logits
    // [vocab], no host readback (non-PLE only). For the async/overlapped decode loop.
    // compiledTail=true routes layers through their compiled stateless tail (mx::compile prototype).
    mx::array lastLogitsDev(const mx::array & tokenId, ESKVCache * cache, int pastLen,
                            bool compiledTail = false) const;

    // Compiled-step decode (P3): traceable single-token step — cos/sin/masks are parameters,
    // cache must be in step mode. Returns last-position logits [vocab]. See ESCompiledStep.
    mx::array stepLogits(const mx::array & tokenId,
                         const std::pair<mx::array, mx::array> & localCS,
                         const std::pair<mx::array, mx::array> & globalCS,
                         const mx::array & maskSliding, const mx::array & maskFull,
                         ESKVCache * cache) const;

    const ESGemma4TextModel & model() const { return model_; }
    const ESModelConfig & config() const { return config_; }

private:
    ESModelConfig     config_;
    ESGemma4TextModel model_;
    float             softcap_;
};

}  // namespace es
