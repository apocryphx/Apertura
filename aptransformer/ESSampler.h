#pragma once
//  ESSamplingConfig / ESSampler — greedy (argmax) plus temperature / top-k / top-p.
//  Phase-1 conformance uses greedy; the sampled path is here for the generation loop.
#include "mlx/mlx.h"

namespace es {
namespace mx = mlx::core;

struct ESSamplingConfig {
    float temperature  = 1.0f;
    float topP         = 0.95f;
    int   topK         = 64;
    bool  greedy       = true;
    int   maxNewTokens = 20;
    int   eosTokenId   = 1;    // Gemma-4 eos
    unsigned long long seed = 0;
};

class ESSampler {
public:
    explicit ESSampler(const ESSamplingConfig & cfg) : cfg_(cfg) {}

    // logits: [vocab]. Returns a token id.
    int sample(const mx::array & logits) const;

    static int argmax(const mx::array & logits);

private:
    ESSamplingConfig cfg_;
};

}  // namespace es
