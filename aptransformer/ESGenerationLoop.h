#pragma once
//  ESGenerationLoop — prefill + decode with the KV cache (mirrors HF generate).
#include "mlx/mlx.h"
#include "ESGemma4TextForCausalLM.h"
#include "ESSampler.h"
#include <vector>

namespace es {
namespace mx = mlx::core;

class ESGenerationLoop {
public:
    ESGenerationLoop(const ESGemma4TextForCausalLM & lm, const ESSamplingConfig & cfg)
        : lm_(lm), cfg_(cfg), sampler_(cfg) {}

    // Returns the generated token ids (not including the prompt). Stops on eos or maxNewTokens.
    std::vector<int> generate(const std::vector<int> & promptTokens) const;

private:
    const ESGemma4TextForCausalLM & lm_;
    ESSamplingConfig                cfg_;
    ESSampler                       sampler_;
};

}  // namespace es
