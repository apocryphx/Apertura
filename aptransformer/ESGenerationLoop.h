#pragma once
//  ESGenerationLoop — prefill + decode with the KV cache (mirrors HF generate).
#include "mlx/mlx.h"
#include "ESGemma4TextForCausalLM.h"
#include "ESSampler.h"
#include "ESKVCache.h"
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

// Persistent multi-turn session over ONE long-lived KV cache. prime() prefills a constant prefix
// (e.g. the persona) once; each respond() appends only the new turn's tokens and decodes, reusing
// all prior K/V — so the persona/history is never re-prefilled. The continuation is byte-identical
// to a from-scratch forward over the concatenated tokens (same RoPE offset + mask anchor via pos_).
//
// THIS IS THE DOMINANT PERFORMANCE WIN for long-prompt / multi-turn use (bigger than every weight/
// KV quant lever combined). A 13.5K-token persona re-prefills in ~128 s EVERY turn (prefill is
// O(L^2), ~104 tok/s at that length); primed once via prime() it is ~3.8 s/turn after — 33.7x.
// Integration: a harness builds the persona prefix (chat-template system block), calls prime() at
// session start, then per turn builds the chat-template delta and calls respond(). The reference
// pattern is AperturaResearch --chat-session; correctness is gated by --session-verify (byte-ident).
class ESSession {
public:
    explicit ESSession(const ESGemma4TextForCausalLM & lm)
        : lm_(lm), cache_(lm.config().numHiddenLayers) {}

    // Prefill a constant prefix once (no generation). Call at session start with the persona.
    void prime(const std::vector<int> & prefixTokens);

    // Append `turnTokens` after the cached context and decode a reply (ids only). The reply's K/V
    // (and the turn's) remain cached for the next call.
    std::vector<int> respond(const std::vector<int> & turnTokens, const ESSamplingConfig & cfg);

    int  position() const { return pos_; }     // tokens currently in the cache
    void reset() { cache_.reset(); pos_ = 0; }

private:
    const ESGemma4TextForCausalLM & lm_;
    ESKVCache cache_;
    int       pos_ = 0;
};

}  // namespace es
