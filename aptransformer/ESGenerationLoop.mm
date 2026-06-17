#include "ESGenerationLoop.h"
#include "ESKVCache.h"

namespace es {

//  The autoregressive loop that turns the LLM function into text. It owns one ESKVCache for the
//  whole generation and drives two phases (mirrors HuggingFace `generate`):
//    PREFILL — push the entire prompt through the model in one forward; the cache fills with K/V
//              for every prompt position; sample the first new token from the last position.
//    DECODE  — feed back one token at a time; each step appends one position to the cache and
//              attends over the full history, so per-step cost is O(1 new token), not O(seq).
//  `pos` tracks how many positions are already cached (the RoPE offset + mask anchor). Sampling
//  (greedy / temperature / top-k / top-p) is delegated to ESSampler; stop on eos or maxNewTokens.

std::vector<int> ESGenerationLoop::generate(const std::vector<int> & promptTokens) const {
    std::vector<int> out;
    ESKVCache cache(lm_.config().numHiddenLayers);

    int pos = 0;  // positions already in the cache

    // ---- prefill: whole prompt -> cache populated, logits for the last position ----
    mx::array ll = lm_.lastLogits(promptTokens, &cache, pos);
    pos += (int) promptTokens.size();
    int next = sampler_.sample(ll);
    out.push_back(next);

    // ---- decode: one token at a time against the growing cache ----
    for (int s = 1; s < cfg_.maxNewTokens; ++s) {
        if (next == cfg_.eosTokenId) break;            // end-of-turn / eos -> stop
        ll = lm_.lastLogits({next}, &cache, pos);      // single-token forward, reuses cached past
        pos += 1;
        next = sampler_.sample(ll);
        out.push_back(next);
    }
    return out;
}

}  // namespace es
