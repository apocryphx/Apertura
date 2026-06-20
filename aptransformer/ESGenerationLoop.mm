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

#pragma mark - ESSession (persistent prefix-cached multi-turn)

void ESSession::prime(const std::vector<int> & prefixTokens) {
    if (prefixTokens.empty()) return;
    // One prefill into the persistent cache; discard the logits (we only want the K/V populated).
    mx::array ll = lm_.lastLogits(prefixTokens, &cache_, pos_);
    mx::eval(ll);
    pos_ += (int) prefixTokens.size();
}

std::vector<int> ESSession::respond(const std::vector<int> & turnTokens, const ESSamplingConfig & cfg) {
    ESSampler sampler(cfg);
    std::vector<int> out;

    // Prefill ONLY the new turn after the already-cached prefix/history (no re-prefill of the past).
    mx::array ll = lm_.lastLogits(turnTokens, &cache_, pos_);
    mx::eval(ll);
    pos_ += (int) turnTokens.size();
    int next = sampler.sample(ll);
    out.push_back(next);

    for (int s = 1; s < cfg.maxNewTokens; ++s) {
        if (next == cfg.eosTokenId) break;
        ll = lm_.lastLogits({next}, &cache_, pos_);    // appends this token's K/V to the cache
        mx::eval(ll);
        pos_ += 1;
        next = sampler.sample(ll);
        out.push_back(next);
    }

    // Cache the final sampled token too, so the next turn attends to the complete reply.
    if (!out.empty()) { mx::array t = lm_.lastLogits({out.back()}, &cache_, pos_); mx::eval(t); pos_ += 1; }
    return out;
}

}  // namespace es
