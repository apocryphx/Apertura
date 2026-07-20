#include "ESGemma4TextForCausalLM.h"
#include "ESWeightLoader.h"
#include "ESOps.h"

namespace es {

//  THE LLM FUNCTION.
//
//  This is the top of the model: the thing a generation loop calls to turn a sequence of token
//  ids into a probability distribution over the next token. Mathematically it is one pure function
//
//      f(tokens, cache_state) -> logits  ∈ ℝ^[seq, vocab]
//
//  built by composing three pieces, each owned elsewhere:
//    1. ESGemma4TextModel  — the decoder stack: scaled embedding -> 60 hybrid local/global
//                            attention+MLP layers (KV cache threaded through) -> final RMSNorm.
//                            Produces a `hidden` vector per position. This is where ~all the
//                            compute and all 58 GB of weights live.
//    2. The tied LM head   — Gemma ties the output projection to the input embedding matrix, so
//                            logits = hidden @ embed_tokens.weight^T. No separate lm_head weight
//                            exists; ESEmbedding::logits performs this (quantized_matmul when the
//                            embedding is quantized).
//    3. Final soft-cap     — Gemma bounds the logits into (-cap, +cap) via tanh(x/cap)*cap
//                            (cap = 30) before they are returned. This is applied to the logits
//                            themselves, NOT to the attention scores (Gemma-4 has no attn soft-cap).
//
//  Statefulness lives entirely in `cache` (ESKVCache, owned by the caller); the function itself is
//  stateless and deterministic — same (tokens, cache, pastLen) always yields the same logits. That
//  is what makes the whole forward pass inspectable and conformance-testable.

ESGemma4TextForCausalLM::ESGemma4TextForCausalLM(const ESModelConfig & config, const ESWeightLoader & weights)
    : config_(config), model_(config, weights), softcap_(config.finalLogitSoftcapping) {}

// Full forward: token ids -> next-token logits for every position. `cache` may be null for a
// stateless prefill; when present, the decoder layers append this step's K/V and attend over the
// whole history, so decode is O(1) new tokens against the cached past (pastLen = positions cached).
mx::array ESGemma4TextForCausalLM::forward(const std::vector<int> & tokens, ESKVCache * cache, int pastLen) const {
    // 1. Run the decoder stack -> final-norm hidden states, one vector per input position.
    mx::array hidden = model_.forward(tokens, cache, pastLen);          // [seq, hidden]

    // 2. Tied LM head: project each hidden vector onto the vocabulary (logits = hidden @ embed^T).
    mx::array logits = model_.embedding().logits(hidden);              // [seq, vocab]

    // 3. Final-logit soft-capping: tanh(logits / cap) * cap, squashing into (-cap, +cap).
    if (softcap_ > 0.0f) {
        mx::array cap = lit(softcap_, logits);
        logits = mx::multiply(mx::tanh(mx::divide(logits, cap)), cap);
    }
    return logits;
}

// Convenience for generation: only the last position's logits are needed to pick the next token.
// (We still compute all positions during prefill; for single-token decode `seq == 1`.)
mx::array ESGemma4TextForCausalLM::lastLogits(const std::vector<int> & tokens, ESKVCache * cache, int pastLen) const {
    mx::array logits = forward(tokens, cache, pastLen);
    int seq = logits.shape(0);
    return mx::reshape(mx::slice(logits, {seq - 1, 0}, {seq, logits.shape(1)}), {logits.shape(1)});
}

// On-device single-token decode: takes the previous step's sampled id as an int32 [1] device array,
// returns last-position logits [vocab] — no host readback, so consecutive steps stay lazy and can
// overlap under mx::async_eval. Non-PLE only (delegates to model_.forwardDev).
mx::array ESGemma4TextForCausalLM::lastLogitsDev(const mx::array & tokenId, ESKVCache * cache, int pastLen,
                                                 bool compiledTail) const {
    mx::array hidden = model_.forwardDev(tokenId, cache, pastLen, compiledTail);  // [seq, hidden], seq==1
    mx::array logits = model_.embedding().logits(hidden);            // [seq, vocab]
    if (softcap_ > 0.0f) {
        mx::array cap = lit(softcap_, logits);
        logits = mx::multiply(mx::tanh(mx::divide(logits, cap)), cap);
    }
    int seq = logits.shape(0);
    return mx::reshape(mx::slice(logits, {seq - 1, 0}, {seq, logits.shape(1)}), {logits.shape(1)});
}

}  // namespace es
