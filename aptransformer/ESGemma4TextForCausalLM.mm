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

// Project ONLY the last position's hidden vector through the tied LM head + softcap -> [vocab].
// Slicing hidden before the LM head is bit-identical to projecting all rows then slicing (the head
// is an independent per-position matmul), but the logits tensor is [1, vocab] instead of
// [seq, vocab] — so prefill memory no longer scales with prompt length × vocab (262k). This is the
// difference between a bounded ~0.5 MB last-row projection and a multi-GB full-sequence one at long
// context (e.g. a ~10k-token system prompt).
static mx::array projectLast(const ESGemma4TextModel & model, float softcap, const mx::array & hidden) {
    const int seq = hidden.shape(0), hd = hidden.shape(1);
    mx::array lastH  = mx::reshape(mx::slice(hidden, {seq - 1, 0}, {seq, hd}), {1, hd});  // [1, hidden]
    mx::array logits = model.embedding().logits(lastH);                                   // [1, vocab]
    if (softcap > 0.0f) {
        mx::array cap = lit(softcap, logits);
        logits = mx::multiply(mx::tanh(mx::divide(logits, cap)), cap);
    }
    return mx::reshape(logits, {logits.shape(1)});                                        // [vocab]
}

// Convenience for generation: only the last position's logits are needed to pick the next token, so
// run the decoder stack (fills the KV cache for every position) but apply the LM head to the last
// position alone. For single-token decode `seq == 1` this is already minimal; the win is at prefill.
mx::array ESGemma4TextForCausalLM::lastLogits(const std::vector<int> & tokens, ESKVCache * cache, int pastLen) const {
    // Chunked prefill (P5, config.prefillChunk > 0): run the prompt through the stack in
    // chunk-sized forwards instead of one L-length forward. Combined with the sliding-layer
    // trim in ESAttention (window + chunk keys per append), 50/60 layers' prefill attention
    // drops from O(L^2) to O(L*(window+chunk)), and the composite-path score/mask transients
    // are bounded at O(chunk*ctx) instead of O(L^2). Each chunk is evaluated before the next
    // (bounds live memory; the cache carries all state). Requires a cache (the chunks
    // communicate through it); cache-less callers keep the single forward.
    const int chunk = config_.prefillChunk, seq = (int) tokens.size();
    if (chunk > 0 && cache && seq > chunk) {
        int done = 0;
        while (seq - done > chunk) {
            std::vector<int> part(tokens.begin() + done, tokens.begin() + done + chunk);
            mx::array h = model_.forward(part, cache, pastLen + done);
            mx::eval(h);
            done += chunk;
        }
        std::vector<int> tail(tokens.begin() + done, tokens.end());
        mx::array hidden = model_.forward(tail, cache, pastLen + done);
        return projectLast(model_, softcap_, hidden);
    }
    mx::array hidden = model_.forward(tokens, cache, pastLen);   // [seq, hidden] — full stack, cache filled
    return projectLast(model_, softcap_, hidden);               // LM head on last position only -> [vocab]
}

// On-device single-token decode: takes the previous step's sampled id as an int32 [1] device array,
// returns last-position logits [vocab] — no host readback, so consecutive steps stay lazy and can
// overlap under mx::async_eval. Non-PLE only (delegates to model_.forwardDev).
mx::array ESGemma4TextForCausalLM::lastLogitsDev(const mx::array & tokenId, ESKVCache * cache, int pastLen,
                                                 bool compiledTail) const {
    mx::array hidden = model_.forwardDev(tokenId, cache, pastLen, compiledTail);  // [seq, hidden], seq==1
    return projectLast(model_, softcap_, hidden);                                 // [vocab]
}

// Compiled-step decode: the traceable top of one token step (embed -> 60 layers -> final norm ->
// last-position LM head + softcap). All position dependence is in the argument arrays.
mx::array ESGemma4TextForCausalLM::stepLogits(const mx::array & tokenId,
                                              const std::pair<mx::array, mx::array> & localCS,
                                              const std::pair<mx::array, mx::array> & globalCS,
                                              const mx::array & maskSliding, const mx::array & maskFull,
                                              ESKVCache * cache) const {
    mx::array hidden = model_.forwardStep(tokenId, localCS, globalCS, maskSliding, maskFull, cache);
    return projectLast(model_, softcap_, hidden);  // [vocab]
}

}  // namespace es
