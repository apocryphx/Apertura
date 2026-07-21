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

// ── ESCompiledStep — P3 prototype: the WHOLE per-token decode step as one mx::compile'd graph.
//
// Motivation (PERFORMANCE_ROADMAP.md P3): eager decode re-traces and dispatches ~86-108 kernels
// per token; the GPU sits ~13% idle at short context and ~half idle at 4K — the dominant residual
// decode cost is CPU graph-rebuild/dispatch serialization. This class records the full step ONCE —
// embed -> RoPE (computed on-device from the position) -> both masks (computed on-device from the
// position over slot indices) -> 60 layers with scatter cache appends -> final norm -> LM head +
// softcap — and replays it every token. Per-token host work collapses to three int32 [1] uploads
// (token, pos, slidingBase) and the argmax readback.
//
// Cache layout: adopts a PREFILLED ESKVCache — every layer is re-homed into a fixed-capacity slot
// buffer (sliding: last `window` positions, capacity window+kSlidingHeadroom; global: all
// positions, capacity rounded up to kGlobalChunk). Appends scatter at a position ARRAY (data, not
// a baked int), attention reads the full-capacity buffers, and the additive masks kill unwritten/
// expired slots — their softmax weight is exactly 0 (the P1 bit-exactness argument), so outputs
// are identical to sliced attention. Between steps, maintain() compacts the sliding window (every
// kSlidingHeadroom tokens; capacity unchanged, so no re-trace) and grows global capacity (every
// kGlobalChunk tokens; shapes change, mx::compile re-traces via its shape-keyed cache).
//
// Scope: dense fused non-PLE non-MoE bf16-KV models with the sliding cache enabled (throws
// otherwise). Verified token-exact vs the eager path via --step-verify.
class ESCompiledStep {
public:
    static constexpr int kSlidingHeadroom = 256;   // sliding compaction period (cap = window + this)
    static constexpr int kGlobalChunk     = 1024;  // global capacity step (re-trace period)

    // Adopt `cache` (already prefilled through the normal path with `pos` tokens). cachePrealloc
    // must say which mode filled it (config.preallocKVCache of the LM that ran the prefill).
    ESCompiledStep(const ESGemma4TextForCausalLM & lm, ESKVCache * cache, int pos, bool cachePrealloc);

    // One decode step: previous sampled token id -> last-position logits [vocab] (lazy).
    mx::array step(int prevToken);

    int position() const { return pos_; }

private:
    void maintain();  // pre-step compaction/growth (rare, eager, outside the compiled graph)

    const ESGemma4TextForCausalLM & lm_;
    ESKVCache * cache_;
    int pos_, slidingBase_, capS_, capG_, window_, nLayers_, firstS_, firstG_;
    std::function<std::vector<mx::array>(const std::vector<mx::array> &)> fn_;
};

}  // namespace es
