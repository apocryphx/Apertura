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

#pragma mark - ESCompiledStep (P3: whole-step compiled decode)

static int roundUpTo(int n, int step) { return ((n + step - 1) / step) * step; }

ESCompiledStep::ESCompiledStep(const ESGemma4TextForCausalLM & lm, ESKVCache * cache, int pos,
                               bool cachePrealloc)
    : lm_(lm), cache_(cache), pos_(pos) {
    const ESModelConfig & cfg = lm.config();
    if (!cfg.fused || cfg.hasPLE() || cfg.enableMoeBlock || cfg.quantKVBits > 0 || !cfg.slidingWindowCache)
        throw std::runtime_error("ESCompiledStep: dense fused bf16-KV models with the sliding cache only");
    window_  = cfg.slidingWindow;
    nLayers_ = cfg.numHiddenLayers;
    capS_    = window_ + kSlidingHeadroom;
    capG_    = roundUpTo(pos + 1, kGlobalChunk);
    firstS_ = firstG_ = -1;
    for (int i = 0; i < nLayers_; ++i) {
        if (cfg.isSliding(i)) { if (firstS_ < 0) firstS_ = i; }
        else                  { if (firstG_ < 0) firstG_ = i; }
    }

    // Adopt the prefilled cache into step layout: fixed-capacity slot buffers, content at the
    // front. Sliding layers keep only the last `window` positions (older ones are already
    // mask-dead — softmax weight exactly 0 — so dropping them is output-identical).
    slidingBase_ = pos;
    for (int i = 0; i < nLayers_; ++i) {
        auto kv = cache->current(i, cachePrealloc);
        const int kvH = kv.first.shape(0), len = kv.first.shape(1), hd = kv.first.shape(2);
        const bool sliding = cfg.isSliding(i);
        const int content = sliding ? std::min(len, window_) : len;
        const int cap = sliding ? capS_ : capG_;
        mx::array k = kv.first, v = kv.second;
        if (content < len) {  // sliding: keep the newest `content` positions
            k = mx::slice(k, {0, len - content, 0}, {kvH, len, hd});
            v = mx::slice(v, {0, len - content, 0}, {kvH, len, hd});
        }
        mx::array bk = mx::zeros({kvH, cap, hd}, k.dtype());
        mx::array bv = mx::zeros({kvH, cap, hd}, v.dtype());
        if (content > 0) {
            bk = mx::slice_update(bk, k, {0, 0, 0}, {kvH, content, hd});
            bv = mx::slice_update(bv, v, {0, 0, 0}, {kvH, content, hd});
        }
        cache->setSlot(i, std::move(bk), std::move(bv));
        if (sliding) slidingBase_ = pos - content;  // same for every sliding layer
    }

    // Record the whole step once. Everything position-dependent is computed ON DEVICE from the
    // int32 [1] inputs (pos, slidingBase), so the recorded graph replays for every token; only a
    // capacity change (new input shapes) re-traces, via mx::compile's shape-keyed cache.
    auto body = [this](const std::vector<mx::array> & in) -> std::vector<mx::array> {
        const ESModelConfig & cfg = lm_.config();
        const mx::array & token = in[0];  // int32 [1]
        const mx::array & pos   = in[1];  // int32 [1] — absolute position of this token
        const mx::array & base  = in[2];  // int32 [1] — absolute position of sliding slot 0
        cache_->beginStep(/*globalIdx=*/pos, /*slidingIdx=*/mx::subtract(pos, base));
        for (int i = 0; i < nLayers_; ++i)
            cache_->setSlot(i, in[3 + 2 * i], in[3 + 2 * i + 1]);
        const int capS = in[3 + 2 * firstS_].shape(1);
        const int capG = in[3 + 2 * firstG_].shape(1);

        // RoPE cos/sin from pos — same f32 math as ESRotaryEmbedding::cosSin (bit-identical).
        auto cs = [&](const std::vector<float> & invF) {
            mx::array posF  = mx::reshape(mx::astype(pos, mx::float32), {1, 1});
            mx::array inv   = mx::array(invF.data(), {1, (int) invF.size()}, mx::float32);
            mx::array freqs = mx::multiply(posF, inv);                 // [1, half]
            mx::array emb   = mx::concatenate({freqs, freqs}, -1);     // [1, headDim]
            return std::make_pair(mx::astype(mx::cos(emb), cfg.computeDtype),
                                  mx::astype(mx::sin(emb), cfg.computeDtype));
        };
        auto lcs = cs(lm_.model().localRope().invFreq());
        auto gcs = cs(lm_.model().globalRope().invFreq());

        // Additive masks over slot indices — same values as buildMask (0 / -1e30 f32), plus
        // -1e30 on unwritten slots (they hold zeros; weight underflows to exactly 0).
        //   global  slot j: abs = j;        visible iff abs <= pos
        //   sliding slot j: abs = base + j; visible iff abs <= pos && abs > pos - window
        auto mkMask = [&](int cap, bool sliding) {
            mx::array j     = mx::arange(0, cap, mx::int32);           // [cap]
            mx::array abs   = sliding ? mx::add(j, base) : j;
            mx::array valid = mx::less_equal(abs, pos);
            if (sliding)
                valid = mx::logical_and(valid,
                        mx::greater(abs, mx::subtract(pos, mx::array(window_, mx::int32))));
            mx::array m = mx::where(valid, mx::array(0.0f), mx::array(-1e30f));
            return mx::reshape(mx::astype(m, mx::float32), {1, cap});
        };
        mx::array maskS = mkMask(capS, true);
        mx::array maskG = mkMask(capG, false);

        mx::array logits = lm_.stepLogits(token, lcs, gcs, maskS, maskG, cache_);  // [vocab]

        std::vector<mx::array> outs;
        outs.reserve(1 + 2 * nLayers_);
        outs.push_back(logits);
        for (int i = 0; i < nLayers_; ++i) {
            outs.push_back(cache_->slotK(i));
            outs.push_back(cache_->slotV(i));
        }
        cache_->endStep();
        return outs;
    };
    fn_ = mx::compile(std::function<std::vector<mx::array>(const std::vector<mx::array> &)>(body));
}

// Pre-step maintenance, outside the compiled graph (rare, eager):
//  - sliding compaction: when the write slot would hit capacity, keep the last `window` positions
//    at the front. Capacity is unchanged, so the recorded graph stays valid (no re-trace).
//  - global growth: when pos hits capacity, re-home into a kGlobalChunk-larger buffer. New input
//    shapes -> mx::compile re-traces once per chunk.
void ESCompiledStep::maintain() {
    const ESModelConfig & cfg = lm_.config();
    if (pos_ - slidingBase_ == capS_) {
        for (int i = 0; i < nLayers_; ++i) {
            if (!cfg.isSliding(i)) continue;
            const mx::array & k = cache_->slotK(i);
            const int kvH = k.shape(0), hd = k.shape(2);
            mx::array nk = mx::zeros({kvH, capS_, hd}, k.dtype());
            mx::array nv = mx::zeros({kvH, capS_, hd}, k.dtype());
            nk = mx::slice_update(nk, mx::slice(cache_->slotK(i), {0, capS_ - window_, 0}, {kvH, capS_, hd}),
                                  {0, 0, 0}, {kvH, window_, hd});
            nv = mx::slice_update(nv, mx::slice(cache_->slotV(i), {0, capS_ - window_, 0}, {kvH, capS_, hd}),
                                  {0, 0, 0}, {kvH, window_, hd});
            cache_->setSlot(i, std::move(nk), std::move(nv));
        }
        slidingBase_ = pos_ - window_;
    }
    if (pos_ == capG_) {
        const int newCap = capG_ + kGlobalChunk;
        for (int i = 0; i < nLayers_; ++i) {
            if (cfg.isSliding(i)) continue;
            const mx::array & k = cache_->slotK(i);
            const int kvH = k.shape(0), oldCap = k.shape(1), hd = k.shape(2);
            mx::array nk = mx::zeros({kvH, newCap, hd}, k.dtype());
            mx::array nv = mx::zeros({kvH, newCap, hd}, k.dtype());
            nk = mx::slice_update(nk, cache_->slotK(i), {0, 0, 0}, {kvH, oldCap, hd});
            nv = mx::slice_update(nv, cache_->slotV(i), {0, 0, 0}, {kvH, oldCap, hd});
            cache_->setSlot(i, std::move(nk), std::move(nv));
        }
        capG_ = newCap;
    }
}

mx::array ESCompiledStep::step(int prevToken) {
    maintain();
    std::vector<mx::array> ins;
    ins.reserve(3 + 2 * nLayers_);
    int tok = prevToken, p = pos_, b = slidingBase_;
    ins.push_back(mx::array(&tok, {1}, mx::int32));
    ins.push_back(mx::array(&p, {1}, mx::int32));
    ins.push_back(mx::array(&b, {1}, mx::int32));
    // Move the slot buffers into the inputs (sole reference) so the compiled call can donate
    // them to the scatter appends — in-place writes, no per-token buffer traffic.
    for (int i = 0; i < nLayers_; ++i) {
        ins.push_back(cache_->takeSlotK(i));
        ins.push_back(cache_->takeSlotV(i));
    }
    std::vector<mx::array> outs = fn_(ins);
    for (int i = 0; i < nLayers_; ++i)
        cache_->setSlot(i, outs[1 + 2 * i], outs[1 + 2 * i + 1]);
    pos_ += 1;
    return outs[0];  // [vocab], lazy
}

}  // namespace es
