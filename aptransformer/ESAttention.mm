#include "ESAttention.h"
#include "ESWeightLoader.h"
#include "ESRotaryEmbedding.h"
#include "ESDecodeAttn.h"
#include "ESOps.h"

#include <string>

namespace es {

static ESRMSNorm makeNorm(const ESWeightLoader & w, int layer, const std::string & name, float eps, bool fused) {
    return ESRMSNorm(w.layer(layer, name), eps, fused);
}

// Align an [seqQ, seqK_abs] mask to a (possibly sliding-window-evicted) K/V of length seqK: keep the
// last `seqK` columns, which correspond to the retained (newest) keys. No-op when lengths match.
static mx::array alignMask(const mx::array & maskF32, int seqK) {
    if (maskF32.size() == 0) return maskF32;
    const int mq = maskF32.shape(0), mk = maskF32.shape(1);
    if (mk == seqK) return maskF32;
    return mx::slice(maskF32, {0, mk - seqK}, {mq, mk});
}

ESAttention::ESAttention(const ESModelConfig & config, int layerIdx, const ESWeightLoader & weights)
    : fused_(config.fused),
      isKvShared_(config.isKvSharedLayer(layerIdx)),
      storeFullKv_(config.storeFullLengthKv(layerIdx)),
      quantKVBits_(config.quantKVBits),
      quantGroupSize_(config.quantGroupSize),
      // Per-layer-type quant-KV engagement: with quantKVGlobalOnly (default), only the global
      // (unwindowed) layers — where the depth-growing KV bytes live — take the quantized core;
      // sliding layers keep bf16 + the fused vector kernel. Shared-KV layers never quantize
      // (the quant core doesn't speak the shared-KV scratch).
      useQuantKV_(config.quantKVBits > 0 &&
                  (!config.quantKVGlobalOnly || !config.isSliding(layerIdx)) &&
                  !config.isKvSharedLayer(layerIdx) && !config.storeFullLengthKv(layerIdx)),
      // Raw-K engagement: global layers only (the depth-growing bytes), never shared-KV
      // or quantized layers. Sliding layers keep the k/v cache + fused vector kernel.
      useRawKV_(config.rawKV && config.quantKVBits == 0 &&
                !config.isSliding(layerIdx) &&
                !config.isKvSharedLayer(layerIdx) && !config.storeFullLengthKv(layerIdx)),
      useRawQ8_(config.rawKVQ8),
      rmsEps_(config.rmsNormEps),
      layerIdx_(layerIdx),
      numQHeads_(config.numAttentionHeads),
      numKVHeads_(config.kvHeadsFor(layerIdx)),
      headDim_(config.headDimFor(layerIdx)),
      groups_(config.numAttentionHeads / config.kvHeadsFor(layerIdx)),
      kEqV_(config.kEqVFor(layerIdx)),
      isSliding_(config.isSliding(layerIdx)),
      slidingWindow_(config.slidingWindow),
      slidingCache_(config.slidingWindowCache),
      preallocCache_(config.preallocKVCache),
      chunkedPrefill_(config.prefillChunk > 0),
      tiledKChunk_(config.tiledKChunk),
      scaling_(1.0f),
      qProj_(esMakeLinear(weights, weights.layerKey(layerIdx, "self_attn.q_proj.weight"), config.quantBits, config.quantGroupSize)),
      kProj_(esMakeLinear(weights, weights.layerKey(layerIdx, "self_attn.k_proj.weight"), config.quantBits, config.quantGroupSize)),
      oProj_(esMakeLinear(weights, weights.layerKey(layerIdx, "self_attn.o_proj.weight"), config.quantBits, config.quantGroupSize)),
      hasVProj_(false),
      qNorm_(makeNorm(weights, layerIdx, "self_attn.q_norm.weight", config.rmsNormEps, config.fused)),
      kNorm_(makeNorm(weights, layerIdx, "self_attn.k_norm.weight", config.rmsNormEps, config.fused)),
      vNorm_(config.rmsNormEps, config.fused) {  // v_norm: with_scale=false
    if (!kEqV_) {
        vProj_.emplace(esMakeLinear(weights, weights.layerKey(layerIdx, "self_attn.v_proj.weight"), config.quantBits, config.quantGroupSize));
        hasVProj_ = true;
    }
    if (useRawKV_) {
        // The global rotary table, built by the SAME ctor math as the model's rotary —
        // the decode kernel and the ops reconstruction both derive cos/sin from it.
        ESRotaryEmbedding rot(headDim_, config.ropeThetaGlobal, config.globalPartialRotaryFactor,
                              config.computeDtype);
        ropeInvFreq_ = rot.invFreq();
    }
}

std::pair<mx::array, mx::array> ESAttention::keyValue(const mx::array & x, const mx::array & cos,
                                                      const mx::array & sin, ESKVCache * cache,
                                                      ESSharedKV * sharedKV) const {
    // Shared (elastic) layers reuse the stored K/V of their type — no projection, no cache write.
    if (isKvShared_ && sharedKV) return sharedKV->get(isSliding_);

    const int seq = x.shape(0);
    mx::array kRaw = mx::reshape(kProj_.forward(x), {seq, numKVHeads_, headDim_});
    // value source is the PRE-NORM k_proj view when k_eq_v (global), else v_proj output.
    mx::array vRaw = hasVProj_
                         ? mx::reshape(vProj_->forward(x), {seq, numKVHeads_, headDim_})
                         : kRaw;
    mx::array k = mx::transpose(ESRotaryEmbedding::apply(kNorm_.forward(kRaw), cos, sin), {1, 0, 2});
    mx::array v = mx::transpose(vNorm_.forward(vRaw), {1, 0, 2});  // NO RoPE on value

    // Sliding-window eviction, sliding layers only, never for elastic shared-KV storing layers
    // (whose full K/V is reused by other layers). Decode (seq == 1): keep the last `window` keys.
    // Multi-token appends (prefill chunks / session turns) keep the full cache by default; with
    // chunked prefill (P5) they trim to the last (window + seq) keys — the oldest key any query
    // in this chunk can see is (past + 0) - window + 1, and every future query reaches back at
    // most `window` from a later position, so no dropped key is visible to anyone (mask weight
    // exactly 0). alignMask slices the mask to the retained keys either way.
    int maxKeep = 0;
    if (slidingCache_ && isSliding_ && !storeFullKv_ && !isKvShared_) {
        if (seq == 1)                maxKeep = slidingWindow_;
        else if (chunkedPrefill_)    maxKeep = slidingWindow_ + seq;
    }
    mx::array Kfull = k, Vfull = v;
    if (cache) { auto kv = cache->update(layerIdx_, k, v, maxKeep, preallocCache_); Kfull = kv.first; Vfull = kv.second; }
    if (storeFullKv_ && sharedKV) sharedKV->store(isSliding_, Kfull, Vfull);  // for shared layers to reuse
    return {Kfull, Vfull};
}

mx::array ESAttention::forward(const mx::array & x,
                               const mx::array & cos,
                               const mx::array & sin,
                               const mx::array & maskF32,
                               ESKVCache *       cache,
                               int               pastLen,
                               ESSharedKV *      sharedKV) const {
    // ── Attention path selection (see PERFORMANCE FINDINGS in ESModelConfig.h) ──
    // quantKVBits>0 takes the hand-rolled quantized-KV path, which FORGOES flash
    // (MLX 0.31.2 has no quantized SDPA — flash XOR quant-KV are mutually exclusive,
    // same as mlx-lm). Measured: flash+bf16-KV beats the two-call quantized path at
    // EVERY context <= 64K (prefill +19%, decode +4% at 1.5K), so quantKVBits is a
    // CAPACITY lever (fit a long KV cache in RAM), NOT a speed one — for speed leave
    // it 0 and keep `fused` on. (2026-07-21: an earlier note here claimed decode
    // "collapses to ~3.5 tok/s at 13.5K" — that was a measurement artifact; with the
    // preallocated cache + sliding eviction, clean decode is bandwidth-bound and nearly
    // flat in depth, see PERFORMANCE_ROADMAP.md §1/§6. A fused quantized-flash kernel
    // remains the lever only for >16K contexts where quant-KV must coexist with flash.)
    // The two-call path never wins on speed.
    if (useQuantKV_) return forwardQuantKV(x, cos, sin, maskF32, cache, pastLen);
    if (useRawKV_) return forwardRawKV(x, cos, sin, maskF32, cache, pastLen);
    // Tiled prefill core (see forwardTiled): prefill shapes only — decode (seq <= 8) keeps the
    // healthy sdpa_vector fused kernel, so tiled and fused compose rather than compete.
    if (tiledKChunk_ > 0 && x.shape(0) > 8) return forwardTiled(x, cos, sin, maskF32, cache, pastLen, sharedKV);
    if (fused_) return forwardFused(x, cos, sin, maskF32, cache, pastLen, sharedKV);

    const int seq = x.shape(0);

    // ---- query: proj -> [seq, heads, headDim] -> q_norm -> RoPE -> [heads, seq, headDim] ----
    mx::array q = qProj_.forward(x);
    q = mx::reshape(q, {seq, numQHeads_, headDim_});
    q = qNorm_.forward(q);
    q = ESRotaryEmbedding::apply(q, cos, sin);
    q = mx::transpose(q, {1, 0, 2});  // [numQ, seq, headDim]

    // ---- key/value (shared-KV aware) + GQA repeat ----
    auto kv = keyValue(x, cos, sin, cache, sharedKV);
    mx::array Kfull = kv.first, Vfull = kv.second;
    mx::array Krep = repeatKV(Kfull, groups_);  // [numQ, seqK, headDim]
    mx::array Vrep = repeatKV(Vfull, groups_);
    mx::array mask = alignMask(maskF32, Kfull.shape(1));  // align to evicted K/V (no-op otherwise)

    // ---- scores = Q @ K^T * scaling(1.0), mask, softmax(f32) ----
    mx::array scores = mx::matmul(q, mx::swapaxes(Krep, -1, -2));  // [numQ, seqQ, seqK]
    if (scaling_ != 1.0f) scores = mx::multiply(scores, lit(scaling_, scores));

    mx::array sf = mx::astype(scores, mx::float32);
    if (mask.size() > 0) sf = mx::add(sf, mask);  // broadcast [seqQ, seqK] over heads
    mx::array w = mx::softmax(sf, -1, /*precise=*/true);
    w = mx::astype(w, x.dtype());

    mx::array out = mx::matmul(w, Vrep);     // [numQ, seqQ, headDim]
    out = mx::transpose(out, {1, 0, 2});     // [seqQ, numQ, headDim]
    out = mx::reshape(out, {seq, numQHeads_ * headDim_});
    out = oProj_.forward(out);  // [seq, hidden]
    return out;
}

mx::array ESAttention::forwardFused(const mx::array & x, const mx::array & cos, const mx::array & sin,
                                    const mx::array & maskF32, ESKVCache * cache, int pastLen,
                                    ESSharedKV * sharedKV) const {
    const int seq = x.shape(0);

    mx::array q = qNorm_.forward(mx::reshape(qProj_.forward(x), {seq, numQHeads_, headDim_}));
    q = ESRotaryEmbedding::apply(q, cos, sin);
    q = mx::transpose(q, {1, 0, 2});  // [numQ, seq, headDim]

    auto kv = keyValue(x, cos, sin, cache, sharedKV);
    mx::array Kfull = kv.first, Vfull = kv.second;
    const int seqK = Kfull.shape(1);
    mx::array maskA = alignMask(maskF32, seqK);  // align to evicted K/V (no-op otherwise)

    // SDPA expects [B, heads, L, headDim]; GQA (numKV < numQ) handled internally.
    mx::array Q = mx::reshape(q,     {1, numQHeads_,  seq,  headDim_});
    mx::array K = mx::reshape(Kfull, {1, numKVHeads_, seqK, headDim_});
    mx::array V = mx::reshape(Vfull, {1, numKVHeads_, seqK, headDim_});
    mx::array M = mx::reshape(mx::astype(maskA, x.dtype()), {1, 1, seq, seqK});  // additive

    mx::array O = mx::fast::scaled_dot_product_attention(Q, K, V, scaling_, "", M);  // [1,numQ,seq,headDim]

    O = mx::transpose(mx::reshape(O, {numQHeads_, seq, headDim_}), {1, 0, 2});  // [seq, numQ, headDim]
    O = mx::reshape(O, {seq, numQHeads_ * headDim_});
    return oProj_.forward(O);
}

mx::array ESAttention::forwardTiled(const mx::array & x, const mx::array & cos, const mx::array & sin,
                                    const mx::array & maskF32, ESKVCache * cache, int pastLen,
                                    ESSharedKV * sharedKV) const {
    const int seq = x.shape(0);

    mx::array q = qNorm_.forward(mx::reshape(qProj_.forward(x), {seq, numQHeads_, headDim_}));
    q = ESRotaryEmbedding::apply(q, cos, sin);
    q = mx::transpose(q, {1, 0, 2});  // [numQ, seq, headDim]

    auto kv = keyValue(x, cos, sin, cache, sharedKV);
    mx::array Kfull = kv.first, Vfull = kv.second;  // [numKV, seqK, headDim]
    const int seqK = Kfull.shape(1);
    mx::array maskA = alignMask(maskF32, seqK);     // f32 additive [seq, seqK] (or empty)

    // GQA via broadcast: Q [nKV, groups, seq, d] against K/V [nKV, 1, chunk, d] — no repeatKV
    // copies (query heads are laid out kv-major, the same fact forwardQuantKV relies on).
    mx::array Q4 = mx::reshape(q, {numKVHeads_, groups_, seq, headDim_});

    // Online-softmax accumulators, all f32: running row max m, normalizer l, unnormalized
    // output acc. Same e-semantics as softmax(precise=true) — global max, f32 sum — reached
    // incrementally: each chunk rescales the running terms by exp(m_old - m_new).
    mx::array m = mx::array(0.0f), l = m, acc = m;  // placeholders until the first chunk
    const int C = tiledKChunk_;
    for (int c0 = 0; c0 < seqK; c0 += C) {
        const int c1 = std::min(c0 + C, seqK);
        mx::array Kc = mx::expand_dims(mx::slice(Kfull, {0, c0, 0}, {numKVHeads_, c1, headDim_}), 1);
        mx::array Vc = mx::expand_dims(mx::slice(Vfull, {0, c0, 0}, {numKVHeads_, c1, headDim_}), 1);

        mx::array Sf = mx::astype(mx::matmul(Q4, mx::swapaxes(Kc, -1, -2)), mx::float32);
        if (scaling_ != 1.0f) Sf = mx::multiply(Sf, mx::array(scaling_));
        if (maskA.size() > 0) Sf = mx::add(Sf, mx::slice(maskA, {0, c0}, {seq, c1}));

        // Row max, floored at a large negative finite value: a row whose every key in this chunk
        // is masked (-inf) would otherwise make m = -inf and exp(S - m) = exp(nan). With the
        // floor, exp(-inf - (-1e30)) = 0 — the chunk correctly contributes nothing to that row.
        mx::array mc = mx::maximum(mx::max(Sf, -1, /*keepdims=*/true), mx::array(-1e30f));
        if (c0 == 0) {
            m = mc;
            mx::array P = mx::exp(mx::subtract(Sf, m));
            l = mx::sum(P, -1, /*keepdims=*/true);
            acc = mx::astype(mx::matmul(mx::astype(P, x.dtype()), Vc), mx::float32);
        } else {
            mx::array mNew = mx::maximum(m, mc);
            mx::array corr = mx::exp(mx::subtract(m, mNew));
            mx::array P = mx::exp(mx::subtract(Sf, mNew));
            l = mx::add(mx::multiply(l, corr), mx::sum(P, -1, /*keepdims=*/true));
            acc = mx::add(mx::multiply(acc, corr),
                          mx::astype(mx::matmul(mx::astype(P, x.dtype()), Vc), mx::float32));
            m = mNew;
        }
    }

    mx::array O = mx::astype(mx::divide(acc, l), x.dtype());  // [nKV, groups, seq, headDim]
    O = mx::transpose(mx::reshape(O, {numQHeads_, seq, headDim_}), {1, 0, 2});
    O = mx::reshape(O, {seq, numQHeads_ * headDim_});
    return oProj_.forward(O);
}

mx::array ESAttention::forwardRawKV(const mx::array & x, const mx::array & cos, const mx::array & sin,
                                    const mx::array & maskF32, ESKVCache * cache, int pastLen) const {
    const int seq = x.shape(0);

    mx::array q = qNorm_.forward(mx::reshape(qProj_.forward(x), {seq, numQHeads_, headDim_}));
    q = ESRotaryEmbedding::apply(q, cos, sin);
    q = mx::transpose(q, {1, 0, 2});  // [numQ, seq, headDim]

    // The cache stores the PRE-NORM k_proj output — the single stream K and V derive from
    // (global layers: attention_k_eq_v, so V's source is this same tensor). With rawKVQ8
    // the stream is per-row affine u8; decode dequantizes inside the fused kernel, other
    // shapes dequantize by ops into the same reconstruction tail.
    mx::array kRaw = mx::transpose(mx::reshape(kProj_.forward(x), {seq, numKVHeads_, headDim_}),
                                   {1, 0, 2});  // [numKV, seq, headDim]

    mx::array rawView = kRaw;
    int seqK = seq;
    if (useRawQ8_) {
        if (cache) {
            ESKVCache::RawQ8 r = cache->updateRawQ8(layerIdx_, kRaw);
            if (seq == 1) {
                mx::array O = esDecodeAttnGlobalQ8(mx::reshape(q, {numQHeads_, headDim_}),
                                                   r.qbuf, r.scbuf, r.bsbuf, r.len, r.pitch,
                                                   kNorm_.weight(), rmsEps_, ropeInvFreq_);
                return oProj_.forward(mx::reshape(O, {1, numQHeads_ * headDim_}));
            }
            rawView = mx::astype(
                mx::add(mx::multiply(mx::astype(r.qview, mx::float32), r.sc), r.bs), x.dtype());
            seqK = r.len;
        } else {
            // Cacheless probe shapes: quantize+dequantize in place so the values match
            // what the cached path would have stored.
            auto qsb = ESKVCache::quantizeRawRows(kRaw);
            rawView = mx::astype(
                mx::add(mx::multiply(mx::astype(qsb[0], mx::float32), qsb[1]), qsb[2]), x.dtype());
        }
    } else {
        ESKVCache::Raw raw = cache
            ? cache->updateRaw(layerIdx_, kRaw, preallocCache_)
            : ESKVCache::Raw{kRaw, kRaw, seq, seq};
        seqK = raw.len;

        if (seq == 1) {
            // Fused decode kernel: streams each cached row once, recomputing K and V in
            // registers. Reads half the bytes of the k/v composite.
            mx::array O = esDecodeAttnGlobal(mx::reshape(q, {numQHeads_, headDim_}),
                                             raw.buffer, raw.len, raw.pitch,
                                             kNorm_.weight(), rmsEps_, ropeInvFreq_);
            return oProj_.forward(mx::reshape(O, {1, numQHeads_ * headDim_}));
        }
        rawView = raw.view;
    }

    // Prefill / multi-token append: reconstruct K and V from the full raw cache by ops
    // (the norms are deterministic per row, so recomputation is value-identical to having
    // cached them), then the fused SDPA — same tail as forwardFused.
    mx::array K = kNorm_.forward(rawView);   // [numKV, seqK, headDim]
    mx::array V = vNorm_.forward(rawView);

    // Full-length cos/sin for positions 0..seqK from the rotary table (f32 angles,
    // computeDtype rounding — the same construction as ESRotaryEmbedding::cosSin).
    const int half = headDim_ / 2;
    std::vector<float> pos(seqK);
    for (int i = 0; i < seqK; ++i) pos[i] = (float) i;
    mx::array freqs = mx::matmul(mx::array(pos.data(), {seqK, 1}, mx::float32),
                                 mx::array(ropeInvFreq_.data(), {1, half}, mx::float32));
    mx::array emb = mx::concatenate({freqs, freqs}, -1);                 // [seqK, headDim]
    mx::array c = mx::expand_dims(mx::astype(mx::cos(emb), x.dtype()), 0);  // [1, seqK, hd]
    mx::array s = mx::expand_dims(mx::astype(mx::sin(emb), x.dtype()), 0);
    K = mx::add(mx::multiply(K, c), mx::multiply(rotateHalf(K), s));

    mx::array Q = mx::reshape(q, {1, numQHeads_,  seq,  headDim_});
    mx::array Kf = mx::reshape(K, {1, numKVHeads_, seqK, headDim_});
    mx::array Vf = mx::reshape(V, {1, numKVHeads_, seqK, headDim_});
    mx::array M  = mx::reshape(mx::astype(maskF32, x.dtype()), {1, 1, seq, seqK});
    mx::array O  = mx::fast::scaled_dot_product_attention(Q, Kf, Vf, scaling_, "", M);

    O = mx::transpose(mx::reshape(O, {numQHeads_, seq, headDim_}), {1, 0, 2});
    O = mx::reshape(O, {seq, numQHeads_ * headDim_});
    return oProj_.forward(O);
}

mx::array ESAttention::forwardQuantKV(const mx::array & x, const mx::array & cos, const mx::array & sin,
                                      const mx::array & maskF32, ESKVCache * cache, int pastLen) const {
    const int seq = x.shape(0);
    const int gs = quantGroupSize_, bits = quantKVBits_;

    // q/k/v projections + QK/V-norm + RoPE (same as the other paths).
    mx::array q = qNorm_.forward(mx::reshape(qProj_.forward(x), {seq, numQHeads_, headDim_}));
    q = ESRotaryEmbedding::apply(q, cos, sin);
    q = mx::transpose(q, {1, 0, 2});  // [numQ, seq, headDim]

    mx::array kRaw = mx::reshape(kProj_.forward(x), {seq, numKVHeads_, headDim_});
    mx::array vRaw = hasVProj_
                         ? mx::reshape(vProj_->forward(x), {seq, numKVHeads_, headDim_})
                         : kRaw;
    mx::array k = mx::transpose(ESRotaryEmbedding::apply(kNorm_.forward(kRaw), cos, sin), {1, 0, 2});
    mx::array v = mx::transpose(vNorm_.forward(vRaw), {1, 0, 2});  // [numKV, seq, headDim]

    // Quantize + append to the cache (or quantize in place when there is no cache, e.g. prefill).
    ESKVCache::QKV qkv = cache
        ? cache->updateQuant(layerIdx_, k, v, gs, bits, preallocCache_)
        : [&] { auto kq = mx::quantize(k, gs, bits); auto vq = mx::quantize(v, gs, bits);
                return ESKVCache::QKV{kq[0], kq[1], kq[2], vq[0], vq[1], vq[2]}; }();
    const int seqK = qkv.kq.shape(1);
    const int nrep = numQHeads_ / numKVHeads_;

    // GQA: [1, nKV, nrep, seq, hd] queries against [1, nKV, 1, seqK, *] quantized K/V (mlx_lm pattern).
    mx::array Q = mx::reshape(q, {1, numKVHeads_, nrep, seq, headDim_});
    auto kdim = [&](const mx::array & a) { return mx::expand_dims(mx::reshape(a, {1, numKVHeads_, a.shape(1), a.shape(2)}), 2); };
    mx::array Kq = kdim(qkv.kq), Ks = kdim(qkv.ks), Kb = kdim(qkv.kb);
    mx::array Vq = kdim(qkv.vq), Vs = kdim(qkv.vs), Vb = kdim(qkv.vb);

    mx::array scores = mx::quantized_matmul(Q, Kq, Ks, Kb, /*transpose=*/true, gs, bits);  // [1,nKV,nrep,seq,seqK]
    mx::array sf = mx::astype(scores, mx::float32);
    if (maskF32.size() > 0) sf = mx::add(sf, maskF32);  // [seq, seqK] broadcasts
    mx::array w = mx::astype(mx::softmax(sf, -1, /*precise=*/true), x.dtype());

    mx::array O = mx::quantized_matmul(w, Vq, Vs, Vb, /*transpose=*/false, gs, bits);  // [1,nKV,nrep,seq,hd]
    O = mx::transpose(mx::reshape(O, {numQHeads_, seq, headDim_}), {1, 0, 2});  // [seq, numQ, headDim]
    O = mx::reshape(O, {seq, numQHeads_ * headDim_});
    return oProj_.forward(O);
}

}  // namespace es
