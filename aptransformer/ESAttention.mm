#include "ESAttention.h"
#include "ESWeightLoader.h"
#include "ESRotaryEmbedding.h"
#include "ESOps.h"

#include <string>

namespace es {

static ESRMSNorm makeNorm(const ESWeightLoader & w, int layer, const std::string & name, float eps, bool fused) {
    return ESRMSNorm(w.layer(layer, name), eps, fused);
}

ESAttention::ESAttention(const ESModelConfig & config, int layerIdx, const ESWeightLoader & weights)
    : fused_(config.fused),
      isKvShared_(config.isKvSharedLayer(layerIdx)),
      storeFullKv_(config.storeFullLengthKv(layerIdx)),
      quantKVBits_(config.quantKVBits),
      quantGroupSize_(config.quantGroupSize),
      layerIdx_(layerIdx),
      numQHeads_(config.numAttentionHeads),
      numKVHeads_(config.kvHeadsFor(layerIdx)),
      headDim_(config.headDimFor(layerIdx)),
      groups_(config.numAttentionHeads / config.kvHeadsFor(layerIdx)),
      kEqV_(config.kEqVFor(layerIdx)),
      isSliding_(config.isSliding(layerIdx)),
      slidingWindow_(config.slidingWindow),
      scaling_(1.0f),
      qProj_(weights.layer(layerIdx, "self_attn.q_proj.weight"), config.quantBits, config.quantGroupSize),
      kProj_(weights.layer(layerIdx, "self_attn.k_proj.weight"), config.quantBits, config.quantGroupSize),
      oProj_(weights.layer(layerIdx, "self_attn.o_proj.weight"), config.quantBits, config.quantGroupSize),
      hasVProj_(false),
      qNorm_(makeNorm(weights, layerIdx, "self_attn.q_norm.weight", config.rmsNormEps, config.fused)),
      kNorm_(makeNorm(weights, layerIdx, "self_attn.k_norm.weight", config.rmsNormEps, config.fused)),
      vNorm_(config.rmsNormEps, config.fused) {  // v_norm: with_scale=false
    if (!kEqV_) {
        vProj_.emplace(weights.layer(layerIdx, "self_attn.v_proj.weight"), config.quantBits, config.quantGroupSize);
        hasVProj_ = true;
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

    mx::array Kfull = k, Vfull = v;
    if (cache) { auto kv = cache->update(layerIdx_, k, v); Kfull = kv.first; Vfull = kv.second; }
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
    if (quantKVBits_ > 0) return forwardQuantKV(x, cos, sin, maskF32, cache, pastLen);
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

    // ---- scores = Q @ K^T * scaling(1.0), mask, softmax(f32) ----
    mx::array scores = mx::matmul(q, mx::swapaxes(Krep, -1, -2));  // [numQ, seqQ, seqK]
    if (scaling_ != 1.0f) scores = mx::multiply(scores, lit(scaling_, scores));

    mx::array sf = mx::astype(scores, mx::float32);
    if (maskF32.size() > 0) sf = mx::add(sf, maskF32);  // broadcast [seqQ, seqK] over heads
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

    // SDPA expects [B, heads, L, headDim]; GQA (numKV < numQ) handled internally.
    mx::array Q = mx::reshape(q,     {1, numQHeads_,  seq,  headDim_});
    mx::array K = mx::reshape(Kfull, {1, numKVHeads_, seqK, headDim_});
    mx::array V = mx::reshape(Vfull, {1, numKVHeads_, seqK, headDim_});
    mx::array M = mx::reshape(mx::astype(maskF32, x.dtype()), {1, 1, seq, seqK});  // additive

    mx::array O = mx::fast::scaled_dot_product_attention(Q, K, V, scaling_, "", M);  // [1,numQ,seq,headDim]

    O = mx::transpose(mx::reshape(O, {numQHeads_, seq, headDim_}), {1, 0, 2});  // [seq, numQ, headDim]
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
        ? cache->updateQuant(layerIdx_, k, v, gs, bits)
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
