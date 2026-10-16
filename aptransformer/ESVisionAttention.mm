#include "ESVisionAttention.h"
#include "ESVisionRotaryEmbedding.h"
#include "ESWeightLoader.h"
#include "ESOps.h"

namespace es {

ESVisionAttention::ESVisionAttention(const ESVisionConfig & cfg, const std::string & prefix,
                                     const ESWeightLoader & w)
    : numHeads_(cfg.numAttentionHeads), numKVHeads_(cfg.numKeyValueHeads), headDim_(cfg.headDim),
      qProj_(esMakeClippableLinear(w, prefix + ".q_proj")),
      kProj_(esMakeClippableLinear(w, prefix + ".k_proj")),
      vProj_(esMakeClippableLinear(w, prefix + ".v_proj")),
      oProj_(esMakeClippableLinear(w, prefix + ".o_proj")),
      qNorm_(w.get(prefix + ".q_norm.weight"), cfg.rmsNormEps),
      kNorm_(w.get(prefix + ".k_norm.weight"), cfg.rmsNormEps),
      vNorm_(cfg.rmsNormEps) {}

mx::array ESVisionAttention::forward(const mx::array & x, const mx::array & cos, const mx::array & sin) const {
    const int n = x.shape(0);

    mx::array q = mx::reshape(qProj_.forward(x), {n, numHeads_, headDim_});
    q = ESVisionRotaryEmbedding::apply(qNorm_.forward(q), cos, sin);
    q = mx::transpose(q, {1, 0, 2});                                   // [heads, n, d]

    mx::array k = mx::reshape(kProj_.forward(x), {n, numKVHeads_, headDim_});
    k = ESVisionRotaryEmbedding::apply(kNorm_.forward(k), cos, sin);
    k = mx::transpose(k, {1, 0, 2});

    mx::array v = mx::reshape(vProj_.forward(x), {n, numKVHeads_, headDim_});
    v = mx::transpose(vNorm_.forward(v), {1, 0, 2});

    k = repeatKV(k, numHeads_ / numKVHeads_);
    v = repeatKV(v, numHeads_ / numKVHeads_);

    // scaling 1.0; softmax in float32, cast back (eager_attention_forward).
    mx::array scores = mx::matmul(q, mx::swapaxes(k, -1, -2));         // [heads, n, n]
    mx::array wts = mx::astype(mx::softmax(mx::astype(scores, mx::float32), -1, /*precise=*/true), x.dtype());
    mx::array out = mx::matmul(wts, v);                                // [heads, n, d]
    out = mx::reshape(mx::transpose(out, {1, 0, 2}), {n, numHeads_ * headDim_});
    return oProj_.forward(out);
}

}  // namespace es
