#pragma once
//  ESAttention — Gemma4TextAttention. Layer-type-aware (local/global), functional KV cache.
//
//  Per layer (selected by config from layer_idx):
//   - local (sliding):  headDim 256, kvHeads 16, has q/k/v_proj.
//   - global (full):    headDim 512, kvHeads 4,  attention_k_eq_v -> v_proj absent, V reuses
//                        the pre-norm k_proj output.
//  Order: q_proj -> view[heads,headDim] -> q_norm -> RoPE -> (k likewise) ;
//         V = v_norm(value source), NO RoPE. scaling = 1.0 (QK-norm absorbs 1/sqrt(d)).
//         softmax in float32. No attention-score softcap in Gemma-4 text.
#include "mlx/mlx.h"
#include "ESModelConfig.h"
#include "ESRMSNorm.h"
#include "ESKVCache.h"
#include "ESLinear.h"
#include <optional>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

class ESAttention {
public:
    ESAttention(const ESModelConfig & config, int layerIdx, const ESWeightLoader & weights);

    // x: [seq, hidden]; cos/sin: [seq, headDim]; mask: additive f32 [seqQ, seqK] (or empty).
    // cache may be null (pure prefill). sharedKV is the per-forward shared-KV scratch (elastic
    // models); null for non-shared models. Returns [seq, hidden].
    mx::array forward(const mx::array & x,
                      const mx::array & cos,
                      const mx::array & sin,
                      const mx::array & maskF32,
                      ESKVCache *       cache,
                      int               pastLen,
                      ESSharedKV *      sharedKV = nullptr) const;

private:
    // Produce the full [numKV, seqK, headDim] K and V for this step: reuse the shared-KV scratch
    // for shared layers, else project/norm/RoPE, append to the cache, and (if a storing layer)
    // write into the scratch for the shared layers of this type to reuse.
    std::pair<mx::array, mx::array> keyValue(const mx::array & x, const mx::array & cos,
                                             const mx::array & sin, ESKVCache * cache,
                                             ESSharedKV * sharedKV) const;

    // Performance-fused core: same q/k/v + QK-norm + RoPE, but the attention math
    // (GQA repeat, QK^T, mask, softmax, AV) runs through mx::fast::scaled_dot_product_attention.
    mx::array forwardFused(const mx::array & x, const mx::array & cos, const mx::array & sin,
                           const mx::array & maskF32, ESKVCache * cache, int pastLen,
                           ESSharedKV * sharedKV) const;

    // Op-level tiled prefill core: K/V processed in chunks of tiledKChunk_ keys with an
    // online-softmax merge (running row max + normalizer, f32), so no full [heads, seqQ, seqK]
    // score matrix ever materializes. GQA via broadcast matmul (no repeatKV copies). Engaged from
    // forward() at prefill shapes (seq > 8) when config.tiledKChunk > 0; math is e-equivalent to
    // the unfused reference (f32 precise softmax), argmax stability gated by --tiled-verify.
    mx::array forwardTiled(const mx::array & x, const mx::array & cos, const mx::array & sin,
                           const mx::array & maskF32, ESKVCache * cache, int pastLen,
                           ESSharedKV * sharedKV) const;

    // Quantized-KV core: K/V stored quantized; attention via quantized_matmul (Q@K^T, scores@V),
    // so full-precision K/V never materialize — the long-context cache-bandwidth lever.
    // (Does not support shared-KV; the elastic models run on the bf16 manual/fused paths.)
    mx::array forwardQuantKV(const mx::array & x, const mx::array & cos, const mx::array & sin,
                             const mx::array & maskF32, ESKVCache * cache, int pastLen) const;

    bool  fused_;
    bool  isKvShared_, storeFullKv_;
    int   quantKVBits_, quantGroupSize_;
    bool  useQuantKV_;   // this layer runs the quantized-KV attention core (see ctor)
    int   layerIdx_;
    int   numQHeads_;
    int   numKVHeads_;
    int   headDim_;
    int   groups_;       // numQ / numKV
    bool  kEqV_;
    bool  isSliding_;
    int   slidingWindow_;
    bool  slidingCache_;  // evict sliding-layer KV to the window on decode (bit-exact)
    bool  preallocCache_; // preallocated slice_update KV storage (vs legacy concat-grow)
    bool  chunkedPrefill_; // ALSO trim sliding K/V on multi-token appends (P5 chunked prefill)
    int   tiledKChunk_;    // >0: op-level tiled prefill attention, K chunk size (see forwardTiled)
    float scaling_;      // 1.0

    ESLinear                qProj_, kProj_, oProj_;
    std::optional<ESLinear> vProj_;   // present iff !kEqV_
    bool                    hasVProj_;

    ESRMSNorm qNorm_, kNorm_, vNorm_;  // vNorm_ weightless
};

}  // namespace es
