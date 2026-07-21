#pragma once
//  ESKVCache — ObjC++ owns KV storage; attention math stays functional.
//
//  Per-layer key/value buffers, shape [kvHeads, seqSoFar, headDim]. update() appends the
//  new tokens' K/V along the sequence axis and returns the full cached K/V. Two storage modes,
//  selected per call (constant per LM instance, from ESModelConfig::preallocKVCache):
//
//   - legacy (prealloc=false): append via mx::concatenate, evict via slice. Simple, but it
//     copies the whole cache every token AND every append allocates a fresh, slightly larger
//     Metal buffer — monotonically growing sizes defeat MLX's buffer cache (its reuse window
//     is [size, size+2 pages)), so decode pays a real allocation per layer per token.
//
//   - prealloc (prealloc=true): fixed-capacity buffers grown in kGrowChunk-position steps;
//     appends write in place via mx::slice_update (buffer donation — no copy, no allocation),
//     and the returned K/V are slice VIEWS of the valid range (MLX's SDPA accepts strided K/V
//     when batch == 1, and prepare_reshape makes the [kv,seq,hd]->[1,kv,seq,hd] reshape a
//     zero-copy view). Sliding-window eviction advances a logical `start` instead of trimming
//     storage; when the write cursor hits capacity the live range is compacted to the front of
//     a fresh buffer (one copy per ~kGrowChunk tokens, amortized O(1/chunk) per token — and the
//     buffer sizes repeat, so MLX's buffer cache recycles them). Returned content is identical
//     to the legacy mode by construction; verified token-exact via --cache-verify.
#include "mlx/mlx.h"
#include <optional>
#include <vector>

namespace es {
namespace mx = mlx::core;

// Shared-KV scratch for ONE forward pass (elastic E2B/E4B). The last non-shared layer of each
// attention type writes its full K/V here; the shared layers of that type read it instead of
// computing their own. Type index: 0 = sliding (local), 1 = full (global).
struct ESSharedKV {
    std::optional<mx::array> k_[2], v_[2];
    void store(bool sliding, const mx::array & k, const mx::array & v) {
        int t = sliding ? 0 : 1; k_[t] = k; v_[t] = v;
    }
    std::pair<mx::array, mx::array> get(bool sliding) const {
        int t = sliding ? 0 : 1; return {*k_[t], *v_[t]};
    }
};

class ESKVCache {
public:
    explicit ESKVCache(int numLayers);

    // Positions the prealloc mode grows capacity by (and the amortization period of its
    // compaction copies). One chunk of sliding-layer K+V is ~4 MB — negligible headroom.
    static constexpr int kGrowChunk = 256;

    // Append kNew/vNew ([kvHeads, nNew, headDim]) for `layer`; returns full {K, V}.
    // maxKeep > 0 keeps only the last `maxKeep` keys after appending (sliding-window eviction;
    // callers pass it only for sliding layers on single-token decode — bit-exact there).
    // prealloc selects the storage mode above; a given cache instance must be driven with a
    // consistent value (it is: ESAttention passes config.preallocKVCache).
    std::pair<mx::array, mx::array> update(int layer, const mx::array & kNew, const mx::array & vNew,
                                           int maxKeep = 0, bool prealloc = false);

    // Quantized KV: quantize kNew/vNew along the head dim, append the packed tuples, return the
    // full quantized cache. K/V are each {packed, scales, biases}. Attention then uses
    // quantized_matmul (Q@K^T then scores@V) — no full-precision K/V ever materializes in DRAM.
    // (Legacy concat storage only — quant-KV is a capacity lever off the hot path.)
    struct QKV { mx::array kq, ks, kb, vq, vs, vb; };
    QKV updateQuant(int layer, const mx::array & kNew, const mx::array & vNew, int groupSize, int bits);

    int seqLen() const { return seqLen_; }     // positions cached (advanced by markStep)
    void markStep(int nNew) { seqLen_ += nNew; }  // call once per forward (not per layer)
    void reset();

private:
    // Legacy storage: the exact cached array per layer.
    std::vector<std::optional<mx::array>> k_, v_;
    // Prealloc storage: [kvHeads, capacity, headDim] buffers; live range is [start, len).
    struct Slot {
        std::optional<mx::array> k, v;
        int len   = 0;  // write cursor (buffer positions filled)
        int start = 0;  // logical window start (advanced by maxKeep eviction)
    };
    std::vector<Slot> slots_;
    // Quantized storage: per-layer {packed, scales, biases} for K and V.
    std::vector<std::optional<mx::array>> kq_, ks_, kb_, vq_, vs_, vb_;
    int seqLen_ = 0;
};

}  // namespace es
