#pragma once
//  ESKVCache — ObjC++ owns KV storage; attention math stays functional.
//
//  Per-layer key/value buffers, shape [kvHeads, seqSoFar, headDim]. update() appends the
//  new tokens' K/V along the sequence axis and returns the full cached K/V. Phase 1 has no
//  sliding-window eviction (out of scope) — buffers simply grow.
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

    // Append kNew/vNew ([kvHeads, nNew, headDim]) for `layer`; returns full {K, V}.
    std::pair<mx::array, mx::array> update(int layer, const mx::array & kNew, const mx::array & vNew);

    // Quantized KV: quantize kNew/vNew along the head dim, append the packed tuples, return the
    // full quantized cache. K/V are each {packed, scales, biases}. Attention then uses
    // quantized_matmul (Q@K^T then scores@V) — no full-precision K/V ever materializes in DRAM.
    struct QKV { mx::array kq, ks, kb, vq, vs, vb; };
    QKV updateQuant(int layer, const mx::array & kNew, const mx::array & vNew, int groupSize, int bits);

    int seqLen() const { return seqLen_; }     // positions cached (advanced by markStep)
    void markStep(int nNew) { seqLen_ += nNew; }  // call once per forward (not per layer)
    void reset();

private:
    std::vector<std::optional<mx::array>> k_, v_;
    // Quantized storage: per-layer {packed, scales, biases} for K and V.
    std::vector<std::optional<mx::array>> kq_, ks_, kb_, vq_, vs_, vb_;
    int seqLen_ = 0;
};

}  // namespace es
