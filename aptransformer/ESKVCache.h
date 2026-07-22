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

    // ── Compiled-step mode (P3). While engaged, update() ignores its mode arguments and instead:
    // scatter-writes kNew/vNew into the fixed-capacity slot buffer at a POSITION GIVEN AS AN
    // ARRAY (so the recorded graph replays for any position), and returns the FULL-CAPACITY
    // buffers — validity/window are enforced by the additive mask, which the compiled step also
    // computes from the position (masked slots contribute exactly 0 through softmax, the P1
    // argument, so this is output-identical to sliced attention). maxKeep>0 still identifies
    // sliding layers: they use `slidingIdx` (slot = pos - slidingBase), others `globalIdx`.
    // The driver (ESCompiledStep) owns slot layout/compaction and re-wires buffers via
    // slotK/slotV/setSlot around each compiled call.
    void beginStep(const mx::array & globalIdx, const mx::array & slidingIdx) {
        stepGlobalIdx_ = globalIdx; stepSlidingIdx_ = slidingIdx; stepMode_ = true;
    }
    void endStep() { stepMode_ = false; stepGlobalIdx_.reset(); stepSlidingIdx_.reset(); }
    const mx::array & slotK(int layer) const { return *slots_[layer].k; }
    const mx::array & slotV(int layer) const { return *slots_[layer].v; }
    void setSlot(int layer, mx::array k, mx::array v) {
        slots_[layer].k = std::move(k); slots_[layer].v = std::move(v);
    }
    mx::array takeSlotK(int layer) { mx::array a = std::move(*slots_[layer].k); slots_[layer].k.reset(); return a; }
    mx::array takeSlotV(int layer) { mx::array a = std::move(*slots_[layer].v); slots_[layer].v.reset(); return a; }

    // The logical cached {K, V} for `layer` WITHOUT appending — what the next update() would
    // build on. Mode must match how the cache was filled. Used by ESCompiledStep to adopt a
    // prefilled cache into step layout.
    std::pair<mx::array, mx::array> current(int layer, bool prealloc) const;

    // ── KV snapshot persistence (prealloc mode): amortize a standing-prefix prefill (a
    // persona) across process launches. saveSnapshot writes the LIVE RANGES only (layout-
    // independent safetensors: k_<i>/v_<i> + metadata {fingerprint, pos}); restoreSnapshot
    // verifies the caller's fingerprint (which must encode model + config + the exact primed
    // token ids), re-homes each layer into fresh chunk-aligned buffers, and returns the cached
    // `pos` — or -1 on missing file / fingerprint mismatch / malformed content (caller falls
    // back to a normal prefill). Restored content is byte-identical to the saved buffers, so
    // continuation is bit-exact (gated via --persist-verify).
    bool saveSnapshot(const std::string & path, const std::string & fingerprint, int pos) const;
    int  restoreSnapshot(const std::string & path, const std::string & fingerprint);

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
    // Compiled-step mode state (see beginStep): scatter position indices for this token.
    bool stepMode_ = false;
    std::optional<mx::array> stepGlobalIdx_, stepSlidingIdx_;
    // Quantized storage: per-layer {packed, scales, biases} for K and V.
    std::vector<std::optional<mx::array>> kq_, ks_, kb_, vq_, vs_, vb_;
    int seqLen_ = 0;
};

}  // namespace es
