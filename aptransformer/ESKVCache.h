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
#include <array>
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
    // prealloc mirrors update()'s storage modes: legacy concat-grow (the bit-exact reference —
    // measured 0.3 tok/s decode at 61K context, six full-cache copies per layer per token) vs
    // fixed-capacity buffers with slice_update appends (the runtime path; same donation
    // argument as update(), one compaction copy per ~kGrowChunk tokens). No eviction: the
    // quantized path serves global (unwindowed) layers.
    struct QKV { mx::array kq, ks, kb, vq, vs, vb; };
    QKV updateQuant(int layer, const mx::array & kNew, const mx::array & vNew, int groupSize, int bits,
                    bool prealloc = false);

    // Raw-K storage (global layers, config.rawKV): the cache holds ONLY kRaw — the
    // pre-norm k_proj output — from which attention derives both K (rope(knorm)) and
    // V (vnorm). Half the bytes of the k/v pair. Uses the Slot's k side; v stays empty.
    // Returns the live-range view plus the UNSLICED buffer and its row pitch, so the
    // fused decode kernel can read in place without a contiguity copy. No eviction
    // (global layers are unwindowed).
    struct Raw { mx::array view, buffer; int len, pitch; };
    Raw updateRaw(int layer, const mx::array & kNew, bool prealloc);

    // q8-packed raw-K storage (config.rawKV + rawKVQ8): kRaw rows quantized per row to
    // affine u8 (q [kvH, cap, hd] u8, scale/bias [kvH, cap, 1] f32) at append time by host
    // ops. Quarter the original K+V bytes. Returns live-range views (prefill ops dequant)
    // plus the unsliced buffers + pitch (the fused decode kernel reads in place). A layer
    // restored from a bf16 raw snapshot migrates lazily: the first q8 append quantizes the
    // restored live range and retires the bf16 slot. Prealloc mode only.
    struct RawQ8 { mx::array qview, sc, bs, qbuf, scbuf, bsbuf; int len, pitch; };
    RawQ8 updateRawQ8(int layer, const mx::array & kNew);
    // The row quantizer behind updateRawQ8 ({q u8, scale f32, bias f32} per row along the
    // last axis) — public so the cacheless probe path can mirror the cached values exactly.
    static std::array<mx::array, 3> quantizeRawRows(const mx::array & kNew);

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
    // Prealloc quantized storage: six [kvHeads, capacity, *] buffers per layer, live range
    // [0, len) — global layers never evict, so no start cursor.
    // q8 raw storage: three tensors (q u8 hd-wide, scale/bias f32 1-wide) share one cursor.
    struct RQ8Slot {
        std::optional<mx::array> q, sc, bs;
        int len = 0;
    };
    std::vector<RQ8Slot> rq8slots_;
    struct QSlot {
        std::optional<mx::array> t[6];  // kq, ks, kb, vq, vs, vb
        int len = 0;
    };
    std::vector<QSlot> qslots_;
    int seqLen_ = 0;
};

}  // namespace es
