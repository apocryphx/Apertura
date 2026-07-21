#include "ESKVCache.h"

namespace es {

ESKVCache::ESKVCache(int numLayers)
    : k_(numLayers), v_(numLayers), slots_(numLayers),
      kq_(numLayers), ks_(numLayers), kb_(numLayers),
      vq_(numLayers), vs_(numLayers), vb_(numLayers) {}

static int roundUpChunk(int n) {
    return ((n + ESKVCache::kGrowChunk - 1) / ESKVCache::kGrowChunk) * ESKVCache::kGrowChunk;
}

std::pair<mx::array, mx::array> ESKVCache::update(int layer, const mx::array & kNew, const mx::array & vNew,
                                                  int maxKeep, bool prealloc) {
    if (!prealloc) {
        // ── Legacy mode: concat-grow (+ slice eviction). Kept as the bit-exact reference and
        // for the --cache-verify A/B; the prealloc mode below is the default runtime path.
        if (!k_[layer].has_value()) {
            k_[layer] = kNew;
            v_[layer] = vNew;
        } else {
            k_[layer] = mx::concatenate({*k_[layer], kNew}, 1);  // append along seq axis
            v_[layer] = mx::concatenate({*v_[layer], vNew}, 1);
        }
        // Sliding-window eviction: keep only the last `maxKeep` keys along the seq axis. The caller
        // only requests this for sliding layers on a single-token (decode) append, where the dropped
        // keys were all masked to -1e30 (softmax weight == 0) — so retaining the tail is bit-exact.
        if (maxKeep > 0 && k_[layer]->shape(1) > maxKeep) {
            const int len = k_[layer]->shape(1), hd = k_[layer]->shape(2), nh = k_[layer]->shape(0);
            k_[layer] = mx::slice(*k_[layer], {0, len - maxKeep, 0}, {nh, len, hd});
            const int vhd = v_[layer]->shape(2), vnh = v_[layer]->shape(0), vlen = v_[layer]->shape(1);
            v_[layer] = mx::slice(*v_[layer], {0, vlen - maxKeep, 0}, {vnh, vlen, vhd});
        }
        return {*k_[layer], *v_[layer]};
    }

    // ── Prealloc mode: in-place slice_update append into a chunk-grown buffer; the live range
    // [start, len) is what the legacy mode would have stored, and the returned views slice it.
    Slot & s = slots_[layer];
    const int kvH = kNew.shape(0), nNew = kNew.shape(1), hd = kNew.shape(2);

    if (!s.k.has_value()) {
        // First append (prefill): store directly, capacity == content — identical to legacy's
        // first store. The first capacity growth below re-homes it into a chunked buffer.
        s.k = kNew; s.v = vNew;
        s.len = nNew; s.start = 0;
    } else {
        const int cap = s.k->shape(1);
        if (s.len + nNew > cap) {
            // Compact the live range to the front of a fresh chunk-rounded buffer. This is the
            // only copy in this mode, and it runs once per ~kGrowChunk tokens (sliding layers:
            // content is capped at the window, so the buffer size repeats and MLX's buffer
            // cache recycles it; global layers: capacity steps up by kGrowChunk).
            const int content = s.len - s.start;
            const int newCap = roundUpChunk(content + nNew + kGrowChunk);
            mx::array nk = mx::zeros({kvH, newCap, hd}, kNew.dtype());
            mx::array nv = mx::zeros({kvH, newCap, hd}, vNew.dtype());
            if (content > 0) {
                nk = mx::slice_update(nk, mx::slice(*s.k, {0, s.start, 0}, {kvH, s.len, hd}),
                                      {0, 0, 0}, {kvH, content, hd});
                nv = mx::slice_update(nv, mx::slice(*s.v, {0, s.start, 0}, {kvH, s.len, hd}),
                                      {0, 0, 0}, {kvH, content, hd});
            }
            s.k = nk; s.v = nv;
            s.len = content; s.start = 0;
        }
        // In-place append: the stored buffer is the only live reference by eval time (the views
        // returned last step died with that step's graph), so slice_update donates — no copy.
        s.k = mx::slice_update(*s.k, kNew, {0, s.len, 0}, {kvH, s.len + nNew, hd});
        s.v = mx::slice_update(*s.v, vNew, {0, s.len, 0}, {kvH, s.len + nNew, hd});
        s.len += nNew;
    }

    // Sliding-window eviction: advance the logical start instead of trimming storage (the
    // dropped keys were masked to -1e30 — softmax weight exactly 0 — so this is bit-exact).
    if (maxKeep > 0 && s.len - s.start > maxKeep) s.start = s.len - maxKeep;

    if (s.start == 0 && s.len == s.k->shape(1)) return {*s.k, *s.v};  // full buffer, no slice needed
    mx::array K = mx::slice(*s.k, {0, s.start, 0}, {kvH, s.len, hd});
    mx::array V = mx::slice(*s.v, {0, s.start, 0}, {kvH, s.len, hd});
    return {K, V};
}

ESKVCache::QKV ESKVCache::updateQuant(int layer, const mx::array & kNew, const mx::array & vNew,
                                      int groupSize, int bits) {
    // Quantize the new tokens along the head dim (last axis). Each seq position is independently
    // quantized, so appending along the seq axis just stacks packed rows — valid.
    auto kq = mx::quantize(kNew, groupSize, bits);  // {packed, scales, biases}
    auto vq = mx::quantize(vNew, groupSize, bits);
    if (!kq_[layer].has_value()) {
        kq_[layer] = kq[0]; ks_[layer] = kq[1]; kb_[layer] = kq[2];
        vq_[layer] = vq[0]; vs_[layer] = vq[1]; vb_[layer] = vq[2];
    } else {
        kq_[layer] = mx::concatenate({*kq_[layer], kq[0]}, 1);
        ks_[layer] = mx::concatenate({*ks_[layer], kq[1]}, 1);
        kb_[layer] = mx::concatenate({*kb_[layer], kq[2]}, 1);
        vq_[layer] = mx::concatenate({*vq_[layer], vq[0]}, 1);
        vs_[layer] = mx::concatenate({*vs_[layer], vq[1]}, 1);
        vb_[layer] = mx::concatenate({*vb_[layer], vq[2]}, 1);
    }
    return {*kq_[layer], *ks_[layer], *kb_[layer], *vq_[layer], *vs_[layer], *vb_[layer]};
}

void ESKVCache::reset() {
    for (auto & a : k_) a.reset();
    for (auto & a : v_) a.reset();
    for (auto & s : slots_) { s.k.reset(); s.v.reset(); s.len = 0; s.start = 0; }
    for (auto & a : kq_) a.reset();
    for (auto & a : ks_) a.reset();
    for (auto & a : kb_) a.reset();
    for (auto & a : vq_) a.reset();
    for (auto & a : vs_) a.reset();
    for (auto & a : vb_) a.reset();
    seqLen_ = 0;
}

}  // namespace es
