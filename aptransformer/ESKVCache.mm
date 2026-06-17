#include "ESKVCache.h"

namespace es {

ESKVCache::ESKVCache(int numLayers)
    : k_(numLayers), v_(numLayers),
      kq_(numLayers), ks_(numLayers), kb_(numLayers),
      vq_(numLayers), vs_(numLayers), vb_(numLayers) {}

std::pair<mx::array, mx::array> ESKVCache::update(int layer, const mx::array & kNew, const mx::array & vNew) {
    if (!k_[layer].has_value()) {
        k_[layer] = kNew;
        v_[layer] = vNew;
    } else {
        k_[layer] = mx::concatenate({*k_[layer], kNew}, 1);  // append along seq axis
        v_[layer] = mx::concatenate({*v_[layer], vNew}, 1);
    }
    return {*k_[layer], *v_[layer]};
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
    for (auto & a : kq_) a.reset();
    for (auto & a : ks_) a.reset();
    for (auto & a : kb_) a.reset();
    for (auto & a : vq_) a.reset();
    for (auto & a : vs_) a.reset();
    for (auto & a : vb_) a.reset();
    seqLen_ = 0;
}

}  // namespace es
