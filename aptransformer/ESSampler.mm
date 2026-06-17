#include "ESSampler.h"
#include "ESOps.h"

namespace es {

int ESSampler::argmax(const mx::array & logits) {
    mx::array idx = mx::argmax(logits, 0);
    mx::eval(idx);
    return (int) idx.item<uint32_t>();
}

int ESSampler::sample(const mx::array & logits) const {
    if (cfg_.greedy || cfg_.temperature <= 0.0f) {
        return argmax(logits);
    }
    // Temperature + top-k + top-p (nucleus). Computed in f32.
    mx::array lf = mx::astype(logits, mx::float32);
    lf = mx::divide(lf, mx::array(cfg_.temperature, mx::float32));

    int vocab = lf.shape(0);
    int k = (cfg_.topK > 0 && cfg_.topK < vocab) ? cfg_.topK : vocab;

    // top-k via argsort (descending).
    mx::array order = mx::argsort(mx::negative(lf), 0);          // indices of largest first
    mx::array topIdx = mx::astype(mx::slice(order, {0}, {k}), mx::int32);  // [k] int32
    mx::array topLog = mx::take(lf, topIdx, 0);                  // [k]
    mx::array probs  = mx::softmax(topLog, 0, /*precise=*/true); // [k]

    // top-p filtering on the sorted (descending) probs.
    mx::array csum = mx::cumsum(probs, 0);
    mx::eval(csum, probs, topIdx);
    const float * pc = csum.data<float>();
    const float * pp = probs.data<float>();

    int cutoff = k;
    for (int i = 0; i < k; ++i) {
        if (pc[i] >= cfg_.topP) { cutoff = i + 1; break; }
    }

    // Renormalize over [0, cutoff) and sample with a simple LCG seeded from cfg_.seed.
    double total = 0.0;
    for (int i = 0; i < cutoff; ++i) total += pp[i];
    static unsigned long long state = 0;
    if (state == 0) state = cfg_.seed * 2862933555777941757ULL + 3037000493ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    double r = ((double) (state >> 11) / (double) (1ULL << 53)) * total;

    const int * ti = topIdx.data<int>();  // int32 indices
    double acc = 0.0;
    for (int i = 0; i < cutoff; ++i) {
        acc += pp[i];
        if (r <= acc) return ti[i];
    }
    return ti[cutoff - 1];
}

}  // namespace es
