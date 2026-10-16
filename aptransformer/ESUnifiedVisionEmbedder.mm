#include "ESUnifiedVisionEmbedder.h"
#include "ESWeightLoader.h"

namespace es {

mx::array ESLayerNorm::forward(const mx::array & x) const {
    mx::array xf   = mx::astype(x, mx::float32);
    mx::array mean = mx::mean(xf, -1, /*keepdims=*/true);
    mx::array d    = mx::subtract(xf, mean);
    mx::array var  = mx::mean(mx::multiply(d, d), -1, /*keepdims=*/true);
    mx::array out  = mx::multiply(d, mx::rsqrt(mx::add(var, mx::array(eps_, mx::float32))));
    out = mx::add(mx::multiply(out, mx::astype(w_, mx::float32)), mx::astype(b_, mx::float32));
    return mx::astype(out, x.dtype());
}

static const std::string kP = "vision_embedder.";

ESUnifiedVisionEmbedder::ESUnifiedVisionEmbedder(const ESVisionConfig & cfg, const ESWeightLoader & w)
    : ln1_(w.get(kP + "patch_ln1.weight"), w.get(kP + "patch_ln1.bias"), cfg.layerNormEps),
      ln2_(w.get(kP + "patch_ln2.weight"), w.get(kP + "patch_ln2.bias"), cfg.layerNormEps),
      posNorm_(w.get(kP + "pos_norm.weight"), w.get(kP + "pos_norm.bias"), cfg.layerNormEps),
      denseW_(w.get(kP + "patch_dense.weight")),
      denseB_(w.get(kP + "patch_dense.bias")),
      posTable_(w.get(kP + "pos_embedding")),
      embedder_(esMakeLinear(w, "embed_vision.embedding_projection.weight", 0, 64), cfg.rmsNormEps) {}

mx::array ESUnifiedVisionEmbedder::run(const mx::array & pixelValues, const mx::array & positionIds, Trace * tr) const {
    mx::array h = mx::astype(pixelValues, denseW_.dtype());        // pixel_values.to(weight dtype)
    h = ln1_.forward(h);                       if (tr) tr->ln1 = h;
    // nn.Linear with bias: one fused addmm (f32 accumulation, one rounding).
    h = mx::addmm(denseB_, h, mx::transpose(denseW_));             if (tr) tr->dense = h;
    h = ln2_.forward(h);                       if (tr) tr->ln2 = h;

    const int n = positionIds.shape(0), P = posTable_.shape(0), D = posTable_.shape(2);
    mx::array xs = mx::reshape(mx::slice(positionIds, {0, 0}, {n, 1}), {n});
    mx::array ys = mx::reshape(mx::slice(positionIds, {0, 1}, {n, 2}), {n});
    mx::array tx = mx::reshape(mx::slice(posTable_, {0, 0, 0}, {P, 1, D}), {P, D});
    mx::array ty = mx::reshape(mx::slice(posTable_, {0, 1, 0}, {P, 2, D}), {P, D});
    // (pos_embedding[clamped, axes] * valid).sum(-2): x-row + y-row, in the table dtype.
    mx::array pos = mx::add(mx::take(tx, xs, 0), mx::take(ty, ys, 0));
    h = mx::add(h, pos);                       if (tr) tr->posAdded = h;
    h = posNorm_.forward(h);                   if (tr) tr->posNorm = h;
    mx::array y = embedder_.forward(h);        if (tr) tr->softTokens = y;
    return y;
}

mx::array ESUnifiedVisionEmbedder::forward(const mx::array & pixelValues, const mx::array & positionIds) const {
    return run(pixelValues, positionIds, nullptr);
}

ESUnifiedVisionEmbedder::Trace ESUnifiedVisionEmbedder::forwardTrace(const mx::array & pixelValues,
                                                                     const mx::array & positionIds) const {
    Trace tr{mx::array(0.0f), mx::array(0.0f), mx::array(0.0f), mx::array(0.0f), mx::array(0.0f), mx::array(0.0f)};
    run(pixelValues, positionIds, &tr);
    return tr;
}

}  // namespace es
