#include "ESVisionPatchEmbedder.h"

#include <utility>

namespace es {

ESVisionPatchEmbedder::ESVisionPatchEmbedder(ESLinear inputProj, mx::array positionTable)
    : inputProj_(std::move(inputProj)), table_(std::move(positionTable)) {}

mx::array ESVisionPatchEmbedder::forward(const mx::array & pixelValues, const mx::array & positionIds) const {
    // 2 * (x - 0.5) in float32, then to the weight dtype (bf16).
    mx::array px = mx::multiply(mx::array(2.0f), mx::subtract(pixelValues, mx::array(0.5f)));
    px = mx::astype(px, table_.dtype());
    mx::array h = inputProj_.forward(px);                                   // [n, hidden]

    const int n = positionIds.shape(0), P = table_.shape(1), H = table_.shape(2);
    mx::array xs = mx::reshape(mx::slice(positionIds, {0, 0}, {n, 1}), {n});
    mx::array ys = mx::reshape(mx::slice(positionIds, {0, 1}, {n, 2}), {n});
    mx::array tx = mx::reshape(mx::slice(table_, {0, 0, 0}, {1, P, H}), {P, H});
    mx::array ty = mx::reshape(mx::slice(table_, {1, 0, 0}, {2, P, H}), {P, H});
    mx::array pos = mx::add(mx::take(tx, xs, 0), mx::take(ty, ys, 0));      // x_emb + y_emb
    return mx::add(h, pos);
}

}  // namespace es
