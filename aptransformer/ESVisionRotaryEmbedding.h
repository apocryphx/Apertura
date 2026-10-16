#pragma once
//  ESVisionRotaryEmbedding — Gemma4VisionRotaryEmbedding: axial 2-D RoPE over patch (x, y).
//
//  inv_freq has headDim/4 entries (16): 1 / theta^(arange(0, headDim/2, 2) / (headDim/2)), theta 100.
//  For a patch at (x, y) the angles are x*inv_freq and y*inv_freq; the reference lays the
//  cos/sin tables out as [c_x, c_x, c_y, c_y] (each block headDim/4 wide, 64 total) and
//  apply_multidimensional_rope then splits the head into two 32-wide halves, rotating the first
//  with the x angles and the second with the y angles — ordinary NEOX rotate_half within each
//  half. Full rotation, no partial factor. cos/sin are computed in float32 and cast to the
//  compute dtype, as the reference casts to x.dtype.
#include "mlx/mlx.h"
#include <vector>

namespace es {
namespace mx = mlx::core;

class ESVisionRotaryEmbedding {
public:
    ESVisionRotaryEmbedding(int headDim, float thetaBase, mx::Dtype computeDtype);

    // positionIds: int32 [n, 2] (x, y). Returns {cos, sin}, each [n, headDim].
    std::pair<mx::array, mx::array> cosSin(const mx::array & positionIds) const;

    // x: [n, heads, headDim]; cos/sin: [n, headDim]. Rotates each spatial half with its own table.
    static mx::array apply(const mx::array & x, const mx::array & cos, const mx::array & sin);

private:
    int                headDim_;
    mx::Dtype          computeDtype_;
    std::vector<float> invFreq_;  // headDim/4 entries
};

}  // namespace es
