#pragma once
//  ESRotaryEmbedding — Gemma4TextRotaryEmbedding, NEOX-style (rotate_half).
//
//  Two variants, selected by config per layer type:
//   - local  (default):      headDim=256, theta=1e4, full rotation.
//   - global (proportional): headDim=512, theta=1e6, partial_rotary_factor=0.25 ->
//       inv_freq = [64 real freqs, 192 zeros]; the zero tail yields cos=1/sin=0 so
//       those dims pass through unrotated (p-RoPE).
//
//  cos/sin are computed in float32 then cast to computeDtype (matches PyTorch x.dtype()).
#include "mlx/mlx.h"
#include <vector>

namespace es {
namespace mx = mlx::core;

class ESRotaryEmbedding {
public:
    // partialRotaryFactor < 1.0 selects proportional p-RoPE; 1.0 = full rotation.
    ESRotaryEmbedding(int headDim, float thetaBase, float partialRotaryFactor, mx::Dtype computeDtype);

    // Returns {cos, sin}, each [seqLen, headDim], for positions [offset, offset+seqLen).
    std::pair<mx::array, mx::array> cosSin(int seqLen, int offset) const;

    // Apply RoPE. x: [seq, heads, headDim]; cos/sin: [seq, headDim] (broadcast over heads).
    static mx::array apply(const mx::array & x, const mx::array & cos, const mx::array & sin);

    int headDim() const { return headDim_; }

private:
    int                headDim_;
    mx::Dtype          computeDtype_;
    std::vector<float> invFreq_;  // length headDim/2, with zero tail for partial rotary
};

}  // namespace es
