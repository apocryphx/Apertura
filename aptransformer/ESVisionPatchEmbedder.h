#pragma once
//  ESVisionPatchEmbedder — Gemma4VisionPatchEmbedder.
//
//    h = input_proj( bf16( 2 * (pixels - 0.5) ) ) + table[0][x] + table[1][y]
//
//  pixels arrive as float32 k/255 (no normalization in the processor); the model centres them
//  in float32 and casts to the weight dtype before the projection. The 2-D position table is
//  [2, position_embedding_size, hidden]: axis 0 indexed by the patch x coordinate, axis 1 by y.
//  (The reference zeroes the embedding of padding patches; the engine never carries padding.)
#include "mlx/mlx.h"
#include "ESLinear.h"

namespace es {
namespace mx = mlx::core;

class ESVisionPatchEmbedder {
public:
    ESVisionPatchEmbedder(ESLinear inputProj, mx::array positionTable);

    // pixelValues: float32 [n, 3*p*p]; positionIds: int32 [n, 2]. Returns [n, hidden].
    mx::array forward(const mx::array & pixelValues, const mx::array & positionIds) const;

private:
    ESLinear  inputProj_;
    mx::array table_;   // [2, P, hidden]
};

}  // namespace es
