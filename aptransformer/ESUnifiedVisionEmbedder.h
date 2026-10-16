#pragma once
//  ESUnifiedVisionEmbedder — Gemma4UnifiedVisionEmbedder (+ its Gemma4UnifiedMultimodalEmbedder):
//  the encoder-free image front end of the unified family (12B QAT).
//
//    x  = bf16(pixels)                      merged 48x48x3 patches, k/255, NO 2(x-0.5) scaling
//    h  = LN1(x) ; h = Dense(h) + b ; h = LN2(h)
//    h  = h + pos[x_block, 0] + pos[y_block, 1]     factorized 2-D table [1120, 2, 3840]
//    h  = LN3(h)                                  (pos_norm)
//    y  = Linear( RMSNorm_noscale(h) )            (embed_vision.embedding_projection, 3840 -> hidden)
//
//  nn.LayerNorm (eps 1e-5) is computed in float32 and rounded once to bf16, as torch does for a
//  bf16 input. Padding tokens (-1, -1) contribute no position embedding in the reference; the
//  engine never carries padding.
#include "mlx/mlx.h"
#include "ESVisionConfig.h"
#include "ESLinear.h"
#include "ESMultimodalEmbedder.h"
#include <string>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader;

// nn.LayerNorm with weight + bias.
class ESLayerNorm {
public:
    ESLayerNorm(mx::array weight, mx::array bias, float eps) : w_(std::move(weight)), b_(std::move(bias)), eps_(eps) {}
    mx::array forward(const mx::array & x) const;
private:
    mx::array w_, b_;
    float eps_;
};

class ESUnifiedVisionEmbedder {
public:
    ESUnifiedVisionEmbedder(const ESVisionConfig & cfg, const ESWeightLoader & w);

    // pixelValues: float32 [n, 6912]; positionIds: int32 [n, 2] (block x, y). Returns [n, textHidden].
    mx::array forward(const mx::array & pixelValues, const mx::array & positionIds) const;

    struct Trace { mx::array ln1, dense, ln2, posAdded, posNorm, softTokens; };
    Trace forwardTrace(const mx::array & pixelValues, const mx::array & positionIds) const;

private:
    mx::array run(const mx::array & pixelValues, const mx::array & positionIds, Trace * trace) const;

    ESLayerNorm ln1_, ln2_, posNorm_;
    mx::array   denseW_, denseB_;    // [3840, 6912], [3840]
    mx::array   posTable_;           // [1120, 2, 3840]
    ESMultimodalEmbedder embedder_;
};

}  // namespace es
