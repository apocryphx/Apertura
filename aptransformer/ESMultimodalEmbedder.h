#pragma once
//  ESMultimodalEmbedder — Gemma4MultimodalEmbedder: soft tokens -> language-model space.
//    y = embedding_projection( rmsnorm_noscale(x) )      768 -> text hidden (1536 on E2B)
#include "mlx/mlx.h"
#include "ESLinear.h"
#include "ESRMSNorm.h"

namespace es {
namespace mx = mlx::core;

class ESMultimodalEmbedder {
public:
    ESMultimodalEmbedder(ESLinear projection, float eps) : proj_(std::move(projection)), norm_(eps) {}
    mx::array forward(const mx::array & x) const { return proj_.forward(norm_.forward(x)); }

private:
    ESLinear  proj_;
    ESRMSNorm norm_;   // with_scale=false
};

}  // namespace es
