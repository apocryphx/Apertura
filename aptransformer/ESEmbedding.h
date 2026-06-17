#pragma once
//  ESEmbedding — the tied token embedding, used two ways:
//    lookup(ids) -> [n, hidden]      (input embedding; caller applies the sqrt(hidden) scale)
//    logits(h)   -> [seq, vocab]     (tied LM head: h @ W^T)
//
//  quantBits 0 -> bf16. quantBits 4/8 -> affine group quantization: the LM-head matmul uses
//  mx::quantized_matmul; the lookup gathers the quantized rows and dequantizes them. This is the
//  remaining per-token weight-bandwidth lever after the layer projections (~2.8 GB bf16).
#include "mlx/mlx.h"
#include <vector>

namespace es {
namespace mx = mlx::core;

class ESEmbedding {
public:
    ESEmbedding(mx::array weight, int quantBits, int groupSize);

    mx::array lookup(const std::vector<int> & ids) const;  // [n, hidden]
    mx::array logits(const mx::array & hidden) const;       // [seq, vocab]

private:
    bool      quant_;
    int       bits_, groupSize_;
    mx::array w_;                 // bf16 [vocab, hidden]  OR  packed quantized weight
    mx::array scales_, biases_;   // valid iff quant_
};

}  // namespace es
