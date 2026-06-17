#pragma once
//  ESExperts — Gemma4TextExperts. A bank of `num_experts` SwiGLU MLPs with 3D weights:
//    gate_up_proj [E, 2*moe_inter, hidden]   (gate and up packed together)
//    down_proj    [E, hidden, moe_inter]
//
//  forward(x, W): for the dense (research) path we evaluate ALL experts and combine by the router
//  weight matrix W [seq, E] (0 for unchosen experts) — out[s] = sum_e W[s,e] * down_e(gelu(gate_e(x))*up_e(x)).
//  This is exactly the PyTorch result (unchosen experts contribute 0). It reads every expert's
//  weights, so it is the correctness path, not the bandwidth-optimal one (a gather_mm fast path
//  that touches only the top-k experts is the MoE analogue of the fused/quantized perf work).
#include "mlx/mlx.h"

namespace es {
namespace mx = mlx::core;

class ESExperts {
public:
    // quantBits>0 also quantizes the 3D expert tensors (affine, group); the sparse path then uses
    // gather_qmm — the experts are ~88% of the 26B's weights, so this is the real MoE bandwidth lever.
    ESExperts(mx::array gateUp, mx::array down, int quantBits = 0, int groupSize = 64);

    // Dense: evaluate ALL experts, combine by the dense weight matrix W [seq,E]. Correctness path (bf16).
    mx::array forward(const mx::array & x, const mx::array & W) const;

    // Sparse: gather only the top-k experts the router picked. idx [seq,k] (int32), w [seq,k].
    // gather_mm (bf16) or gather_qmm (quantized) — reads ~top_k/num_experts of expert weights/token.
    mx::array sparseForward(const mx::array & x, const mx::array & idx, const mx::array & w) const;

private:
    mx::array gateUp_, down_;                 // bf16 (dense path; sparse bf16 path)
    bool      quant_;
    int       bits_, gs_;
    mx::array gateUpQ_, gateUpS_, gateUpB_;   // quantized gate_up (valid iff quant_)
    mx::array downQ_, downS_, downB_;         // quantized down
};

}  // namespace es
