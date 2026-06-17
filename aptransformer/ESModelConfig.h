#pragma once
//  ESModelConfig — Gemma-4 31B text decoder hyperparameters.
//
//  Parsed from the HF `config.json` (text_config). Dtype is a runtime property
//  (mx::Dtype), cast once at weight load — never a C++ template parameter.
//  Per-layer helpers encode the hybrid local/global routing that breaks naive ports.
#include "mlx/mlx.h"
#include <string>
#include <vector>

namespace es {
namespace mx = mlx::core;

class ESModelConfig {
public:
    // Build from a model directory containing config.json (the HF snapshot dir).
    static ESModelConfig fromConfigJSON(const std::string & configJsonPath);

    // --- core dims ---
    int hiddenSize        = 5376;
    int numHiddenLayers   = 60;
    int numAttentionHeads = 32;
    int numKeyValueHeads  = 16;   // local (sliding) layers
    int numGlobalKVHeads  = 4;    // global (full) layers
    int headDim           = 256;  // local head dim
    int globalHeadDim     = 512;  // global head dim
    int intermediateSize  = 21504;
    int slidingWindow     = 1024;
    int vocabSize         = 262144;
    int maxPositionEmbeddings = 262144;

    float rmsNormEps              = 1e-6f;
    float finalLogitSoftcapping  = 30.0f;   // <=0 means disabled

    // --- rope ---
    float ropeThetaLocal   = 10000.0f;
    float ropeThetaGlobal  = 1000000.0f;
    float globalPartialRotaryFactor = 0.25f;

    // --- behavior flags ---
    bool  attentionKEqV = true;   // global layers: V reuses K projection (no v_proj weight)
    bool  tieWordEmbeddings = true;

    // --- Mixture-of-Experts (26B-a4b; the 31B dense model leaves this off) ---
    // When on, each decoder layer runs the dense MLP AND a sparse MoE block in parallel and sums
    // them. The router picks top_k of num_experts experts per token; experts are SwiGLU.
    bool  enableMoeBlock      = false;
    int   numExperts          = 0;
    int   topKExperts         = 0;
    int   moeIntermediateSize = 0;
    // false = dense (evaluate all experts; correctness path). true = sparse gather_mm (touch only
    // the top-k experts the router picks — reads ~top_k/num_experts of the expert weights/token).
    bool  moeSparse           = false;

    // --- compute ---
    mx::Dtype computeDtype = mx::bfloat16;

    // Dual-path: false = probe-unfused (the research instrument; every op materializes).
    // true = performance-fused (mx::fast kernels + mx::compile). Must stay argmax bit-stable.
    bool fused = false;

    // Weight quantization for the linear projections (q/k/v/o, gate/up/down).
    // 0 = bf16 (canonical); 4 or 8 = affine group quantization at load. Norms/scalars/embed
    // stay bf16. Cuts the ~58 GB/token decode bandwidth — the lever for this bandwidth-bound model.
    int quantBits      = 0;
    int quantGroupSize = 64;
    // Bits for the tied embedding / LM head (separate from the layers — the output projection is
    // precision-sensitive, so the standard scheme is layers Q4 + embed Q8). 0 = keep embed bf16.
    int quantEmbedBits = 0;

    // KV-cache quantization (0 = bf16 cache; 4/8 = quantized K/V via quantized_matmul attention).
    // The per-token cache read grows with context, so this is the long-context bandwidth lever.
    int quantKVBits = 0;

    // --- Per-Layer Embeddings (PLE) + shared-KV (the elastic E2B/E4B models) ---
    // PLE: an auxiliary embedding feeds a per-layer residual signal into every decoder layer.
    // 0 = off (31B, 26B). >0 = the per-layer embedding dim (E2B: 256).
    int hiddenSizePerLayerInput = 0;
    int vocabSizePerLayerInput  = 0;
    // shared-KV: the last `numKvSharedLayers` layers reuse the K/V of the last non-shared layer of
    // their type instead of computing their own (0 = off; E2B: 20).
    int numKvSharedLayers = 0;

    // layer_types[i] == true  => sliding (local);  false => full (global)
    std::vector<bool> layerIsSliding;

    // --- derived per-layer helpers ---
    bool isSliding(int layerIdx) const { return layerIsSliding[layerIdx]; }
    int  headDimFor(int layerIdx) const { return isSliding(layerIdx) ? headDim : globalHeadDim; }
    // attention_k_eq_v applies only to non-sliding (global) layers.
    bool kEqVFor(int layerIdx) const { return attentionKEqV && !isSliding(layerIdx); }
    // KV head count keys off k_eq_v (NOT sliding): global+k_eq_v layers use the unified global count;
    // every other layer uses num_key_value_heads. (For 31B/26B these coincide; for E2B they differ.)
    int  kvHeadsFor(int layerIdx) const { return kEqVFor(layerIdx) ? numGlobalKVHeads : numKeyValueHeads; }

    bool hasPLE() const { return hiddenSizePerLayerInput > 0; }
    int  firstKvSharedIdx() const { return numHiddenLayers - numKvSharedLayers; }
    bool isKvSharedLayer(int i) const { return numKvSharedLayers > 0 && i >= firstKvSharedIdx(); }
    // True iff layer i is the LAST non-shared layer of its type — it stores its K/V for the shared
    // layers of that type to reuse.
    bool storeFullLengthKv(int i) const {
        if (isKvSharedLayer(i)) return false;
        for (int j = i + 1; j < firstKvSharedIdx(); ++j)
            if (layerIsSliding[j] == layerIsSliding[i]) return false;  // a later non-shared layer of same type exists
        return numKvSharedLayers > 0;
    }
    float embedScale() const;  // sqrt(hiddenSize)
};

}  // namespace es
