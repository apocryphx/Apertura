#pragma once
//  ESModelConfig — Gemma-4 31B text decoder hyperparameters.
//
//  Parsed from the HF `config.json` (text_config). Dtype is a runtime property
//  (mx::Dtype), cast once at weight load — never a C++ template parameter.
//  Per-layer helpers encode the hybrid local/global routing that breaks naive ports.
//
//  ──────────────────────────────────────────────────────────────────────────
//  PERFORMANCE FINDINGS  (Gemma-4-31B QAT, Apple silicon, 128 GB unified memory)
//  ──────────────────────────────────────────────────────────────────────────
//  DECODE is MEMORY-BANDWIDTH-bound: one token = a batch-1 GEMV that reads every
//  weight once; the compute units idle waiting on memory, so the math finishing
//  faster buys nothing. Speed ≈ (bytes moved per token) / (memory bandwidth). The
//  only levers are "less data" or "fewer memory passes":
//
//    • quantBits = 4 (group 64): ~2.5x DECODE (measured 6.6 -> 16.8 tok/s vs bf16).
//      g64 = 4.5 bits/weight = q4_0 parity; g32 (5 bits) is quality-equivalent at
//      our sample but bigger/slower — prefer g64. NOTE: PREFILL is compute-bound,
//      so 4-bit is ~11% SLOWER there (203 -> 180 tok/s) — the dequant work isn't
//      hidden. 4-bit is a decode win, a small prefill loss.
//    • quantEmbedBits = 8: the tied LM head is precision-sensitive; keep it Q8.
//      (Q8 head ≈ +4% decode bandwidth vs Q4 — worth it. Q4/g64 sits ~95-98%
//      top-1 vs full-precision BF16 on real prompts, so quality holds.)
//    • fused = true: mx::fast SDPA (flash) + compile. ~5-14% decode, big PREFILL
//      win (flash avoids the O(L^2) score matrix). ALWAYS prefer it; argmax-stable.
//    • quantKVBits is a CAPACITY lever, NOT a speed one. The quantized-KV attention
//      path forgoes flash (MLX 0.31.2 has no quantized SDPA; flash XOR quant-KV),
//      and the two-call quantized path is SLOWER than flash+bf16-KV at EVERY
//      context <= 64K (measured, isolated attention + full model). Set it only to
//      fit a very long KV cache in RAM; for speed leave it 0. See ESAttention.
//    • preallocKVCache = true (default): slice_update KV storage. The old concat-grow
//      cache copied the whole cache every token AND defeated MLX's buffer cache with
//      monotonically growing sizes (a real Metal allocation per layer per token) —
//      ~6-7 ms/token at every context length, measured in isolation. Prealloc appends
//      in place via buffer donation (~0.9 ms/token, context-independent). Bit-exact
//      (--cache-verify; 4 gates incl. mid-decode turn appends + session byte-identity).
//      Decode +42% @1030 ctx fresh-process; also unblocks whole-step compile (P3).
//    • PREFIX CACHING (ESSession) is the DOMINANT win for long prompts / multi-turn:
//      a 13.5K-token persona re-prefills in ~128 s EVERY turn (≈104 tok/s at that
//      length), but primed ONCE it drops to ~3.8 s/turn — 33.7x. Bigger than all
//      weight/KV levers combined. See ESGenerationLoop / ESSession.
//
//  RECOMMENDED (e.g. the long-persona Isolde workload): weights Q4 g64 + embed Q8,
//  fused = true, quantKVBits = 0, and prime the persona once via ESSession.
//  vs the most-tuned llama.cpp (LM Studio) q4_0: ~86% of its decode t/s — the gap
//  is its hand-written q4_0 Metal kernels + our Q8 head, not anything fundamental.
//  ──────────────────────────────────────────────────────────────────────────
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
    // Sliding-window KV-cache eviction (decode): local/sliding layers only ever attend the last
    // `slidingWindow` keys (the rest are masked to -1e30 -> exp underflows to exactly 0), so during
    // single-token decode the older keys can be dropped from the cache with ZERO numerical change.
    // Bounds decode attention cost for 5/6 of the layers at long context. Bit-exact (verified
    // 129/129 & 65/65 greedy tokens, --swa-verify); 2.35x decode @2048 ctx, 3.44x @4096. ON by
    // default — never fires below the window (short-ctx & conformance unaffected). Disable via
    // --no-swa-cache for A/B. Gated off for elastic shared-KV and quant-KV paths internally.
    bool slidingWindowCache = true;
    // Preallocated KV storage: append via mx::slice_update into chunk-grown fixed-capacity
    // buffers instead of mx::concatenate. The legacy concat path copies the whole cache every
    // token AND its monotonically growing buffer sizes defeat MLX's buffer cache (a real Metal
    // allocation per layer per token) — measured ~6-7 ms/token at ALL context lengths, and the
    // entire pre-P1 long-context decode collapse (~48 ms/token @4096 in appends alone with
    // eviction off). Prealloc replaces that with amortized O(1/256) copies. Returned K/V are
    // bit-identical by construction (same values, same order; attention reads slice views —
    // MLX's SDPA takes strided K/V at batch 1). Verified token-exact via --cache-verify. ON by
    // default; --no-prealloc-cache for A/B. Quant-KV storage is unaffected (still concat).
    bool preallocKVCache = true;
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
