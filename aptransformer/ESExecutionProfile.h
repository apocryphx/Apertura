#pragma once
//  ESExecutionProfile — HOW a Gemma-4 variant is run, as opposed to WHAT it computes.
//
//  The architecture (PLE, shared-KV layers, MoE routing, dual head_dim, k_eq_v, …) stays in the
//  layer classes, driven by config.json, so the forward math remains one faithful port of the
//  HuggingFace reference. The decisions below are execution policy: they change speed, never
//  results (each is exact, or a re-chunking that only moves bf16 near-ties). They are chosen
//  once per model family in ESExecutionProfile::forConfig (ESModelConfig.mm) and read from
//  config.profile, so every tuning decision and the measurement behind it live in one place.
//
//  Measurements: Apple M6 (12-core GPU, 24 GB), warm, median of 3 — see SESSION-2026-10-13.md.

namespace es {

class ESModelConfig;

enum class ESModelFamily {
    dense,     // 31B, 12B: one MLP per layer
    moe,       // 26B-A4B: routed experts
    elastic,   // E2B, E4B: per-layer embeddings + trailing shared-KV layers
};

struct ESExecutionProfile {
    ESModelFamily family = ESModelFamily::dense;

    // Default prefill chunk length (ESModelConfig::prefillChunk starts from it; CLI and
    // AperturaKit can still override). Elastic 1024: E2B +8% @4K / +14% @8K, E4B +5-9% vs 512
    // (small matmuls fill the GPU better, half as many chunk tails). Dense/MoE 512: 12B was 1-2%
    // slower at 1024.
    int prefillChunk = 512;

    // Elastic only: in prefill, run the shared-KV layers (which write no cache) on the last row
    // only, and skip them entirely in middle chunks (ESGemma4TextModel::forward lastOnly /
    // cacheOnly). Exact. E2B 2,477 -> 5,932 tok/s @2048 cold; middle-chunk skip +9.5% on top.
    bool sharedKVPrefillSkip = false;

    // MoE only: sort (token, expert) pairs by expert once a prefill has at least this many, so
    // gather_qmm runs one grouped GEMM per expert (NAX on M5+) instead of seq*k matvecs
    // (ESExperts::sparseForward; mlx-lm uses the same 64). 26B-A4B ~140 -> 626 tok/s @4096; the
    // sorted kernel is also closer to an fp32 reference (rel-L2 4.7e-3 vs 7.9e-3). 0 = never.
    int moeSortMinPairs = 0;

    // Tokens in the load-time warm-up prefill (APModel prewarmWithCompletion:) that builds MLX's
    // pipeline states before the first real prompt. 16 warms as fully as a full-length prefill
    // (E2B first 2K prefill 7,940 vs ~3,000-6,500 tok/s cold).
    int warmupPrefillTokens = 16;

    static ESExecutionProfile forConfig(const ESModelConfig & config);
    const char * familyName() const {
        switch (family) {
            case ESModelFamily::moe:     return "moe";
            case ESModelFamily::elastic: return "elastic";
            default:                     return "dense";
        }
    }
};

}  // namespace es
