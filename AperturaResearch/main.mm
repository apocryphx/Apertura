//  AperturaResearch — conformance + generation driver for the Gemma-4 31B text decoder.
//
//  Usage: AperturaResearch [modelDir] [fixturesPath]
//    modelDir     HF snapshot dir (config.json + shards). Default: the cached gemma-4-31b-it.
//    fixturesPath fixtures.safetensors from Tools/generate_fixtures.py.
//
//  Loads weights, runs one forward pass over the fixture prompt, and compares Apertura's
//  scaled embedding, every decoder-layer output, the final norm, and the logits against the
//  PyTorch reference. Then greedy-generates and checks the token-id sequence matches.

#import <Foundation/Foundation.h>
#include "mlx/mlx.h"

#include "ESModelConfig.h"
#include "ESWeightLoader.h"
#include "ESGemma4TextForCausalLM.h"
#include "ESGenerationLoop.h"
#include "ESSampler.h"
#include "ESConformance.h"
#include "ESRMSNorm.h"
#include "ESRotaryEmbedding.h"
#include "ESAttention.h"
#include "ESMLPBlock.h"
#include "ESTokenizer.h"
#include "ESChatTemplate.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

namespace mx = mlx::core;

static bool hasFlag(int argc, const char ** argv, const char * f) {
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], f) == 0) return true;
    return false;
}

// Value of "--flag <value>", or "" if absent.
static std::string argValue(int argc, const char ** argv, const char * f) {
    for (int i = 1; i + 1 < argc; ++i) if (std::strcmp(argv[i], f) == 0) return argv[i + 1];
    return "";
}

static double secsSince(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

// Times prefill (process P tokens) and decode (generate D tokens) throughput for one LM.
static void benchOne(const es::ESGemma4TextForCausalLM & lm, const char * label, int P, int D) {
    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;  // arbitrary in-vocab ids

    // Prefill (1 warmup + best of 2). Use lastLogits (last-position LM head) — this is what real
    // generation does for time-to-first-token, and it keeps prefill memory bounded at long context
    // (a full-sequence [P, vocab] logits tensor is multi-GB once P reaches ~10k).
    double bestPre = 1e9;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        mx::array logits = lm.lastLogits(toks, nullptr, 0);
        mx::eval(logits);
        if (it > 0) bestPre = std::min(bestPre, secsSince(t0));
    }

    // Decode from a populated cache. Run twice and measure only the second pass — the first pays
    // MLX kernel compilation, gather_mm dispatch setup, and allocation warmup; the second is steady state.
    auto decodePass = [&]() -> double {
        es::ESKVCache cache(lm.config().numHiddenLayers);
        mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        int pos = P, next = es::ESSampler::argmax(ll);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int d = 0; d < D; ++d) {
            ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
            pos += 1; next = es::ESSampler::argmax(ll);
        }
        return secsSince(t0);
    };
    decodePass();                  // warmup (discarded)
    double dt = decodePass();      // measured
    std::printf("[%-9s] prefill %d tok: %6.1f tok/s (%.3fs)   decode %d tok: %6.1f tok/s (%.3fs)\n",
                label, P, P / bestPre, bestPre, D, D / dt, dt);
}

// A/B decode: sync path (host readback per token, via ESSampler::argmax + rebuild) vs async path
// (token stays on-device via argmaxDev/lastLogitsDev + mx::async_eval, so token N+1's graph is
// built while the GPU still runs N). Both greedy -> the token streams MUST match; we verify that,
// then report the throughput delta. Isolates the per-token CPU<->GPU sync stall the Metal trace found.
static void benchAsyncOne(const es::ESGemma4TextForCausalLM & lm, const char * label, int P, int D) {
    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;   // arbitrary in-vocab ids (same as benchOne)
    const int L = lm.config().numHiddenLayers;

    // ---- sync decode (baseline): eval + item() readback every token ----
    auto syncPass = [&](std::vector<int> * capture) -> double {
        es::ESKVCache cache(L);
        mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        int pos = P, next = es::ESSampler::argmax(ll);
        if (capture) capture->push_back(next);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int d = 0; d < D; ++d) {
            ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
            pos += 1; next = es::ESSampler::argmax(ll);
            if (capture) capture->push_back(next);
        }
        return secsSince(t0);
    };

    // ---- async decode: keep the sampled id on-device, overlap consecutive steps ----
    //  compiledTail=true additionally routes each layer through its per-instance mx::compile'd
    //  stateless tail (post-attn norm + MLP sandwich + layer_scalar), encoded once vs re-traced.
    auto asyncPass = [&](bool compiledTail, std::vector<int> * capture) -> double {
        es::ESKVCache cache(L);
        mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        mx::array tok = es::ESSampler::argmaxDev(ll);          // int32 [1], on device
        mx::eval(tok);
        int pos = P;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int d = 0; d < D; ++d) {
            ll  = lm.lastLogitsDev(tok, &cache, pos, compiledTail); pos += 1;
            tok = es::ESSampler::argmaxDev(ll);
            mx::async_eval(tok);                               // non-blocking: overlap next build
        }
        mx::eval(tok);                                         // final barrier before we stop timing
        double dt = secsSince(t0);
        if (capture) {                                         // re-run capturing ids (host reads ok here)
            es::ESKVCache c2(L);
            mx::array l2 = lm.lastLogits(toks, &c2, 0); mx::eval(l2);
            mx::array t2 = es::ESSampler::argmaxDev(l2); int p2 = P;
            for (int d = 0; d < D; ++d) {
                mx::eval(t2); capture->push_back((int) t2.item<int32_t>());
                l2 = lm.lastLogitsDev(t2, &c2, p2, compiledTail); p2 += 1; t2 = es::ESSampler::argmaxDev(l2);
            }
        }
        return dt;
    };

    // warmup each (also triggers the one-time compile), then measure
    syncPass(nullptr);  asyncPass(false, nullptr);  asyncPass(true, nullptr);
    std::vector<int> syncIds, asyncIds, compIds;
    double dtSync  = syncPass(&syncIds);
    double dtAsync = asyncPass(false, &asyncIds);
    double dtComp  = asyncPass(true,  &compIds);

    auto matchFrac = [&](const std::vector<int> & a) {
        size_t n = std::min(syncIds.size(), a.size()), m = 0;
        for (size_t i = 0; i < n; ++i) if (syncIds[i] == a[i]) m++;
        return std::make_pair(m, n);
    };
    auto [ma, na] = matchFrac(asyncIds);
    auto [mc, nc] = matchFrac(compIds);

    std::printf("[%-9s] decode %d tok\n", label, D);
    std::printf("   sync          %6.1f tok/s (%.3fs)\n", D / dtSync, dtSync);
    std::printf("   async         %6.1f tok/s (%.3fs)  %.2fx  tokens %s (%zu/%zu)\n",
                D / dtAsync, dtAsync, dtSync / dtAsync, (ma == na && na) ? "MATCH" : "DIVERGE", ma, na);
    std::printf("   async+compile %6.1f tok/s (%.3fs)  %.2fx  tokens %s (%zu/%zu)\n",
                D / dtComp, dtComp, dtSync / dtComp, (mc == nc && nc) ? "MATCH" : "DIVERGE", mc, nc);
}

// P1 verification + measurement: sliding-window KV-cache eviction. Runs the SAME greedy decode with
// the eviction OFF vs ON (both fused), from a prompt LONGER than the window so eviction actually
// fires. Gate 1 (correctness): the two token streams MUST be identical (eviction is bit-exact — the
// dropped keys were masked to -1e30). Gate 2 (perf): decode tok/s off vs on shows the long-context win.
static void swaVerify(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    es::ESModelConfig cOff = base; cOff.fused = true; cOff.slidingWindowCache = false;
    es::ESModelConfig cOn  = base; cOn.fused  = true; cOn.slidingWindowCache = true;
    es::ESGemma4TextForCausalLM lmOff(cOff, weights), lmOn(cOn, weights);
    const int L = base.numHiddenLayers, W = base.slidingWindow;

    auto prompt = [](int n) { std::vector<int> t(n); for (int i = 0; i < n; ++i) t[i] = 100 + i; return t; };

    // ---- gate 1: NULL CONTROL ----------------------------------------------------------
    // Below the window, eviction never fires, so the two arms are literally the same
    // computation and MUST agree bit-for-bit. If this fails the harness is broken and
    // nothing below it means anything.
    int Pn = std::min(P, std::max(8, W / 2));
    std::vector<int> nOff, nOn;
    {
        auto run = [&](const es::ESGemma4TextForCausalLM & lm, std::vector<int> & out) {
            es::ESKVCache cache(L);
            std::vector<int> toks = prompt(Pn);
            mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
            int pos = Pn, next = es::ESSampler::argmax(ll);
            out.push_back(next);
            for (int d = 0; d < 8; ++d) {
                ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
                pos += 1; next = es::ESSampler::argmax(ll); out.push_back(next);
            }
        };
        run(lmOff, nOff); run(lmOn, nOn);
    }
    bool nullOk = (nOff == nOn);

    // ---- gate 2: LOCKSTEP -------------------------------------------------------------
    // Both arms are driven along ONE token stream, so a single early divergence cannot
    // cascade into an unrelated continuation. (Free-running reported 7/129 for a divergence
    // that is really ~1.5% of steps.)
    //
    // The gate is NOT token identity. Eviction drops keys whose softmax weight is exactly
    // zero, so the result is mathematically unchanged — but the arms contract over a
    // different NUMBER of keys, which changes the reduction tree and the GEMM tiling, so the
    // surviving terms are summed in a different order. That is ~1 ULP per layer and it
    // compounds. Eviction is numerically EQUIVALENT, not bit-exact.
    //
    // What would be a real fault is a flip on a decision the model was CONFIDENT about. So
    // each flip is checked against the arithmetic: if the top-2 margin exceeds the observed
    // |dlogit| at that step, the divergence cannot be explained by accumulation.
    es::ESKVCache cOffCache(L), cOnCache(L);
    std::vector<int> toks = prompt(P);
    mx::array llOff = lmOff.lastLogits(toks, &cOffCache, 0); mx::eval(llOff);
    mx::array llOn  = lmOn.lastLogits(toks,  &cOnCache,  0); mx::eval(llOn);

    double maxAbs = 0, meanAbs = 0;
    int flips = 0, unexplained = 0, pos = P, next = es::ESSampler::argmax(llOff);
    for (int d = 0; d < D; ++d) {
        llOff = lmOff.lastLogits({next}, &cOffCache, pos);
        llOn  = lmOn.lastLogits({next},  &cOnCache,  pos);
        pos += 1;
        mx::array dl = mx::abs(mx::subtract(mx::astype(llOff, mx::float32),
                                            mx::astype(llOn,  mx::float32)));
        mx::array dmaxA = mx::max(dl), dmeanA = mx::mean(dl);
        int aOff = es::ESSampler::argmax(llOff), aOn = es::ESSampler::argmax(llOn);
        mx::eval(dmaxA, dmeanA);
        double dmax = dmaxA.item<float>();
        maxAbs = std::max(maxAbs, dmax); meanAbs += dmeanA.item<float>();
        if (aOff != aOn) {
            flips++;
            // top-2 margin on the reference (off) arm at this step
            mx::array t2 = mx::topk(mx::astype(llOff, mx::float32), 2);
            mx::eval(t2);
            mx::array srt = mx::sort(t2);
            mx::eval(srt);
            double margin = std::abs(srt.data<float>()[1] - srt.data<float>()[0]);
            bool explained = (margin <= dmax);
            if (!explained) unexplained++;
            if (flips <= 5)
                std::printf("  step %4d (pos %5d): flip off=%d on=%d  margin=%.3e |dlogit|max=%.3e  %s\n",
                            d, pos - 1, aOff, aOn, margin, dmax,
                            explained ? "explained by accumulation" : "UNEXPLAINED");
        }
        next = aOff;
    }

    bool ok = nullOk && (unexplained == 0);
    std::printf("\n-- sliding-window eviction verify (prefill %d, window %d, %d steps) --\n", P, W, D);
    if (P <= W)
        std::printf("  WARNING: prefill %d <= window %d — eviction never fires; use a larger --prefill\n", P, W);
    std::printf("  null control (prefill %d < window, eviction inert): %s\n",
                Pn, nullOk ? "bit-identical PASS" : "DIVERGED  FAIL <- harness bug, not a model bug");
    std::printf("  lockstep: max |dlogit| %.3e  mean %.3e  flips %d/%d (%.2f%%), unexplained %d\n",
                maxAbs, meanAbs / D, flips, D, 100.0 * flips / D, unexplained);
    std::printf("  gate: %s  (eviction is numerically equivalent, not bit-exact)\n",
                ok ? "PASS" : "FAIL");
}

// Decode-shape numerics: fused vs unfused attention, in lockstep on a forced token stream.
//
// Why this is a different measurement from the attn_oproj probe. That probe runs a pure prefill
// forward at the fixture's seq_len, and at those shapes mx::fast::scaled_dot_product_attention
// does NOT engage a fused kernel at all: sdpa_full supports head_dim {64,80,96,128} (Apertura is
// 256/512) and sdpa_vector requires seq <= 8. use_fallback returns true, so "fused" there is
// MLX's fallback COMPOSITION versus Apertura's manual one — two composed graphs, differing mainly
// in GQA handling (MLX broadcasts 5-D, Apertura materialises repeatKV 3-D), hence different GEMM
// tiling and reduction order.
//
// At decode the routing changes: seq=1 with gqa 2 satisfies (seq * gqa) <= 32 and head_dim 256 is
// supported, so LOCAL layers hit the real fused vector kernel. Global layers (head_dim 512) still
// fall back. That mix is what production generation actually runs, and nothing measured it.
//
// Gate: unfused is the reference-faithful path (98.8-100% within 1 ULP of PyTorch at prefill), so
// this reports how far the fused path moves from it, per decode step.
static void fusedLockstep(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    es::ESModelConfig cOff = base; cOff.fused = false;
    es::ESModelConfig cOn  = base; cOn.fused  = true;
    es::ESGemma4TextForCausalLM lmOff(cOff, weights), lmOn(cOn, weights);
    const int L = base.numHiddenLayers;

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;

    es::ESKVCache cacheOff(L), cacheOn(L);
    mx::array llOff = lmOff.lastLogits(toks, &cacheOff, 0); mx::eval(llOff);
    mx::array llOn  = lmOn.lastLogits(toks,  &cacheOn,  0); mx::eval(llOn);

    // prefill-shape divergence, for contrast with the decode steps below
    mx::array dPre = mx::max(mx::abs(mx::subtract(mx::astype(llOff, mx::float32),
                                                  mx::astype(llOn, mx::float32))));
    mx::eval(dPre);
    double preMax = dPre.item<float>();

    double maxAbs = 0, meanAbs = 0;
    int flips = 0, pos = P, next = es::ESSampler::argmax(llOff);
    for (int d = 0; d < D; ++d) {
        llOff = lmOff.lastLogits({next}, &cacheOff, pos);
        llOn  = lmOn.lastLogits({next},  &cacheOn,  pos);
        pos += 1;
        mx::array dl = mx::abs(mx::subtract(mx::astype(llOff, mx::float32),
                                            mx::astype(llOn,  mx::float32)));
        mx::array dmaxA = mx::max(dl), dmeanA = mx::mean(dl);
        int aOff = es::ESSampler::argmax(llOff), aOn = es::ESSampler::argmax(llOn);
        mx::eval(dmaxA, dmeanA);
        maxAbs = std::max(maxAbs, (double) dmaxA.item<float>());
        meanAbs += dmeanA.item<float>();
        if (aOff != aOn) {
            flips++;
            if (flips <= 5)
                std::printf("  step %4d (pos %5d): argmax flip unfused=%d fused=%d  maxD=%.3e\n",
                            d, pos - 1, aOff, aOn, (double) dmaxA.item<float>());
        }
        next = aOff;
    }
    std::printf("\n-- fused vs unfused, decode shapes (prefill %d, %d steps, forced stream) --\n", P, D);
    std::printf("  routing: local head_dim %d -> vector kernel at seq=1; global head_dim %d -> fallback\n",
                base.headDim, base.globalHeadDim);
    std::printf("  prefill-shape (seq %d, both fall back): max |dlogit| %.3e\n", P, preMax);
    std::printf("  decode-shape: max |dlogit| %.3e  mean %.3e  argmax flips %d/%d (%.2f%%)\n",
                maxAbs, meanAbs / D, flips, D, 100.0 * flips / D);
}

// P1 numerics: sliding-window eviction off vs on, in lockstep on a FORCED token stream, so a
// single early divergence cannot cascade into an unrelated continuation. Reports |dlogit| and the
// argmax flip rate rather than token identity.
//
// Why token identity is the wrong gate here: eviction drops keys that were masked to -1e30, so the
// dropped terms carry weight exactly 0 and the RESULT is mathematically unchanged. But the two arms
// contract over a different NUMBER of keys (~P vs window), which changes the reduction tree shape
// and the GEMM tiling, so the surviving non-zero terms are summed in a different order. That is
// ~1 ULP per layer, and over numHiddenLayers it compounds into a logit delta large enough to flip
// an argmax whenever the top-2 margin is smaller than it. Eviction is therefore numerically
// EQUIVALENT, not bit-exact; this measures the size of that equivalence.
static void swaLockstep(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    es::ESModelConfig cOff = base; cOff.fused = true; cOff.slidingWindowCache = false;
    es::ESModelConfig cOn  = base; cOn.fused  = true; cOn.slidingWindowCache = true;
    es::ESGemma4TextForCausalLM lmOff(cOff, weights), lmOn(cOn, weights);
    const int L = base.numHiddenLayers;

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;

    es::ESKVCache cacheOff(L), cacheOn(L);
    mx::array llOff = lmOff.lastLogits(toks, &cacheOff, 0); mx::eval(llOff);
    mx::array llOn  = lmOn.lastLogits(toks,  &cacheOn,  0); mx::eval(llOn);

    double maxAbs = 0, meanAbs = 0;
    int flips = 0, pos = P;
    int next = es::ESSampler::argmax(llOff);
    for (int d = 0; d < D; ++d) {
        llOff = lmOff.lastLogits({next}, &cacheOff, pos);
        llOn  = lmOn.lastLogits({next},  &cacheOn,  pos);
        pos += 1;
        mx::array dl = mx::abs(mx::subtract(mx::astype(llOff, mx::float32),
                                            mx::astype(llOn,  mx::float32)));
        mx::array dmaxA = mx::max(dl), dmeanA = mx::mean(dl);
        int aOff = es::ESSampler::argmax(llOff), aOn = es::ESSampler::argmax(llOn);
        mx::eval(dmaxA, dmeanA);
        double dmax = dmaxA.item<float>(), dmean = dmeanA.item<float>();
        maxAbs = std::max(maxAbs, dmax); meanAbs += dmean;
        if (aOff != aOn) {
            flips++;
            if (flips <= 5)
                std::printf("  step %4d (pos %5d): argmax flip off=%d on=%d  maxD=%.3e meanD=%.3e\n",
                            d, pos - 1, aOff, aOn, dmax, dmean);
        }
        next = aOff;   // both arms continue along the OFF stream
    }
    std::printf("\n-- sliding-window lockstep numerics (prefill %d, window %d, %d steps, forced stream) --\n",
                P, base.slidingWindow, D);
    if (P <= base.slidingWindow)
        std::printf("  WARNING: prefill %d <= window %d — eviction never fires\n", P, base.slidingWindow);
    std::printf("  max |dlogit| %.3e   mean |dlogit| %.3e   argmax flips %d/%d (%.2f%%)\n",
                maxAbs, meanAbs / D, flips, D, 100.0 * flips / D);
}

// Cache-redesign verification + measurement: legacy concat-grow KV storage vs preallocated
// slice_update storage (ESModelConfig::preallocKVCache). Runs the SAME greedy decode on both
// (fused, sliding eviction as configured) and requires identical token streams. To exercise every
// structural boundary of the prealloc mode, use decode > ESKVCache::kGrowChunk (crosses capacity
// growth on global layers and compaction on sliding layers) and prefill > window (fires eviction).
static void cacheVerify(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    es::ESModelConfig cOld = base; cOld.fused = true; cOld.preallocKVCache = false;
    es::ESModelConfig cNew = base; cNew.fused = true; cNew.preallocKVCache = true;
    es::ESGemma4TextForCausalLM lmOld(cOld, weights), lmNew(cNew, weights);  // share weight arrays

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;
    const int L = base.numHiddenLayers;

    auto run = [&](const es::ESGemma4TextForCausalLM & lm, std::vector<int> * out, double * preS) -> double {
        es::ESKVCache cache(L);
        auto tp = std::chrono::high_resolution_clock::now();
        mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        if (preS) *preS = secsSince(tp);
        int pos = P, next = es::ESSampler::argmax(ll);
        if (out) out->push_back(next);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int d = 0; d < D; ++d) {
            if (d == D / 2) {
                // Mid-decode multi-token append (a "turn" prefill): exercises seq>1 appends after
                // sliding eviction has advanced — the ESSession transition. Same on both arms, so
                // the token streams stay comparable.
                ll = lm.lastLogits({100, 101, 102, 103, 104}, &cache, pos); mx::eval(ll);
                pos += 5; next = es::ESSampler::argmax(ll);
                if (out) out->push_back(next);
            }
            ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
            pos += 1; next = es::ESSampler::argmax(ll);
            if (out) out->push_back(next);
        }
        return secsSince(t0);
    };

    run(lmOld, nullptr, nullptr); run(lmNew, nullptr, nullptr);   // warmup (JIT compile), discarded
    std::vector<int> tOld, tNew;
    double preOld = 0, preNew = 0;
    double dtOld = run(lmOld, &tOld, &preOld);
    double dtNew = run(lmNew, &tNew, &preNew);

    size_t n = std::min(tOld.size(), tNew.size()), match = 0;
    for (size_t i = 0; i < n; ++i) if (tOld[i] == tNew[i]) match++;
    bool ok = (match == n) && (n > 0);
    std::printf("\n-- prealloc KV cache verify (prefill %d, window %d, decode %d, chunk %d) --\n",
                P, base.slidingWindow, D, es::ESKVCache::kGrowChunk);
    if (D <= es::ESKVCache::kGrowChunk)
        std::printf("  NOTE: decode %d <= chunk %d — growth/compaction boundaries not crossed\n",
                    D, es::ESKVCache::kGrowChunk);
    if (P <= base.slidingWindow)
        std::printf("  NOTE: prefill %d <= window %d — sliding eviction never fires\n",
                    P, base.slidingWindow);
    std::printf("  bit-exactness: %zu/%zu greedy tokens match  %s\n", match, n, ok ? "PASS" : "FAIL");
    std::printf("  prefill: legacy %6.1f tok/s (%.3fs)   prealloc %6.1f tok/s (%.3fs)\n",
                P / preOld, preOld, P / preNew, preNew);
    std::printf("  decode : legacy %6.1f tok/s (%.3fs)   prealloc %6.1f tok/s (%.3fs)   speedup %.2fx\n",
                D / dtOld, dtOld, D / dtNew, dtNew, dtOld / dtNew);
}

// P3 verification + measurement: whole-step compiled decode (ESCompiledStep) vs the eager decode
// loop, same greedy stream from the same prefill. Each arm decodes WARM+D tokens from one prefill
// (the first WARM are timed separately — they absorb the one-time trace/JIT); the FULL streams
// must match. Use decode > ESCompiledStep::kGlobalChunk to also cross a growth re-trace, and
// > kSlidingHeadroom to cross a sliding compaction.
static void stepVerify(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    es::ESModelConfig c = base; c.fused = true;
    es::ESGemma4TextForCausalLM lm(c, weights);
    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;
    const int L = base.numHiddenLayers, WARM = 16;

    auto eagerArm = [&](std::vector<int> * out, double * meas) -> void {
        es::ESKVCache cache(L);
        mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        int pos = P, next = es::ESSampler::argmax(ll);
        out->push_back(next);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int d = 0; d < WARM + D; ++d) {
            if (d == WARM) t0 = std::chrono::high_resolution_clock::now();
            ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
            pos += 1; next = es::ESSampler::argmax(ll);
            out->push_back(next);
        }
        *meas = secsSince(t0);
    };
    auto stepArm = [&](std::vector<int> * out, double * meas) -> void {
        es::ESKVCache cache(L);
        mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        int next = es::ESSampler::argmax(ll);
        out->push_back(next);
        es::ESCompiledStep step(lm, &cache, P, c.preallocKVCache);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int d = 0; d < WARM + D; ++d) {
            if (d == WARM) t0 = std::chrono::high_resolution_clock::now();
            ll = step.step(next); mx::eval(ll);
            next = es::ESSampler::argmax(ll);
            out->push_back(next);
        }
        *meas = secsSince(t0);
    };

    std::vector<int> tE, tS;
    double dtE = 0, dtS = 0;
    eagerArm(&tE, &dtE);
    stepArm(&tS, &dtS);

    size_t n = std::min(tE.size(), tS.size()), match = 0;
    for (size_t i = 0; i < n; ++i) if (tE[i] == tS[i]) match++;
    bool ok = (match == n) && (n > 0);
    std::printf("\n-- compiled-step verify (prefill %d, decode %d+%d warm, window %d, headroom %d, gchunk %d) --\n",
                P, D, WARM, base.slidingWindow, es::ESCompiledStep::kSlidingHeadroom,
                es::ESCompiledStep::kGlobalChunk);
    std::printf("  bit-exactness: %zu/%zu greedy tokens match  %s\n", match, n, ok ? "PASS" : "FAIL");
    std::printf("  decode (last %d, eager warmed / step traced): eager %6.1f tok/s (%.3fs)   compiled-step %6.1f tok/s (%.3fs)   speedup %.2fx\n",
                D, D / dtE, dtE, D / dtS, dtS, dtE / dtS);
    std::printf("  NOTE: arms measured sequentially in one process — for iso-thermal absolutes use\n");
    std::printf("        cold fresh-process pairs (--bench-step vs --bench, Tools/hidtemp gating).\n");
}

// Fresh-process compiled-step decode bench arm: one prefill, WARM discarded steps (absorbs the
// trace + JIT), D measured steps. Pair with `--bench` (fused row) from another cold process.
static void benchStep(const es::ESGemma4TextForCausalLM & lm, int P, int D) {
    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;
    const int WARM = 32;
    es::ESKVCache cache(lm.config().numHiddenLayers);
    auto tp = std::chrono::high_resolution_clock::now();
    mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
    double preS = secsSince(tp);
    int next = es::ESSampler::argmax(ll);
    es::ESCompiledStep step(lm, &cache, P, lm.config().preallocKVCache);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int d = 0; d < WARM + D; ++d) {
        if (d == WARM) t0 = std::chrono::high_resolution_clock::now();
        ll = step.step(next); mx::eval(ll);
        next = es::ESSampler::argmax(ll);
    }
    double dt = secsSince(t0);
    std::printf("[step     ] prefill %d tok: %6.1f tok/s (%.3fs)   decode %d tok: %6.1f tok/s (%.3fs)\n",
                P, P / preS, preS, D, D / dt, dt);
}

// Structure-matched EAGER decode bench arm (fused only): one prefill, WARM discarded steps, D
// measured — the exact shape of benchStep, so `--bench-eager` vs `--bench-step` cold pairs are
// apples-to-apples. (`--bench` runs its unfused arm first in-process, which pollutes the buffer
// pool and heats the die before the fused row — a measured ~1-5 tok/s tax; see roadmap §6.)
static void benchEager(const es::ESGemma4TextForCausalLM & lm, int P, int D,
                       const std::string & snap = "", const std::string & snapFp = "") {
    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;
    const int WARM = 32;
    es::ESKVCache cache(lm.config().numHiddenLayers);

    // --kv-snapshot: restore the post-prefill cache instead of recomputing it (decode-only
    // benchmarking at depth: a 60K prefill is ~10 min AND heats the die, thermally polluting
    // the decode row it precedes). On a fingerprint miss, prefill normally and save for next
    // time. When restored, the first decode token is forced (100) — decode timing is
    // shape-dependent, not content-dependent, and both arms restore the same stream.
    double preS = 0; int pos = P, next;
    bool restored = false;
    const es::ESKVCache::RawMode mode = lm.config().rawKV
        ? (lm.config().rawKVQ8 ? es::ESKVCache::RawMode::rawQ8 : es::ESKVCache::RawMode::raw)
        : es::ESKVCache::RawMode::composite;
    if (!snap.empty() && cache.restoreSnapshot(snap, snapFp, mode) == P) {
        restored = true; next = 100;
        std::printf("[eager    ] prefill %d tok: restored from %s\n", P, snap.c_str());
    }
    mx::array ll = mx::array(0.0f);
    if (!restored) {
        auto tp = std::chrono::high_resolution_clock::now();
        ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        preS = secsSince(tp);
        next = es::ESSampler::argmax(ll);
        if (!snap.empty()) {
            bool ok = cache.saveSnapshot(snap, snapFp, P);
            std::printf("[eager    ] kv snapshot %s: %s\n", snap.c_str(), ok ? "saved" : "SAVE FAILED");
        }
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int d = 0; d < WARM + D; ++d) {
        if (d == WARM) t0 = std::chrono::high_resolution_clock::now();
        ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
        pos += 1; next = es::ESSampler::argmax(ll);
    }
    double dt = secsSince(t0);
    if (restored)
        std::printf("[eager    ] prefill %d tok: (restored)   decode %d tok: %6.1f tok/s (%.3fs)\n",
                    P, D, D / dt, dt);
    else
        std::printf("[eager    ] prefill %d tok: %6.1f tok/s (%.3fs)   decode %d tok: %6.1f tok/s (%.3fs)\n",
                    P, P / preS, preS, D, D / dt, dt);
}

// P3 numerics diagnostic: run eager and compiled-step over the SAME forced token stream (the
// eager arm's greedy choices) and compare the logits vectors at every step — no stream forking.
// Reports max/mean absolute logit difference and how often the argmax disagrees. Distinguishes
// e-scale reassociation drift (fixed-capacity kernels reduce in a different order than
// length-exact ones) from a real correctness bug (large differences).
static void stepLockstep(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    es::ESModelConfig c = base; c.fused = true;
    es::ESGemma4TextForCausalLM lm(c, weights);
    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;
    const int L = base.numHiddenLayers;

    // Eager pass: record the greedy stream and keep each step's logits (host copies are too big;
    // instead re-run compiled with forced tokens and diff on-device per step).
    es::ESKVCache cacheE(L);
    mx::array llE = lm.lastLogits(toks, &cacheE, 0); mx::eval(llE);
    std::vector<int> stream;
    stream.push_back(es::ESSampler::argmax(llE));

    es::ESKVCache cacheS(L);
    mx::array llS = lm.lastLogits(toks, &cacheS, 0); mx::eval(llS);
    es::ESCompiledStep step(lm, &cacheS, P, c.preallocKVCache);

    double maxAbs = 0, meanAbs = 0;
    int flips = 0, pos = P;
    for (int d = 0; d < D; ++d) {
        int tok = stream.back();
        llE = lm.lastLogits({tok}, &cacheE, pos); pos += 1;
        llS = step.step(tok);
        mx::array diff = mx::max(mx::abs(mx::subtract(mx::astype(llE, mx::float32),
                                                      mx::astype(llS, mx::float32))));
        mx::array mean = mx::mean(mx::abs(mx::subtract(mx::astype(llE, mx::float32),
                                                       mx::astype(llS, mx::float32))));
        int aE = es::ESSampler::argmax(llE), aS = es::ESSampler::argmax(llS);
        mx::eval(diff, mean);
        double dmax = diff.item<float>(), dmean = mean.item<float>();
        maxAbs = std::max(maxAbs, dmax); meanAbs += dmean;
        if (aE != aS) {
            flips++;
            if (flips <= 5)
                std::printf("  step %4d (pos %5d): argmax flip eager=%d step=%d  maxD=%.3e meanD=%.3e\n",
                            d, pos - 1, aE, aS, dmax, dmean);
        }
        stream.push_back(aE);  // continue along the EAGER stream in both arms
    }
    std::printf("\n-- compiled-step lockstep numerics (prefill %d, %d steps, forced eager stream) --\n", P, D);
    std::printf("  max |dlogit| %.3e   mean |dlogit| %.3e   argmax flips %d/%d (%.2f%%)\n",
                maxAbs, meanAbs / D, flips, D, 100.0 * flips / D);
}

// P4 quality gate: Q8 (bundle) head vs Q4 (re-quantized) head, TEACHER-FORCED on the Q8 arm's
// greedy stream (both models share every layer weight; only the embed/LM-head packing differs —
// the same measurement pattern as --vs-bf16). Reports next-token argmax agreement + logit deltas.
// This is a QUALITY/SPEED TRADE gate, not a bit-exactness gate: Q4 changes the head weights.
static void headVerify(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    // Compare the bundle head (Q8) against --quant-embed N's re-quantized head (default Q4;
    // pass --quant-embed 6 to gate the Q6 trade — llama.cpp's q4_0 GGUFs ship ~Q6_K heads).
    const int bits = (base.quantEmbedBits > 0 && base.quantEmbedBits != 8) ? base.quantEmbedBits : 4;
    es::ESModelConfig c8 = base; c8.fused = true; c8.quantEmbedBits = 0;    // 0 -> bundle verbatim (Q8)
    es::ESModelConfig c4 = base; c4.fused = true; c4.quantEmbedBits = bits;
    es::ESGemma4TextForCausalLM lm8(c8, weights), lm4(c4, weights);         // layer arrays shared

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;
    const int L = base.numHiddenLayers;

    es::ESKVCache cache8(L), cache4(L);
    mx::array ll8 = lm8.lastLogits(toks, &cache8, 0);
    mx::array ll4 = lm4.lastLogits(toks, &cache4, 0);
    mx::eval(ll8, ll4);
    int pos = P, tok = es::ESSampler::argmax(ll8);
    int agree = (tok == es::ESSampler::argmax(ll4)) ? 1 : 0;
    double maxAbs = 0, meanAbs = 0;
    for (int d = 0; d < D; ++d) {
        ll8 = lm8.lastLogits({tok}, &cache8, pos);
        ll4 = lm4.lastLogits({tok}, &cache4, pos);
        pos += 1;
        mx::array dv = mx::abs(mx::subtract(mx::astype(ll8, mx::float32), mx::astype(ll4, mx::float32)));
        mx::array dmax = mx::max(dv), dmean = mx::mean(dv);
        int a8 = es::ESSampler::argmax(ll8), a4 = es::ESSampler::argmax(ll4);
        mx::eval(dmax, dmean);
        if (a8 == a4) agree++;
        maxAbs = std::max(maxAbs, (double) dmax.item<float>());
        meanAbs += dmean.item<float>();
        tok = a8;  // teacher: continue along the Q8 stream
    }
    const int n = D + 1;
    std::printf("\n-- Q%d-head quality gate (prefill %d, %d teacher-forced steps, reference = Q8 head) --\n",
                bits, P, D);
    std::printf("  top-1 agreement: %d/%d (%.2f%%)   |dlogit| mean %.3e  max %.3e\n",
                agree, n, 100.0 * agree / n, meanAbs / D, maxAbs);
    std::printf("  (head weights differ by design — this is the quality/speed trade of --quant-embed %d)\n", bits);
}

// P5 verification + measurement: chunked prefill (config.prefillChunk = N, sliding-layer trim
// active) vs the whole-prompt single forward, same greedy decode after. The dropped sliding keys
// carry exactly-zero softmax weight, so outputs SHOULD match token-for-token; the gate measures
// whether tiled-GEMM reassociation (different seqK -> different reduction tree over the kept
// terms) breaks that in practice — the P3 lesson says test, don't assume. Use prefill > window
// + 2 chunks so trims actually fire.
static void chunkVerify(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D, int chunkN) {
    es::ESModelConfig cA = base; cA.fused = true; cA.prefillChunk = 0;
    es::ESModelConfig cB = base; cB.fused = true; cB.prefillChunk = chunkN;
    es::ESGemma4TextForCausalLM lmA(cA, weights), lmB(cB, weights);  // share weight arrays

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;
    const int L = base.numHiddenLayers;

    auto run = [&](const es::ESGemma4TextForCausalLM & lm, std::vector<int> * out, double * preS) -> double {
        es::ESKVCache cache(L);
        auto tp = std::chrono::high_resolution_clock::now();
        mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        if (preS) *preS = secsSince(tp);
        int pos = P, next = es::ESSampler::argmax(ll);
        if (out) out->push_back(next);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int d = 0; d < D; ++d) {
            ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
            pos += 1; next = es::ESSampler::argmax(ll);
            if (out) out->push_back(next);
        }
        return secsSince(t0);
    };

    run(lmA, nullptr, nullptr); run(lmB, nullptr, nullptr);   // warmup (JIT), discarded
    std::vector<int> tA, tB;
    double preA = 0, preB = 0;
    double dtA = run(lmA, &tA, &preA);
    double dtB = run(lmB, &tB, &preB);

    size_t n = std::min(tA.size(), tB.size()), match = 0;
    for (size_t i = 0; i < n; ++i) if (tA[i] == tB[i]) match++;
    bool ok = (match == n) && (n > 0);
    std::printf("\n-- chunked-prefill verify (prefill %d, chunk %d, window %d, decode %d) --\n",
                P, chunkN, base.slidingWindow, D);
    if (P <= base.slidingWindow + 2 * chunkN)
        std::printf("  NOTE: prefill %d <= window+2*chunk — sliding trims barely fire\n", P);
    std::printf("  token match: %zu/%zu  %s\n", match, n, ok ? "PASS" : "FAIL");
    std::printf("  prefill: whole %6.1f tok/s (%.3fs)   chunked %6.1f tok/s (%.3fs)   speedup %.2fx\n",
                P / preA, preA, P / preB, preB, preA / preB);
    std::printf("  decode : whole %6.1f tok/s (%.3fs)   chunked %6.1f tok/s (%.3fs)\n",
                D / dtA, dtA, D / dtB, dtB);
    std::printf("  NOTE: same-process arms — for iso-thermal absolutes use cold fresh-process\n");
    std::printf("        --bench-eager pairs with/without --prefill-chunk.\n");
}

// Tiled-prefill numerics: tiled vs stock (both fused) in lockstep on a FORCED token stream, so a
// single early flip cannot cascade into an unrelated continuation. The tiled merge reaches the
// same e-semantics as softmax(precise) incrementally, but per-chunk rescales reassociate the
// reduction — expect e-scale drift, same class as --swa-lockstep / --step-lockstep. Judge the
// numbers against the --fused-lockstep control at the same prefill: tiled drift at or below the
// fused-vs-unfused drift the product already ships is acceptance.
static void tiledLockstep(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D, int chunkN) {
    es::ESModelConfig cOff = base; cOff.fused = true; cOff.tiledKChunk = 0;
    es::ESModelConfig cOn  = base; cOn.fused  = true; cOn.tiledKChunk  = chunkN;
    es::ESGemma4TextForCausalLM lmOff(cOff, weights), lmOn(cOn, weights);
    const int L = base.numHiddenLayers;

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;

    es::ESKVCache cacheOff(L), cacheOn(L);
    mx::array llOff = lmOff.lastLogits(toks, &cacheOff, 0); mx::eval(llOff);
    mx::array llOn  = lmOn.lastLogits(toks,  &cacheOn,  0); mx::eval(llOn);

    mx::array dPre = mx::max(mx::abs(mx::subtract(mx::astype(llOff, mx::float32),
                                                  mx::astype(llOn, mx::float32))));
    mx::eval(dPre);
    double preMax = dPre.item<float>();
    bool preFlip = es::ESSampler::argmax(llOff) != es::ESSampler::argmax(llOn);

    double maxAbs = 0, meanAbs = 0;
    int flips = 0, pos = P, next = es::ESSampler::argmax(llOff);
    for (int d = 0; d < D; ++d) {
        llOff = lmOff.lastLogits({next}, &cacheOff, pos);
        llOn  = lmOn.lastLogits({next},  &cacheOn,  pos);
        pos += 1;
        mx::array dl = mx::abs(mx::subtract(mx::astype(llOff, mx::float32),
                                            mx::astype(llOn,  mx::float32)));
        mx::array dmaxA = mx::max(dl), dmeanA = mx::mean(dl);
        int aOff = es::ESSampler::argmax(llOff), aOn = es::ESSampler::argmax(llOn);
        mx::eval(dmaxA, dmeanA);
        maxAbs = std::max(maxAbs, (double) dmaxA.item<float>());
        meanAbs += dmeanA.item<float>();
        if (aOff != aOn) {
            flips++;
            if (flips <= 5)
                std::printf("  step %4d (pos %5d): argmax flip stock=%d tiled=%d  maxD=%.3e\n",
                            d, pos - 1, aOff, aOn, (double) dmaxA.item<float>());
        }
        next = aOff;
    }
    std::printf("\n-- tiled vs stock fused, forced stream (prefill %d, K chunk %d, %d steps) --\n",
                P, chunkN, D);
    std::printf("  prefill-shape: max |dlogit| %.3e  argmax %s\n", preMax, preFlip ? "FLIP" : "match");
    std::printf("  decode-shape (drift carried in the cache; decode math identical): max |dlogit| %.3e  mean %.3e  argmax flips %d/%d (%.2f%%)\n",
                maxAbs, meanAbs / D, flips, D, 100.0 * flips / D);
    std::printf("  control: run --fused-lockstep --prefill %d for the shipping fused-vs-unfused drift.\n", P);
}

// Raw-K numerics: rawKV vs stock (both fused) in lockstep on a FORCED token stream.
// Raw-K recomputes the norms and rope per use instead of caching post-processed K/V —
// deterministic per row, so drift is e-scale (kernel reduction order, per-use rounding).
// Judge against the --fused-lockstep control at the same prefill.
static void rawLockstep(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    es::ESModelConfig cOff = base; cOff.fused = true; cOff.rawKV = false;
    es::ESModelConfig cOn  = base; cOn.fused  = true; cOn.rawKV  = true;
    es::ESGemma4TextForCausalLM lmOff(cOff, weights), lmOn(cOn, weights);
    const int L = base.numHiddenLayers;

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;

    es::ESKVCache cacheOff(L), cacheOn(L);
    mx::array llOff = lmOff.lastLogits(toks, &cacheOff, 0); mx::eval(llOff);
    mx::array llOn  = lmOn.lastLogits(toks,  &cacheOn,  0); mx::eval(llOn);

    mx::array dPre = mx::max(mx::abs(mx::subtract(mx::astype(llOff, mx::float32),
                                                  mx::astype(llOn, mx::float32))));
    mx::eval(dPre);
    double preMax = dPre.item<float>();
    bool preFlip = es::ESSampler::argmax(llOff) != es::ESSampler::argmax(llOn);

    double maxAbs = 0, meanAbs = 0;
    int flips = 0, pos = P, next = es::ESSampler::argmax(llOff);
    for (int d = 0; d < D; ++d) {
        llOff = lmOff.lastLogits({next}, &cacheOff, pos);
        llOn  = lmOn.lastLogits({next},  &cacheOn,  pos);
        pos += 1;
        mx::array dl = mx::abs(mx::subtract(mx::astype(llOff, mx::float32),
                                            mx::astype(llOn,  mx::float32)));
        mx::array dmaxA = mx::max(dl), dmeanA = mx::mean(dl);
        int aOff = es::ESSampler::argmax(llOff), aOn = es::ESSampler::argmax(llOn);
        mx::eval(dmaxA, dmeanA);
        maxAbs = std::max(maxAbs, (double) dmaxA.item<float>());
        meanAbs += dmeanA.item<float>();
        if (aOff != aOn) {
            flips++;
            if (flips <= 5)
                std::printf("  step %4d (pos %5d): argmax flip stock=%d raw=%d  maxD=%.3e\n",
                            d, pos - 1, aOff, aOn, (double) dmaxA.item<float>());
        }
        next = aOff;
    }
    std::printf("\n-- rawKV vs stock fused, forced stream (prefill %d, %d steps) --\n", P, D);
    std::printf("  prefill-shape: max |dlogit| %.3e  argmax %s\n", preMax, preFlip ? "FLIP" : "match");
    std::printf("  decode-shape (fused decode kernel live on global layers): max |dlogit| %.3e  mean %.3e  argmax flips %d/%d (%.2f%%)\n",
                maxAbs, meanAbs / D, flips, D, 100.0 * flips / D);
    std::printf("  control: run --fused-lockstep --prefill %d for the shipping fused-vs-unfused drift.\n", P);
}

// Prefix fingerprint for id-stream snapshots: model + quant/chunk config + FNV-1a64 of
// the first P token ids. Marker restores match on these (see ESKVCache::markPrefix).
static std::string apIdsFingerprint(const std::string & modelDir, const es::ESModelConfig & c,
                                    const std::vector<int> & ids, int P) {
    if (P > (int) ids.size()) return "";
    unsigned long long h = 1469598103934665603ULL;
    for (int t = 0; t < P; ++t) { h ^= (unsigned long long) (unsigned) ids[t]; h *= 1099511628211ULL; }
    char buf[32]; std::snprintf(buf, sizeof buf, "%016llx", h);
    return modelDir + "|ids64=" + std::string(buf) + "|P=" + std::to_string(P) +
           "|qb=" + std::to_string(c.quantBits) + "|chunk=" + std::to_string(c.prefillChunk);
}

static es::ESKVCache::RawMode apRawModeFor(const es::ESModelConfig & c) {
    if (!c.rawKV) return es::ESKVCache::RawMode::composite;
    return c.rawKVQ8 ? es::ESKVCache::RawMode::rawQ8 : es::ESKVCache::RawMode::raw;
}

// Marker gate: one snapshot, four consumers — all must continue BIT-EXACTLY.
//   ref     fresh prime of ids[0..M), greedy decode           (the truth)
//   seg     segmented prime WITH a marker, save, decode       == ref (chunk invariance)
//   full    restoreSnapshot full-fp match, decode             == ref
//   marker  restore with a DIVERGENT tail after the marker,
//           prefill the divergent tail, decode                == fresh prime of the
//                                                                divergent stream
static void markerVerify(const es::ESWeightLoader & weights, es::ESModelConfig base,
                         const std::string & modelDir, int P, int D) {
    es::ESModelConfig c = base; c.fused = true;
    if (P < 6144) P = 8192;
    const int MARK = (P / 2) - ((P / 2) % 512);
    const int M = P - 1;
    es::ESGemma4TextForCausalLM lm(c, weights);
    const int L = c.numHiddenLayers;

    std::vector<int> ids(P);
    for (int i = 0; i < P; ++i) ids[i] = 100 + (i % 29000);
    std::vector<int> ids2 = ids;
    for (int i = MARK; i < P; ++i) ids2[i] = 131 + (i % 17000);   // divergent tail

    auto decodeFrom = [&](es::ESKVCache & cache, const std::vector<int> & v, int pos) {
        mx::array ll = lm.lastLogits(std::vector<int>(v.begin() + pos, v.end()), &cache, pos);
        mx::eval(ll);
        int p = (int) v.size();
        std::vector<int> out;
        int next = es::ESSampler::argmax(ll); out.push_back(next);
        for (int s = 1; s < D; ++s) {
            ll = lm.lastLogits({next}, &cache, p); p += 1;
            next = es::ESSampler::argmax(ll); out.push_back(next);
        }
        return out;
    };
    auto fpOf = [&](const std::vector<int> & v, int p) { return apIdsFingerprint(modelDir, c, v, p); };
    const std::string tmp = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp")
                            + "/apertura_marker_verify.safetensors";

    std::printf("\n-- marker-verify (P %d, marker %d, %d greedy steps, mode %s) --\n",
                P, MARK, D, c.rawKV ? (c.rawKVQ8 ? "rawQ8" : "raw") : "composite");

    es::ESKVCache A(L);
    { mx::array ll = lm.lastLogits(std::vector<int>(ids.begin(), ids.begin() + M), &A, 0); mx::eval(ll); }
    auto sRef = decodeFrom(A, ids, M);

    es::ESKVCache B(L);
    { mx::array ll = lm.lastLogits(std::vector<int>(ids.begin(), ids.begin() + MARK), &B, 0); mx::eval(ll); }
    B.markPrefix(MARK, fpOf(ids, MARK));
    { mx::array ll = lm.lastLogits(std::vector<int>(ids.begin() + MARK, ids.begin() + M), &B, MARK); mx::eval(ll); }
    const bool saved = B.saveSnapshot(tmp, fpOf(ids, M), M);
    auto sSeg = decodeFrom(B, ids, M);

    es::ESKVCache C(L);
    auto pf1 = [&](int p) { return fpOf(ids, p); };
    const int posC = C.restoreSnapshot(tmp, fpOf(ids, M), apRawModeFor(c), pf1);
    std::vector<int> sFull; if (posC == M) sFull = decodeFrom(C, ids, M);

    es::ESKVCache Dm(L);
    auto pf2 = [&](int p) { return fpOf(ids2, p); };
    const int posD = Dm.restoreSnapshot(tmp, fpOf(ids2, M), apRawModeFor(c), pf2);
    std::vector<int> sMark; if (posD == MARK) sMark = decodeFrom(Dm, ids2, MARK);

    // Reference for the marker leg is segmented IDENTICALLY to the consumer (prime to
    // MARK, then the tail as one append): q8's kernel-vs-ops decode drift means token
    // identity is only guaranteed under identical segmentation — the property gated
    // here is marker-restore == fresh-prime-of-the-same-prefix, not chunk invariance
    // (composite covers that in the seg leg; q8's is bounded by --raw-lockstep).
    es::ESKVCache E(L);
    { mx::array ll = lm.lastLogits(std::vector<int>(ids2.begin(), ids2.begin() + MARK), &E, 0); mx::eval(ll); }
    auto sRef2 = decodeFrom(E, ids2, MARK);
    std::remove(tmp.c_str());

    auto match = [](const std::vector<int> & a, const std::vector<int> & b) {
        if (a.size() != b.size()) return -1;
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return (int) i;
        return (int) a.size();
    };
    const bool okSeg  = match(sSeg, sRef)   == D;
    const bool okFull = posC == M    && match(sFull, sRef)  == D;
    const bool okMark = posD == MARK && match(sMark, sRef2) == D;
    std::printf("  save: %s   segmented prime vs fresh: %d/%d %s\n",
                saved ? "ok" : "FAILED", match(sSeg, sRef), D, okSeg ? "PASS" : "FAIL");
    std::printf("  full restore  (pos %d, expect %d): %d/%d %s\n",
                posC, M, sFull.empty() ? 0 : match(sFull, sRef), D, okFull ? "PASS" : "FAIL");
    std::printf("  marker restore (pos %d, expect %d) + divergent tail vs fresh: %d/%d %s\n",
                posD, MARK, sMark.empty() ? 0 : match(sMark, sRef2), D, okMark ? "PASS" : "FAIL");
    std::printf("  marker-verify: %s\n", (okSeg && okFull && okMark) ? "PASS" : "FAIL");
}

// Excision gate (reasoning-trace rewind, Kolja's design): a session turn decodes
// [thought T][answer R] into the cache, then rewinds to the turn mark and re-prefills R
// only. The gated property: the excised cache continues EXACTLY like a cache that never
// saw T. Two legs per mode:
//   deep (P 8192):    sliding layers have evicted — the mark's deep-copied windows install
//   shallow (P 2048): sliding layers still hold the full prefix — pure cursor truncation
// The turn is fed token-by-token (the live decode shape), so mid-turn compactions run and
// the mark's copies are load-bearing. Segmentation caveat as in the marker gate: the ref
// prefills A+R contiguously while excision prefills A then R — bit-exact for composite and
// raw bf16; q8's kernel-vs-ops seam bounds any drift (--raw-lockstep).
static void exciseVerify(const es::ESWeightLoader & weights, es::ESModelConfig base, int D) {
    es::ESModelConfig c = base; c.fused = true;
    es::ESGemma4TextForCausalLM lm(c, weights);
    const int L = c.numHiddenLayers;
    const int TH = 384, AN = 256;

    auto leg = [&](const char * name, int P) {
        std::vector<int> A(P), T(TH), R(AN);
        for (int i = 0; i < P;  ++i) A[i] = 100 + (i % 29000);
        for (int i = 0; i < TH; ++i) T[i] = 131 + (i % 17000);
        for (int i = 0; i < AN; ++i) R[i] = 173 + (i % 23000);

        auto greedyFrom = [&](es::ESKVCache & cache, mx::array ll, int pos) {
            std::vector<int> out;
            int next = es::ESSampler::argmax(ll); out.push_back(next);
            for (int s = 1; s < D; ++s) {
                ll = lm.lastLogits({next}, &cache, pos); pos += 1;
                next = es::ESSampler::argmax(ll); out.push_back(next);
            }
            return out;
        };

        // ref: A + R in one prefill, never sees T.
        es::ESKVCache ref(L);
        std::vector<int> AR = A; AR.insert(AR.end(), R.begin(), R.end());
        mx::array lr = lm.lastLogits(AR, &ref, 0); mx::eval(lr);
        auto sRef = greedyFrom(ref, lr, (int) AR.size());

        // exc: A, mark, decode T+R token-by-token, rewind, re-prefill R.
        es::ESKVCache exc(L);
        { mx::array ll = lm.lastLogits(A, &exc, 0); mx::eval(ll); }
        const bool marked = exc.markRewindPoint(P);
        int pos = P;
        for (int t : T) { mx::array ll = lm.lastLogits({t}, &exc, pos); mx::eval(ll); pos += 1; }
        for (int t : R) { mx::array ll = lm.lastLogits({t}, &exc, pos); mx::eval(ll); pos += 1; }
        const int rp = marked ? exc.rewindToMark() : -1;
        std::vector<int> sExc;
        if (rp == P) {
            mx::array le = lm.lastLogits(R, &exc, P); mx::eval(le);
            sExc = greedyFrom(exc, le, (int) AR.size());
        }
        int m = -1;
        if (sExc.size() == sRef.size()) {
            m = (int) sRef.size();
            for (size_t i = 0; i < sRef.size(); ++i) if (sRef[i] != sExc[i]) { m = (int) i; break; }
        }
        const bool ok = marked && rp == P && m == D;
        std::printf("  %s (P %d): mark %s, rewind pos %d (expect %d), excised vs never-saw-T: %d/%d %s\n",
                    name, P, marked ? "ok" : "FAILED", rp, P, m < 0 ? 0 : m, D, ok ? "PASS" : "FAIL");
        return ok;
    };

    std::printf("\n-- excise-verify (thought %d + answer %d, %d greedy steps, mode %s) --\n",
                TH, AN, D, c.rawKV ? (c.rawKVQ8 ? "rawQ8" : "raw") : "composite");
    const bool okDeep    = leg("deep   ", 8192);
    const bool okShallow = leg("shallow", 2048);
    std::printf("  excise-verify: %s\n", (okDeep && okShallow) ? "PASS" : "FAIL");
}

// Rehydration gate: prime a q8 raw cache, snapshot it, restore the snapshot into a bf16
// raw cache with the mode-aware restore (q8 -> bf16 dequantizes — the "switch to BF16 and
// rehydrate" diagnostic path), then decode the same forced stream through both. The two
// caches hold IDENTICAL logical values by construction, so the drift bound is the
// q8-kernel-vs-bf16-path class, NOT the quantization loss — a clean separation of the two.
// Also asserts the composite mode refuses raw content (-1, re-prime fallback).
static void rehydrateVerify(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D) {
    es::ESModelConfig cQ8 = base; cQ8.fused = true; cQ8.rawKV = true; cQ8.rawKVQ8 = true;
    es::ESModelConfig cRaw = base; cRaw.fused = true; cRaw.rawKV = true; cRaw.rawKVQ8 = false;
    es::ESGemma4TextForCausalLM lmQ8(cQ8, weights), lmRaw(cRaw, weights);  // share weight arrays
    const int L = base.numHiddenLayers;

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;

    es::ESKVCache cacheQ8(L);
    mx::array ll = lmQ8.lastLogits(toks, &cacheQ8, 0); mx::eval(ll);

    const std::string tmp = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp")
                            + "/apertura_rehydrate_verify.safetensors";
    const std::string fp = "rehydrate-verify";
    std::printf("\n-- rehydrate-verify (q8 prime %d -> snapshot -> bf16 restore, %d steps) --\n", P, D);
    if (!cacheQ8.saveSnapshot(tmp, fp, P)) { std::printf("  snapshot save FAILED\n"); return; }

    es::ESKVCache cacheRaw(L);
    int pos = cacheRaw.restoreSnapshot(tmp, fp, es::ESKVCache::RawMode::raw);
    std::printf("  restore as raw (rehydrate): pos %d  %s\n", pos, pos == P ? "ok" : "FAILED");
    es::ESKVCache cacheC(L);
    int posC = cacheC.restoreSnapshot(tmp, fp, es::ESKVCache::RawMode::composite);
    std::printf("  restore as composite: %d  %s (raw content must re-prime)\n",
                posC, posC == -1 ? "ok" : "FAILED");

    // Control leg: the same snapshot back into a q8 cache — byte-identical values, same
    // kernel. Any drift here is a restore/rehome fault, so the bound is exactly zero.
    es::ESKVCache cacheQ2(L);
    int posQ = cacheQ2.restoreSnapshot(tmp, fp, es::ESKVCache::RawMode::rawQ8);
    std::printf("  restore as rawQ8 (identity): pos %d  %s\n", posQ, posQ == P ? "ok" : "FAILED");
    std::remove(tmp.c_str());
    if (pos != P || posQ != P) return;

    double maxAbs = 0, meanAbs = 0, maxId = 0;
    int flips = 0, next = es::ESSampler::argmax(ll);
    int p = P;
    for (int d = 0; d < D; ++d) {
        mx::array a = lmQ8.lastLogits({next}, &cacheQ8, p);
        mx::array b = lmRaw.lastLogits({next}, &cacheRaw, p);
        mx::array c = lmQ8.lastLogits({next}, &cacheQ2, p);
        p += 1;
        mx::array dl = mx::abs(mx::subtract(mx::astype(a, mx::float32), mx::astype(b, mx::float32)));
        mx::array di = mx::max(mx::abs(mx::subtract(mx::astype(a, mx::float32), mx::astype(c, mx::float32))));
        mx::array dmaxA = mx::max(dl), dmeanA = mx::mean(dl);
        int na = es::ESSampler::argmax(a), nb = es::ESSampler::argmax(b);
        mx::eval(dmaxA, dmeanA, di);
        maxAbs = std::max(maxAbs, (double) dmaxA.item<float>());
        maxId  = std::max(maxId, (double) di.item<float>());
        meanAbs += dmeanA.item<float>();
        if (na != nb) flips++;
        next = na;
    }
    std::printf("  identity control (q8 -> q8 restore, same kernel): max |dlogit| %.3e (bound: 0)\n", maxId);
    std::printf("  rehydration (q8 kernel vs bf16 path; bf16 rounding + divergent appends): "
                "max |dlogit| %.3e  mean %.3e  argmax flips %d/%d (%.2f%%)\n",
                maxAbs, meanAbs / D, flips, D, 100.0 * flips / D);
}

// Tiled-prefill verification: op-level tiled attention (config.tiledKChunk = N, online-softmax
// K-chunk merge) vs the shipping fused config, same greedy decode after. The tiled math is
// e-equivalent to the length-exact softmax (global max, f32 normalizer, reached incrementally),
// so tokens SHOULD match; the gate measures whether the per-chunk rescale reassociation breaks
// that in practice — same discipline as --chunk-verify. Use prefill > 2*N so the merge fires.
static void tiledVerify(const es::ESWeightLoader & weights, es::ESModelConfig base, int P, int D, int chunkN) {
    es::ESModelConfig cA = base; cA.fused = true; cA.tiledKChunk = 0;
    es::ESModelConfig cB = base; cB.fused = true; cB.tiledKChunk = chunkN;
    es::ESGemma4TextForCausalLM lmA(cA, weights), lmB(cB, weights);  // share weight arrays

    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;
    const int L = base.numHiddenLayers;

    auto run = [&](const es::ESGemma4TextForCausalLM & lm, std::vector<int> * out, double * preS) -> double {
        es::ESKVCache cache(L);
        auto tp = std::chrono::high_resolution_clock::now();
        mx::array ll = lm.lastLogits(toks, &cache, 0); mx::eval(ll);
        if (preS) *preS = secsSince(tp);
        int pos = P, next = es::ESSampler::argmax(ll);
        if (out) out->push_back(next);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int d = 0; d < D; ++d) {
            ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
            pos += 1; next = es::ESSampler::argmax(ll);
            if (out) out->push_back(next);
        }
        return secsSince(t0);
    };

    run(lmA, nullptr, nullptr); run(lmB, nullptr, nullptr);   // warmup (JIT), discarded
    std::vector<int> tA, tB;
    double preA = 0, preB = 0;
    double dtA = run(lmA, &tA, &preA);
    double dtB = run(lmB, &tB, &preB);

    size_t n = std::min(tA.size(), tB.size()), match = 0;
    for (size_t i = 0; i < n; ++i) if (tA[i] == tB[i]) match++;
    bool ok = (match == n) && (n > 0);
    std::printf("\n-- tiled-prefill verify (prefill %d, K chunk %d, prefill-chunk %d, decode %d) --\n",
                P, chunkN, base.prefillChunk, D);
    if (P <= 2 * chunkN)
        std::printf("  NOTE: prefill %d <= 2*chunk — the online-softmax merge barely fires\n", P);
    std::printf("  token match: %zu/%zu  %s\n", match, n, ok ? "PASS" : "FAIL");
    std::printf("  prefill: fused %6.1f tok/s (%.3fs)   tiled %6.1f tok/s (%.3fs)   speedup %.2fx\n",
                P / preA, preA, P / preB, preB, preA / preB);
    std::printf("  decode : fused %6.1f tok/s (%.3fs)   tiled %6.1f tok/s (%.3fs)\n",
                D / dtA, dtA, D / dtB, dtB);
    std::printf("  NOTE: same-process arms — for iso-thermal absolutes use cold fresh-process\n");
    std::printf("        --bench-eager pairs with/without --tiled-prefill.\n");
}

#import "APInternal.h"
#import "APGenerationOptions.h"
#import "APSelectorTool.h"

// AperturaKit facade gate: the ObjC APSession must produce BYTE-IDENTICAL token streams
// to the reference es::ESSession + chat-template-delta path (the --chat-session pattern)
// for the same persona and turns, greedy. Streamed text must equal the accumulated
// deltas. This is the facade's license to exist: same engine calls, same tokens.
static int facadeVerify(const std::string & modelDir, const std::string & personaFile) {
    NSString * pf = [NSString stringWithContentsOfFile:@(personaFile.c_str())
                                              encoding:NSUTF8StringEncoding error:nil];
    std::string persona = pf ? std::string(pf.UTF8String) : std::string();
    if (persona.empty()) { std::fprintf(stderr, "cannot read persona %s\n", personaFile.c_str()); return 1; }
    const std::vector<std::string> turns = { "In one sentence, who are you?",
                                             "And what do you fear losing?" };
    const int kMaxNew = 220;

    // ---- reference arm: es::ESSession + manual turn deltas (the gated pattern) ----
    std::vector<std::vector<int>> refIds;
    {
        es::ESModelConfig c = es::ESModelConfig::fromConfigJSON(modelDir + "/config.json");
        c.computeDtype = mx::bfloat16; c.fused = true;
        es::ESWeightLoader w(modelDir, c);
        es::ESGemma4TextForCausalLM lm(c, w);
        es::ESTokenizer tok(modelDir + "/tokenizer.json");
        es::ESChatTemplate chat(tok);
        es::ESChatTokens T = chat.tokens();
        auto enc = [&](const char * s) { return tok.encode(s, false); };
        auto push = [](std::vector<int> & a, const std::vector<int> & b) {
            a.insert(a.end(), b.begin(), b.end()); };

        es::ESSession sess(lm);
        sess.prime(chat.build({{"system", persona}}, false, false));
        es::ESSamplingConfig sc; sc.greedy = true; sc.maxNewTokens = kMaxNew; sc.eosTokenId = T.turnClose;
        bool first = true;
        for (const std::string & um : turns) {
            std::vector<int> d;
            if (!first) push(d, enc("\n"));
            d.push_back(T.turnOpen); push(d, enc("user\n")); push(d, tok.encode(um, false));
            d.push_back(T.turnClose); push(d, enc("\n"));
            d.push_back(T.turnOpen); push(d, enc("model\n"));
            d.push_back(T.channelOpen); push(d, enc("thought\n")); d.push_back(T.channelClose);
            first = false;
            refIds.push_back(sess.respond(d, sc));
        }
    }

    // ---- facade arm: the public AperturaKit path (fresh model instance, same weights on disk) ----
    NSError * err = nil;
    APModel * model = [APModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(modelDir.c_str())]
                                        configuration:nil error:&err];
    if (!model) { std::fprintf(stderr, "APModel load failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    APLocalSession * session = [[APLocalSession alloc] initWithModel:model];
    dispatch_queue_t cbq = dispatch_queue_create("facade-verify.cb", DISPATCH_QUEUE_SERIAL);
    session.callbackQueue = cbq;

    __block NSError * primeErr = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [session primeWithMessages:@[ [APMessage systemMessageWithText:@(persona.c_str())] ]
                    completion:^(NSError * e) { primeErr = e; dispatch_semaphore_signal(sem); }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    if (primeErr) { std::fprintf(stderr, "prime failed: %s\n", primeErr.localizedDescription.UTF8String); return 1; }

    APGenerationOptions * opts = [APGenerationOptions deterministicOptions];
    opts.maximumResponseTokens = kMaxNew;

    bool ok = true;
    for (size_t t = 0; t < turns.size(); ++t) {
        NSMutableString * streamed = [NSMutableString string];
        __block APResponse * resp = nil; __block NSError * respErr = nil;
        [session respondToMessage:[APMessage userMessageWithText:@(turns[t].c_str())]
                          options:opts
                     deltaHandler:^(APResponseDelta * d) { [streamed appendString:d.text]; }
                       completion:^(APResponse * r, NSError * e) {
                           resp = r; respErr = e; dispatch_semaphore_signal(sem); }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        if (respErr) { std::fprintf(stderr, "respond failed: %s\n", respErr.localizedDescription.UTF8String); return 1; }

        NSArray<NSNumber *> * got = session.lastResponseTokenIDsForTesting;
        const std::vector<int> & ref = refIds[t];
        bool match = ((size_t) got.count == ref.size());
        for (size_t i = 0; match && i < ref.size(); ++i) match = (got[i].intValue == ref[i]);
        // Streamed deltas must reassemble to the skip-special decode of the id stream.
        es::ESTokenizer tok2(modelDir + "/tokenizer.json");
        std::vector<int> gotIds; for (NSNumber * n in got) gotIds.push_back(n.intValue);
        bool streamOK = (std::string(streamed.UTF8String) == tok2.decode(gotIds, true));
        std::printf("  turn %zu: ids %s (%zu vs %zu tok)   stream-reassembly %s   finish=%ld  %4.1f tok/s\n",
                    t + 1, match ? "MATCH" : "DIVERGE", (size_t) got.count, ref.size(),
                    streamOK ? "OK" : "FAIL", (long) resp.finishReason,
                    resp.stats.decodeTokensPerSecond);
        ok = ok && match && streamOK;
    }
    // ---- reasoning ("thinking") mode: reference = the --think chat-session pattern
    // (system turn carries <|think|>, model turn left OPEN); facade = reasoningEnabled.
    std::vector<std::vector<int>> refThinkIds;
    {
        es::ESModelConfig c = es::ESModelConfig::fromConfigJSON(modelDir + "/config.json");
        c.computeDtype = mx::bfloat16; c.fused = true;
        es::ESWeightLoader w(modelDir, c);
        es::ESGemma4TextForCausalLM lm(c, w);
        es::ESTokenizer tok(modelDir + "/tokenizer.json");
        es::ESChatTemplate chat(tok);
        es::ESChatTokens T = chat.tokens();
        auto enc = [&](const char * s) { return tok.encode(s, false); };
        auto push = [](std::vector<int> & a, const std::vector<int> & b) {
            a.insert(a.end(), b.begin(), b.end()); };
        es::ESSession sess(lm);
        sess.prime(chat.build({{"system", persona}}, /*think=*/true, false));
        // ONE turn with a generous budget: reasoning turns are long, and after a
        // max-tokens truncation the facade's deliberate turn-repair (injected <turn|>)
        // diverges from this raw reference — so multi-turn identity is only meaningful
        // for completed turns (the non-reasoning arms already gate multi-turn).
        es::ESSamplingConfig sc; sc.greedy = true; sc.maxNewTokens = 1024; sc.eosTokenId = T.turnClose;
        {
            std::vector<int> d;
            d.push_back(T.turnOpen); push(d, enc("user\n")); push(d, tok.encode(turns[0], false));
            d.push_back(T.turnClose); push(d, enc("\n"));
            d.push_back(T.turnOpen); push(d, enc("model\n"));   // OPEN: model writes its thought channel
            refThinkIds.push_back(sess.respond(d, sc));
        }
    }
    APLocalSession * thinkSession = [[APLocalSession alloc] initWithModel:model];
    thinkSession.callbackQueue = cbq;
    thinkSession.reasoningEnabled = YES;
    __block NSError * tpErr = nil;
    [thinkSession primeWithMessages:@[ [APMessage systemMessageWithText:@(persona.c_str())] ]
                         completion:^(NSError * e) { tpErr = e; dispatch_semaphore_signal(sem); }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    if (tpErr) { std::fprintf(stderr, "think prime failed: %s\n", tpErr.localizedDescription.UTF8String); return 1; }
    APGenerationOptions * thinkOpts = [APGenerationOptions deterministicOptions];
    thinkOpts.maximumResponseTokens = 1024;
    for (size_t t = 0; t < 1; ++t) {
        NSMutableString * thoughtStream = [NSMutableString string];
        NSMutableString * answerStream = [NSMutableString string];
        __block APResponse * resp = nil; __block NSError * respErr = nil;
        [thinkSession respondToMessage:[APMessage userMessageWithText:@(turns[t].c_str())]
                               options:thinkOpts
                          deltaHandler:^(APResponseDelta * d) {
                              [d.isThought ? thoughtStream : answerStream appendString:d.text];
                          }
                            completion:^(APResponse * r, NSError * e) {
                                resp = r; respErr = e; dispatch_semaphore_signal(sem); }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        if (respErr) { std::fprintf(stderr, "think respond failed: %s\n", respErr.localizedDescription.UTF8String); return 1; }
        NSArray<NSNumber *> * got = thinkSession.lastResponseTokenIDsForTesting;
        const std::vector<int> & ref = refThinkIds[t];
        bool match = ((size_t) got.count == ref.size());
        for (size_t i = 0; match && i < ref.size(); ++i) match = (got[i].intValue == ref[i]);
        // Truncated reasoning (max-tokens before the channel closed) legitimately has no
        // answer text; the identity check still holds token-for-token.
        bool truncated = (resp.finishReason == APFinishReasonMaxTokens);
        bool channels = thoughtStream.length > 0 && (truncated || (resp.reasoning.length > 0 &&
                                                                   answerStream.length > 0));
        NSString * trimmedAnswer = [answerStream stringByTrimmingCharactersInSet:
                                    NSCharacterSet.whitespaceAndNewlineCharacterSet];
        NSString * trimmedMessage = [resp.message.textRepresentation stringByTrimmingCharactersInSet:
                                     NSCharacterSet.whitespaceAndNewlineCharacterSet];
        bool answerConsistent = [trimmedAnswer isEqualToString:trimmedMessage];
        std::printf("  think turn %zu: ids %s (%zu vs %zu tok)   thought %lu ch / answer %lu ch %s   answer-consistency %s\n",
                    t + 1, match ? "MATCH" : "DIVERGE", (size_t) got.count, ref.size(),
                    (unsigned long) thoughtStream.length, (unsigned long) answerStream.length,
                    channels ? "OK" : "MISSING", answerConsistent ? "OK" : "FAIL");
        ok = ok && match && channels && answerConsistent;
    }

    std::printf("\n-- facade-verify (AperturaKit APSession vs es::ESSession reference, incl. reasoning mode) --\n");
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// KV-snapshot persistence gate: a session primed by RESTORING a saved snapshot must
// generate byte-identically to a session that prefilled fresh; a changed persona must
// invalidate the snapshot (fingerprint mismatch -> normal prime). Also reports the
// restore-vs-prefill speedup — the whole point of the feature.
static int persistVerify(const std::string & modelDir, const std::string & personaFile) {
    NSString * pf = [NSString stringWithContentsOfFile:@(personaFile.c_str())
                                              encoding:NSUTF8StringEncoding error:nil];
    if (pf.length == 0) { std::fprintf(stderr, "cannot read persona %s\n", personaFile.c_str()); return 1; }
    NSString * cachePath = [NSTemporaryDirectory() stringByAppendingPathComponent:
                            @"apertura-persist-verify.safetensors"];
    [NSFileManager.defaultManager removeItemAtPath:cachePath error:nil];
    NSURL * cacheURL = [NSURL fileURLWithPath:cachePath];

    NSError * err = nil;
    APModel * model = [APModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(modelDir.c_str())]
                                        configuration:nil error:&err];
    if (!model) { std::fprintf(stderr, "APModel load failed: %s\n", err.localizedDescription.UTF8String); return 1; }

    dispatch_queue_t cbq = dispatch_queue_create("persist-verify.cb", DISPATCH_QUEUE_SERIAL);
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    const NSArray<NSString *> * turns = @[ @"In one sentence, who are you?",
                                           @"And what do you fear losing?" ];

    // prime (optionally via snapshot) + two deterministic turns -> token id streams
    NSArray<NSArray<NSNumber *> *> * (^run)(NSURL *, double *, BOOL *) =
    ^(NSURL * url, double * primeSeconds, BOOL * restoredOut) {
        APLocalSession * s = [[APLocalSession alloc] initWithModel:model];
        s.callbackQueue = cbq;
        NSDate * t0 = [NSDate date];
        __block NSError * pe = nil;
        [s primeWithMessages:@[ [APMessage systemMessageWithText:pf] ] cacheURL:url
                  completion:^(NSError * e) { pe = e; dispatch_semaphore_signal(sem); }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        if (primeSeconds) *primeSeconds = -[t0 timeIntervalSinceNow];
        if (restoredOut) *restoredOut = s.lastPrimeRestoredFromSnapshot;
        if (pe) return (NSArray<NSArray<NSNumber *> *> *) nil;
        APGenerationOptions * opts = [APGenerationOptions deterministicOptions];
        opts.maximumResponseTokens = 64;
        NSMutableArray * streams = [NSMutableArray array];
        for (NSString * turn in turns) {
            __block NSError * re = nil;
            [s respondToMessage:[APMessage userMessageWithText:turn] options:opts
                   deltaHandler:nil
                     completion:^(APResponse * r, NSError * e) { re = e; dispatch_semaphore_signal(sem); }];
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            if (re) return (NSArray<NSArray<NSNumber *> *> *) nil;
            [streams addObject:s.lastResponseTokenIDsForTesting];
        }
        return (NSArray<NSArray<NSNumber *> *> *) streams;
    };

    double tPrime = 0, tRestore = 0, tMismatch = 0;
    BOOL r1 = NO, r2 = NO, r3 = NO;
    NSArray * fresh = run(cacheURL, &tPrime, &r1);      // primes + writes the snapshot
    NSArray * restored = run(cacheURL, &tRestore, &r2); // must restore
    NSString * tampered = [@"TAMPERED. " stringByAppendingString:pf];
    APLocalSession * s3 = [[APLocalSession alloc] initWithModel:model];
    s3.callbackQueue = cbq;
    NSDate * t3 = [NSDate date];
    __block NSError * e3 = nil;
    [s3 primeWithMessages:@[ [APMessage systemMessageWithText:tampered] ] cacheURL:cacheURL
               completion:^(NSError * e) { e3 = e; dispatch_semaphore_signal(sem); }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    tMismatch = -[t3 timeIntervalSinceNow];
    r3 = s3.lastPrimeRestoredFromSnapshot;

    bool ok = fresh && restored && !r1 && r2 && !e3 && !r3;
    if (fresh && restored) {
        for (NSUInteger t = 0; t < fresh.count; ++t) {
            NSArray<NSNumber *> * a = fresh[t], * b = restored[t];
            NSUInteger firstDiff = NSNotFound;
            for (NSUInteger i = 0; i < MIN(a.count, b.count); ++i)
                if (![a[i] isEqual:b[i]]) { firstDiff = i; break; }
            std::printf("  turn %lu: fresh %lu tok vs restored %lu tok, first diff at %ld\n",
                        (unsigned long) t + 1, (unsigned long) a.count, (unsigned long) b.count,
                        firstDiff == NSNotFound ? -1L : (long) firstDiff);
            ok = ok && [a isEqualToArray:b];
        }
    }
    std::printf("\n-- persist-verify (KV snapshot restore vs fresh prime) --\n");
    std::printf("  fresh prime : %6.1fs (restored=%d)   snapshot restore : %6.1fs (restored=%d)  %.0fx faster\n",
                tPrime, r1, tRestore, r2, tRestore > 0 ? tPrime / tRestore : 0);
    std::printf("  tampered persona re-primed (restored=%d, %.1fs)\n", r3, tMismatch);
    std::printf("  byte-identity of both turns: %s\n", ok ? "PASS" : "FAIL");
    [NSFileManager.defaultManager removeItemAtPath:cachePath error:nil];
    return ok ? 0 : 1;
}

// Tool-dispatch gate helpers: a deterministic tool handler + a delegate that records
// invocation callbacks. Wired through APSelectorTool so the adapter is gated too.
@interface APToolGateHandler : NSObject
@property (nonatomic) int invocations;
@end
@implementation APToolGateHandler
- (void)handleArgs:(NSDictionary<NSString *, id> *)args
        completion:(void (^)(APContent *, NSError *))completion {
    self.invocations += 1;
    completion([APContent textContent:@"AZURE-HERON-42"], nil);
}
@end

@interface APToolGateDelegate : NSObject <APSessionDelegate>
@property (nonatomic) int didInvokeCount;
@property (nonatomic, copy) NSString * lastToolName;
@end
@implementation APToolGateDelegate
- (void)session:(APSession *)session didInvokeTool:(NSString *)toolName
      arguments:(NSDictionary<NSString *, id> *)arguments result:(NSString *)result {
    self.didInvokeCount += 1;
    self.lastToolName = toolName;
}
@end

// Tool-dispatch gate: register a deterministic tool, ask a question that requires it,
// and assert the whole grammar loop ran — declaration advertised at prime, call emitted
// and dispatched, response spliced, and the tool's datum present in the final answer.
static int toolsVerify(const std::string & modelDir) {
    NSError * err = nil;
    APModel * model = [APModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(modelDir.c_str())]
                                        configuration:nil error:&err];
    if (!model) { std::fprintf(stderr, "APModel load failed: %s\n", err.localizedDescription.UTF8String); return 1; }

    APToolGateHandler * handler = [[APToolGateHandler alloc] init];
    APSelectorTool * tool = [APSelectorTool
        toolWithName:@"lookup_codeword"
     toolDescription:@"Returns the current secret codeword. Call this whenever the user asks for the codeword."
     parameterSchema:@{ @"type" : @"object", @"properties" : @{} }
              target:handler action:@selector(handleArgs:completion:)];

    APLocalSession * session = [[APLocalSession alloc] initWithModel:model];
    dispatch_queue_t cbq = dispatch_queue_create("tools-verify.cb", DISPATCH_QUEUE_SERIAL);
    session.callbackQueue = cbq;
    APToolGateDelegate * delegate = [[APToolGateDelegate alloc] init];
    session.delegate = delegate;
    [session registerTool:tool];   // BEFORE prime: advertised in the system turn

    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block NSError * pe = nil;
    [session primeWithMessages:@[ [APMessage systemMessageWithText:
        @"You are a terse assistant. Use your tools when they can answer the question."] ]
                    completion:^(NSError * e) { pe = e; dispatch_semaphore_signal(sem); }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    if (pe) { std::fprintf(stderr, "prime failed: %s\n", pe.localizedDescription.UTF8String); return 1; }

    APGenerationOptions * opts = [APGenerationOptions deterministicOptions];
    opts.maximumResponseTokens = 256;
    NSMutableString * streamed = [NSMutableString string];
    __block APResponse * resp = nil; __block NSError * re = nil;
    [session respondToMessage:[APMessage userMessageWithText:
        @"Use the lookup_codeword tool and tell me the codeword exactly."]
                      options:opts
                 deltaHandler:^(APResponseDelta * d) { [streamed appendString:d.text]; }
                   completion:^(APResponse * r, NSError * e) {
                       resp = r; re = e; dispatch_semaphore_signal(sem); }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    if (re) { std::fprintf(stderr, "respond failed: %s\n", re.localizedDescription.UTF8String); return 1; }

    bool invoked   = handler.invocations >= 1;
    bool named     = [resp.executedToolNames containsObject:@"lookup_codeword"];
    bool delegated = delegate.didInvokeCount >= 1 &&
                     [delegate.lastToolName isEqualToString:@"lookup_codeword"];
    bool inAnswer  = [resp.message.textRepresentation containsString:@"AZURE-HERON-42"];
    bool inStream  = [streamed containsString:@"AZURE-HERON-42"];
    bool noLeak    = ![streamed containsString:@"call:lookup_codeword"];   // machinery suppressed
    bool ok = invoked && named && delegated && inAnswer && inStream && noLeak;

    std::printf("\n-- tools-verify (grammar dispatch: advertise -> call -> splice -> answer) --\n");
    std::printf("  invoked=%d(x%d) named=%d delegate=%d answer-has-datum=%d stream-has-datum=%d machinery-suppressed=%d\n",
                invoked, handler.invocations, named, delegated, inAnswer, inStream, noLeak);
    std::printf("  answer: %s\n", resp.message.textRepresentation.UTF8String);
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Greedy-decode N tokens after `prompt` (KV-cache loop, mirrors benchOne's decode path).
static std::vector<int> greedyDecode(const es::ESGemma4TextForCausalLM & lm,
                                     const std::vector<int> & prompt, int N) {
    es::ESKVCache cache(lm.config().numHiddenLayers);
    mx::array ll = lm.lastLogits(prompt, &cache, 0); mx::eval(ll);
    int pos = (int) prompt.size(), next = es::ESSampler::argmax(ll);
    std::vector<int> out; out.reserve(N);
    for (int d = 0; d < N; ++d) {
        out.push_back(next);
        ll = lm.lastLogits({next}, &cache, pos); mx::eval(ll);
        pos += 1; next = es::ESSampler::argmax(ll);
    }
    return out;
}

static const char * kDefaultModelDir =
    "/Users/apocryphx/.cache/huggingface/hub/models--google--gemma-4-31b-it/"
    "snapshots/3548789868c5356dbf307c98e6f609007b82b3eb";

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
            std::printf(
                "AperturaResearch <modelDir|bundle.apml> [fixtures.safetensors] [mode] [flags]\n"
                "Default (no mode): conformance vs the PyTorch fixtures.\n"
                "\n"
                "GENERATION\n"
                "  --chat <prompt> [--system <s>]   one chat turn (chat template, greedy)\n"
                "  --generate <prompt>              raw completion\n"
                "  --chat-session <persona.md>      persistent session: persona primed once, scripted turns\n"
                "  --chat-file <f>                  --chat with the user turn read from a file\n"
                "  --sample [--top-k N] [--seed N]  sampled decoding (temp 1.0, top-p 0.95; det. LCG)\n"
                "  --kv-snapshot <f> [--kv-markers p1,p2]   (chat modes) restore-or-prime the prompt\n"
                "                     cache; markers are prefix waypoints — later runs auto-restore\n"
                "                     the longest marked prefix matching their prompt and prefill\n"
                "                     only the tail (gate: --marker-verify)\n"
                "  --prompt-file <f>                prompt from file   --chat-ids <f> pre-tokenized ids\n"
                "  --tools <f> / --tool-result <f>  tool-use turns\n"
                "\n"
                "BENCH (cold-gate arms with Tools/hidtemp; one arm per process — roadmap §6)\n"
                "  --bench            unfused THEN fused benchOne (fused row is pool-polluted; A/B only)\n"
                "  --bench-eager      clean fused single arm: 1 prefill + 32 warm + D decode\n"
                "  --kv-snapshot <f>  (--bench-eager) restore the post-prefill cache from <f>, or\n"
                "                     prefill once and save it — decode-only benching at depth\n"
                "                     without the 10-min prefill or its thermal pollution\n"
                "  --bench-step       same shape, whole-step-compiled decode (P3 prototype)\n"
                "  --bench-async      sync vs async vs async+compiled-tail decode\n"
                "  --prefill N / --decode N          workload sizes (default 512 / 128)\n"
                "\n"
                "VERIFY GATES (greedy-token match unless noted)\n"
                "  --swa-verify       P1 sliding-window eviction off vs on (token identity)\n"
                "  --swa-lockstep     P1 numerics: forced-stream |dlogit| + flip rate\n"
                "  --fused-lockstep   fused vs unfused attention at DECODE shapes, where the\n"
                "                     sdpa_vector kernel actually engages (prefill falls back)\n"
                "  --cache-verify     P0 legacy concat vs prealloc slice_update cache (bit-exact,\n"
                "                     includes a mid-decode multi-token turn append)\n"
                "  --chunk-verify     P5 chunked vs whole-prompt prefill (bit-exact)\n"
                "  --tiled-verify     op-level tiled prefill attention vs fused (greedy-token match)\n"
                "  --tiled-lockstep   tiled numerics: forced-stream |dlogit| + flip rate\n"
                "  --raw-lockstep     raw-K numerics: forced-stream |dlogit| + flip rate\n"
                "  --rehydrate-verify q8 cache -> snapshot -> bf16 restore (dequant rehydration):\n"
                "                     forced-stream drift of q8 kernel vs rehydrated bf16 path\n"
                "  --excise-verify    reasoning excision: mark, decode thought+answer, rewind,\n"
                "                     re-prefill answer — must equal a cache that never saw the thought\n"
                "  --marker-verify    prefix-marker snapshots: segmented prime, full restore, and\n"
                "                     marker restore + divergent tail — all bit-exact vs fresh\n"
                "  --step-verify      P3 compiled step vs eager (e-equivalent; ~0.5%% shallow flips)\n"
                "  --step-lockstep    P3 numerics: forced-stream |dlogit| + flip rate\n"
                "  --head-verify      Q4/Q6 head vs Q8: teacher-forced top-1 agreement (quality trade)\n"
                "  --session-verify <persona.md>    ESSession byte-identity + per-turn speedup\n"
                "  --verify-bundle <o.apml>         bundle reload == in-memory quant\n"
                "  --vs-bf16 <o.apml>               quantized vs full-precision top-1\n"
                "  --longctx <fixture>              PyTorch long-context oracle (argmax + greedy)\n"
                "\n"
                "CONFIG\n"
                "  --cpu                      run through MLX's CPU backend instead of Metal\n"
                "                             (different kernels; isolates Metal-specific faults)\n"
                "  --dump-mlx <out.safetensors>  write our own tensors keyed by fixture name, to\n"
                "                             localise a divergence offline (which layer, which element)\n"
                "  --ulp                      per-op gate in bf16 ULP units (scale-free) instead of\n"
                "                             absolute tolerances; gates on distribution not max\n"
                "  --probe-all                run op-level probes on every layer, not just 0 and 5\n"
                "  --fused                    mx::fast SDPA path (benches force it on where noted)\n"
                "  --quant N / --quant-group N / --quant-kv N     runtime quantization (HF dir loads)\n"
                "  --quant-kv-all             quant-KV on ALL layers (default: global layers only,\n"
                "                             sliding layers keep bf16 + fused vector kernel)\n"
                "  --quant-embed N            head bits; on a bundle re-quantizes the packed head\n"
                "                             (Q4: +3.3-3.6%% decode at 99.4%% top-1 — roadmap P4)\n"
                "  --prefill-chunk N          chunked prefill (default 512; 0 = whole-prompt) — P5\n"
                "  --tiled-prefill N          op-level tiled prefill attention, K chunk N (0 = off)\n"
                "  --raw-kv                   raw-K global cache + fused decode kernel (half depth bytes)\n"
                "  --raw-q8                   ... with per-row q8 packing (quarter depth bytes; lossy,\n"
                "                             in-kernel register dequant — implies --raw-kv)\n"
                "  --no-swa-cache / --no-prealloc-cache           A/B off-switches (P1 / P0)\n"
                "  --moe-sparse               sparse expert path (26B)\n"
                "  --export <out.apml>        write a quantized bundle\n"
                "\n"
                "Measured standing + methodology: aptransformer/PERFORMANCE_ROADMAP.md\n");
            return 0;
        }
        // Positional args (skip --flags and their values like the --generate operand).
        std::vector<std::string> pos;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--generate" || a == "--quant" || a == "--decode") { i++; continue; }
            if (a == "--quant-embed") { if (i + 1 < argc && std::atoi(argv[i + 1]) > 0) i++; continue; }
            if (a == "--prefill-chunk") { i++; continue; }
            if (a == "--tiled-prefill") { i++; continue; }
            if (a == "--kv-snapshot") { i++; continue; }
            if (a == "--facade-verify") { i++; continue; }
            if (a == "--persist-verify") { i++; continue; }
            if (a == "--longctx" || a == "--quant-kv" || a == "--prefill" || a == "--chat-ids"
                || a == "--expert-ladder" || a == "--chat" || a == "--chat-file" || a == "--system"
                || a == "--top-k" || a == "--seed" || a == "--kv-markers"
                || a == "--export" || a == "--quant-group" || a == "--verify-bundle"
                || a == "--vs-bf16" || a == "--prompt-file" || a == "--session-verify"
                || a == "--chat-session" || a == "--tools" || a == "--tool-result"
                || a == "--dump-mlx") { i++; continue; }
            if (a.rfind("--", 0) == 0) continue;
            pos.push_back(a);
        }
        std::string modelDir = pos.size() > 0 ? pos[0] : kDefaultModelDir;
        std::string fixturesPath;
        if (pos.size() > 1) {
            fixturesPath = pos[1];
        } else {
            // Fixtures live in this repo; derive the path from __FILE__ so it
            // follows the checkout instead of hardcoding a location.
            std::string self = __FILE__;
            fixturesPath = self.substr(0, self.rfind("/AperturaResearch/"))
                + "/aptransformerTests/Fixtures/fixtures.safetensors";
        }

        // --cpu runs the whole graph through MLX's CPU backend instead of Metal.
        // Different kernels entirely, so a Metal-specific numerical fault shows up
        // as a CPU/GPU disagreement on identical MLX code. Slow; use short seqs.
        const bool useCPU = hasFlag(argc, argv, "--cpu");
        mx::set_default_device(useCPU ? mx::Device::cpu : mx::Device::gpu);

        std::printf("== Apertura conformance ==\n");
        std::printf("modelDir : %s\n", modelDir.c_str());
        std::printf("fixtures : %s\n", fixturesPath.c_str());
        std::printf("device   : %s\n", useCPU ? "CPU (MLX cpu backend)" : "GPU (Metal)");

        es::ESModelConfig config = es::ESModelConfig::fromConfigJSON(modelDir + "/config.json");
        config.computeDtype = mx::bfloat16;
        bool useFused = hasFlag(argc, argv, "--fused");
        bool bench    = hasFlag(argc, argv, "--bench");
        bool benchAsync = hasFlag(argc, argv, "--bench-async");
        bool swaVerifyFlag = hasFlag(argc, argv, "--swa-verify");
        bool cacheVerifyFlag = hasFlag(argc, argv, "--cache-verify");
        bool stepVerifyFlag = hasFlag(argc, argv, "--step-verify");
        bool benchStepFlag = hasFlag(argc, argv, "--bench-step");
        bool benchEagerFlag = hasFlag(argc, argv, "--bench-eager");
        bool stepLockstepFlag = hasFlag(argc, argv, "--step-lockstep");
        bool swaLockstepFlag = hasFlag(argc, argv, "--swa-lockstep");
        bool fusedLockstepFlag = hasFlag(argc, argv, "--fused-lockstep");
        bool ulpFlag      = hasFlag(argc, argv, "--ulp");        // ULP-relative per-op gate
        bool probeAllFlag = hasFlag(argc, argv, "--probe-all");  // op probes on every layer
        bool headVerifyFlag = hasFlag(argc, argv, "--head-verify");
        bool chunkVerifyFlag = hasFlag(argc, argv, "--chunk-verify");
        bool tiledVerifyFlag = hasFlag(argc, argv, "--tiled-verify");
        bool tiledLockstepFlag = hasFlag(argc, argv, "--tiled-lockstep");
        bool rawLockstepFlag = hasFlag(argc, argv, "--raw-lockstep");
        bool rehydrateVerifyFlag = hasFlag(argc, argv, "--rehydrate-verify");
        config.fused  = useFused;
        if (hasFlag(argc, argv, "--no-swa-cache")) config.slidingWindowCache = false;  // A/B off-switch
        if (hasFlag(argc, argv, "--no-prealloc-cache")) config.preallocKVCache = false;  // A/B off-switch
        for (int i = 1; i < argc - 1; ++i)  // --prefill-chunk N: P5 chunked prefill (0 = off)
            if (std::strcmp(argv[i], "--prefill-chunk") == 0) config.prefillChunk = std::atoi(argv[i + 1]);
        for (int i = 1; i < argc - 1; ++i)  // --tiled-prefill N: op-level tiled prefill attention, K chunk N (0 = off)
            if (std::strcmp(argv[i], "--tiled-prefill") == 0) config.tiledKChunk = std::atoi(argv[i + 1]);
        for (int i = 1; i < argc - 1; ++i)
            if (std::strcmp(argv[i], "--quant") == 0) config.quantBits = std::atoi(argv[i + 1]);
        // --quant-embed [N]: quantize embed/lm_head at N bits (default 8 — the precision-sensitive
        // output projection). Independent of the layer bits, enabling layers Q4 + embed Q8.
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--quant-embed") == 0) {
                config.quantEmbedBits = 8;  // default
                if (i + 1 < argc && std::atoi(argv[i + 1]) > 0) config.quantEmbedBits = std::atoi(argv[i + 1]);
            }
        }
        for (int i = 1; i < argc - 1; ++i)
            if (std::strcmp(argv[i], "--quant-kv") == 0) config.quantKVBits = std::atoi(argv[i + 1]);
        // --quant-kv-all: quantize every layer's KV (pre-hybrid behavior; default is global-only)
        if (hasFlag(argc, argv, "--quant-kv-all")) config.quantKVGlobalOnly = false;
        // --raw-kv: raw-K global-layer cache + fused decode kernel (half the depth bytes)
        if (hasFlag(argc, argv, "--raw-kv")) config.rawKV = true;
        if (hasFlag(argc, argv, "--raw-q8")) { config.rawKV = true; config.rawKVQ8 = true; }
        config.moeSparse = hasFlag(argc, argv, "--moe-sparse");
        for (int i = 1; i < argc - 1; ++i)
            if (std::strcmp(argv[i], "--quant-group") == 0) config.quantGroupSize = std::atoi(argv[i + 1]);

        // ---- --export <out.apml>: quantize this model dir into an .apml bundle, then exit ----
        // Uses --quant / --quant-embed / --quant-group for the recipe (e.g. layers Q4 + embed Q8).
        // exportQuantizedBundle loads bf16 itself, so this runs without the heavy model build below.
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--export") == 0 && i + 1 < argc) {
                std::string outPath = argv[i + 1];
                es::ESBundleExportOptions opts;
                opts.bits          = config.quantBits;        // --quant
                opts.embedBits     = config.quantEmbedBits;   // --quant-embed
                opts.groupSize     = config.quantGroupSize;   // --quant-group
                opts.sourceModelId = modelDir;
                std::printf("== export .apml bundle ==\n  from : %s\n  to   : %s\n"
                            "  bits=%d embed_bits=%d group=%d\n",
                            modelDir.c_str(), outPath.c_str(), opts.bits, opts.embedBits, opts.groupSize);
                std::string err;
                bool ok = es::exportQuantizedBundle(modelDir, outPath, opts, &err);
                if (ok) std::printf("  OK: wrote %s\n", outPath.c_str());
                else    std::printf("  FAILED: %s\n", err.c_str());
                return ok ? 0 : 1;
            }
        }

        // ---- --verify-bundle <out.apml>: full-scale round-trip gate ----
        // Loads the .apml (reload path) AND the source modelDir quantized in memory with the SAME
        // recipe (read from the bundle), runs one forward over a fixed probe, and asserts the
        // per-position argmax is identical. Heavy (both models resident); the small-scale invariant
        // is already proven by ESPrimitivesTests/testReloadMatchesInMemoryQuant.
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--verify-bundle") == 0 && i + 1 < argc) {
                std::string apml = argv[i + 1];
                std::printf("== verify-bundle ==\n  bundle : %s\n  source : %s\n", apml.c_str(), modelDir.c_str());
                bool moeSparse = hasFlag(argc, argv, "--moe-sparse");

                // Reload path (from the .apml — pre-quantized).
                es::ESModelConfig cfgB = es::ESModelConfig::fromConfigJSON(apml + "/config.json");
                cfgB.computeDtype = mx::bfloat16; cfgB.moeSparse = moeSparse;
                es::ESWeightLoader wB(apml, cfgB);
                es::ESGemma4TextForCausalLM lmReload(cfgB, wB);

                // In-memory-quant path (from the source dir), matching the bundle's recipe.
                es::ESModelConfig cfgM = es::ESModelConfig::fromConfigJSON(modelDir + "/config.json");
                cfgM.computeDtype   = mx::bfloat16; cfgM.moeSparse = moeSparse;
                cfgM.quantBits      = wB.bundleBits();
                cfgM.quantEmbedBits = wB.bundleEmbedBits();
                cfgM.quantGroupSize = wB.bundleGroupSize();
                es::ESWeightLoader wM(modelDir, cfgM);
                es::ESGemma4TextForCausalLM lmMem(cfgM, wM);

                std::vector<int> probe = {2, 1, 17, 235, 4096, 100, 9999, 3};  // BOS + arbitrary ids < vocab
                mx::array lr = lmReload.forward(probe, nullptr, 0);  // [seq, vocab]
                mx::array lm = lmMem.forward(probe, nullptr, 0);
                mx::array aR = mx::argmax(lr, -1), aM = mx::argmax(lm, -1);
                mx::array same = mx::all(mx::equal(aR, aM));
                mx::array dmax = mx::max(mx::abs(mx::subtract(mx::astype(lr, mx::float32), mx::astype(lm, mx::float32))));
                mx::eval(same); mx::eval(dmax);
                bool ok = same.item<bool>();
                std::printf("  positions=%zu  argmax-identical=%s  max|Δlogit|=%.3e  %s\n",
                            probe.size(), ok ? "yes" : "NO", dmax.item<float>(), ok ? "PASS" : "FAIL");
                return ok ? 0 : 1;
            }
        }

        // ---- --vs-bf16 <out.apml>: quantization QUALITY vs full precision ----
        // For each probe prompt, greedy-decode the BF16 reference continuation, then teacher-force
        // it through the quantized bundle and count how often Q4's next-token argmax matches BF16's
        // (given identical context). This measures how far Q4 sits from full precision -- distinct
        // from --verify-bundle, which only checks the engine is faithful to its own recipe.
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--vs-bf16") == 0 && i + 1 < argc) {
                std::string apml = argv[i + 1];
                bool moeSparse = hasFlag(argc, argv, "--moe-sparse");
                es::ESTokenizer tok(modelDir + "/tokenizer.json");
                es::ESChatTemplate chat(tok);  // -it model: measure on the chat distribution it expects

                es::ESModelConfig cBF = es::ESModelConfig::fromConfigJSON(modelDir + "/config.json");
                cBF.computeDtype = mx::bfloat16; cBF.moeSparse = moeSparse;  // quantBits 0 -> full precision
                es::ESWeightLoader wBF(modelDir, cBF);
                es::ESGemma4TextForCausalLM lmBF(cBF, wBF);

                es::ESModelConfig cQ = es::ESModelConfig::fromConfigJSON(apml + "/config.json");
                cQ.computeDtype = mx::bfloat16; cQ.moeSparse = moeSparse;
                es::ESWeightLoader wQ(apml, cQ);
                es::ESGemma4TextForCausalLM lmQ(cQ, wQ);

                // Prompt set: a single long real prompt from --prompt-file (chat-templated via the
                // model's own template, like Isolde uses), else built-in probes. The long prompt
                // yields a long answer -> many answer tokens -> a statistically firm number.
                std::string promptFile;
                for (int j = 1; j < argc; ++j)
                    if (std::strcmp(argv[j], "--prompt-file") == 0 && j + 1 < argc) promptFile = argv[j + 1];

                std::vector<std::pair<std::string, std::vector<int>>> cases;
                int N;
                if (!promptFile.empty()) {
                    NSString * fc = [NSString stringWithContentsOfFile:@(promptFile.c_str())
                                                              encoding:NSUTF8StringEncoding error:nil];
                    std::string content = fc ? std::string(fc.UTF8String) : std::string();
                    std::vector<es::ESChatMessage> msgs = {{"user", content}};
                    cases.emplace_back("prompt-file", chat.build(msgs, false, true));
                    N = 256;
                    for (int j = 1; j < argc - 1; ++j)
                        if (std::strcmp(argv[j], "--decode") == 0) N = std::atoi(argv[j + 1]);
                } else {
                    for (const char * p : {"The capital of France is",
                                           "In one sentence, explain why the sky is blue.",
                                           "List three primary colors:"}) {
                        std::vector<es::ESChatMessage> msgs = {{"user", p}};
                        cases.emplace_back(p, chat.build(msgs, false, true));
                    }
                    N = 48;
                }

                int total = 0, agree = 0;
                std::printf("== vs-bf16 (Q4 quality vs full precision) ==\n  bundle : %s\n", apml.c_str());
                for (auto & c : cases) {
                    const std::vector<int> & ids = c.second;
                    std::vector<int> ref = greedyDecode(lmBF, ids, N);             // BF16 reference answer
                    std::vector<int> full = ids; full.insert(full.end(), ref.begin(), ref.end());
                    mx::array aQ = mx::argmax(lmQ.forward(full, nullptr, 0), -1);  // Q4 teacher-forced [seq]
                    mx::eval(aQ);
                    const uint32_t * a = aQ.data<uint32_t>();
                    // Count only the answer region (through the first end_of_turn=106); past it both
                    // models emit degenerate junk that says nothing about quantization quality.
                    int M = N; for (int k = 0; k < N; ++k) if (ref[k] == 106) { M = k + 1; break; }
                    int base = (int) ids.size() - 1, m = 0;
                    for (int k = 0; k < M; ++k) if ((int) a[base + k] == ref[k]) m++;
                    total += M; agree += m;
                    std::vector<int> ans(ref.begin(), ref.begin() + M);
                    std::string txt = tok.decode(ans, true);
                    if (txt.size() > 220) txt = txt.substr(0, 220) + " …";
                    std::printf("  [%3d/%-3d top-1] %s (%zu prompt tok)\n      bf16 -> %s\n",
                                m, M, c.first.c_str(), ids.size(), txt.c_str());
                }
                std::printf("  AGGREGATE Q4-vs-BF16 top-1 agreement: %d/%d = %.1f%%\n",
                            agree, total, 100.0 * agree / total);
                return 0;
            }
        }

        // ---- --session-verify <persona.md>: prefix-cache correctness + per-turn speedup ----
        // Proves ESSession (prime persona once, respond per turn) is byte-identical to a from-scratch
        // forward over the concatenated tokens, and times prime-once vs re-prefill-every-turn.
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--session-verify") == 0 && i + 1 < argc) {
                std::string personaFile = argv[i + 1];
                es::ESModelConfig c = es::ESModelConfig::fromConfigJSON(modelDir + "/config.json");
                c.computeDtype = mx::bfloat16; c.fused = true;            // flash path
                c.moeSparse = hasFlag(argc, argv, "--moe-sparse");
                es::ESWeightLoader w(modelDir, c);
                es::ESGemma4TextForCausalLM lm(c, w);
                es::ESTokenizer tok(modelDir + "/tokenizer.json");

                NSString * pf = [NSString stringWithContentsOfFile:@(personaFile.c_str())
                                                          encoding:NSUTF8StringEncoding error:nil];
                std::string persona = pf ? std::string(pf.UTF8String) : std::string();
                std::vector<int> A = tok.encode(persona, /*addSpecialTokens=*/true);        // bos + persona (constant prefix)
                std::vector<int> B = tok.encode("\n\nIn one sentence, who are you?", false); // the turn delta
                std::printf("== session-verify ==\n  persona prefix: %zu tok | turn: %zu tok\n", A.size(), B.size());

                es::ESSamplingConfig sc; sc.greedy = true; sc.maxNewTokens = 16; sc.eosTokenId = 106;

                std::vector<int> full = A; full.insert(full.end(), B.begin(), B.end());
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<int> fs = es::ESGenerationLoop(lm, sc).generate(full);   // re-prefills A+B
                double t_fs = secsSince(t0);

                es::ESSession sess(lm);
                t0 = std::chrono::high_resolution_clock::now(); sess.prime(A); double t_prime = secsSince(t0);
                t0 = std::chrono::high_resolution_clock::now();
                std::vector<int> se = sess.respond(B, sc);                           // turn only
                double t_resp = secsSince(t0);

                size_t nmin = std::min(fs.size(), se.size()), match = 0;
                for (size_t k = 0; k < nmin; ++k) { if (fs[k] == se[k]) match++; else break; }
                bool ok = (match == nmin && fs.size() == se.size());
                std::printf("  byte-identity: %zu/%zu greedy tokens match  %s\n", match, nmin, ok ? "PASS" : "FAIL");
                std::printf("  from-scratch turn : %6.2fs  (re-prefills the %zu-tok persona)\n", t_fs, A.size());
                std::printf("  session prime     : %6.2fs  (persona, ONCE per session)\n", t_prime);
                std::printf("  session respond   : %6.2fs  (turn only)\n", t_resp);
                std::printf("  => per extra turn : %6.2fs (session) vs %6.2fs (from-scratch) = %.1fx faster\n",
                            t_resp, t_fs, t_fs / std::max(t_resp, 1e-6));
                std::printf("  reply: %s\n", tok.decode(se, true).c_str());
                return ok ? 0 : 1;
            }
        }

        // ---- --chat-session <persona.md>: the real thing — persona primed once, fast turns ----
        // Primes the persona system block into a persistent ESSession, then runs scripted turns,
        // each appended as a chat-template delta (mirrors ESChatTemplate::build, so no re-tokenizing
        // the reply). This is the integration pattern a harness mirrors: prime() once, respond() per turn.
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--chat-session") == 0 && i + 1 < argc) {
                std::string personaFile = argv[i + 1];
                bool think = hasFlag(argc, argv, "--think");
                es::ESModelConfig c = es::ESModelConfig::fromConfigJSON(modelDir + "/config.json");
                c.computeDtype = mx::bfloat16; c.fused = true; c.moeSparse = hasFlag(argc, argv, "--moe-sparse");
                es::ESWeightLoader w(modelDir, c);
                es::ESGemma4TextForCausalLM lm(c, w);
                es::ESTokenizer tok(modelDir + "/tokenizer.json");
                es::ESChatTemplate chat(tok);
                es::ESChatTokens T = chat.tokens();
                auto enc = [&](const std::string & s) { return tok.encode(s, /*addSpecial=*/false); };
                auto push = [](std::vector<int> & a, const std::vector<int> & b) { a.insert(a.end(), b.begin(), b.end()); };

                NSString * pf = [NSString stringWithContentsOfFile:@(personaFile.c_str())
                                                          encoding:NSUTF8StringEncoding error:nil];
                std::string persona = pf ? std::string(pf.UTF8String) : std::string();

                // Constant prefix: the persona system block. Primed ONCE.
                std::vector<int> prefix = chat.build({{"system", persona}}, think, /*addGen=*/false);
                es::ESSession sess(lm);
                auto t0 = std::chrono::high_resolution_clock::now();
                sess.prime(prefix);
                std::printf("== chat-session ==\n  persona: %zu tok primed in %.1fs (ONCE per session)\n\n",
                            prefix.size(), secsSince(t0));

                es::ESSamplingConfig sc; sc.greedy = true; sc.maxNewTokens = 220; sc.eosTokenId = T.turnClose;

                // Build a turn delta the same way build() lays out a user turn + generation prompt.
                auto turnDelta = [&](const std::string & userMsg, bool first) {
                    std::vector<int> d;
                    if (!first) push(d, enc("\n"));                 // the \n after the previous model <turn|>
                    d.push_back(T.turnOpen); push(d, enc("user\n")); push(d, enc(userMsg));
                    d.push_back(T.turnClose); push(d, enc("\n"));
                    d.push_back(T.turnOpen); push(d, enc("model\n"));
                    if (!think) { d.push_back(T.channelOpen); push(d, enc("thought\n")); d.push_back(T.channelClose); }
                    return d;
                };

                const char * turns[] = { "In one sentence, who are you?", "And what do you fear losing?" };
                bool first = true;
                for (const char * um : turns) {
                    std::vector<int> d = turnDelta(um, first);
                    if (first) {  // prove the manual delta matches build() exactly
                        std::vector<int> ref = chat.build({{"system", persona}, {"user", um}}, think, /*addGen=*/true);
                        std::vector<int> got = prefix; push(got, d);
                        std::printf("  [delta byte-identity vs build(): %s]\n", got == ref ? "PASS" : "FAIL");
                    }
                    first = false;
                    auto t1 = std::chrono::high_resolution_clock::now();
                    std::vector<int> reply = sess.respond(d, sc);
                    double dt = secsSince(t1);
                    es::ESParsedResponse pr = chat.parse(reply);
                    std::printf("user   : %s\n", um);
                    std::printf("Isolde : %s\n  (%zu tok in %.1fs, %.1f tok/s)\n\n",
                                pr.answer.c_str(), reply.size(), dt, reply.size() / std::max(dt, 1e-6));
                }
                return 0;
            }
        }

        int decodeLen = 32, prefillLen = 64;
        for (int i = 1; i < argc - 1; ++i) {
            if (std::strcmp(argv[i], "--decode") == 0)  decodeLen  = std::atoi(argv[i + 1]);
            if (std::strcmp(argv[i], "--prefill") == 0) prefillLen = std::atoi(argv[i + 1]);
        }
        std::string longctxPath, chatIdsPath, ladderPath, chatUser, chatSystem, toolsPath, toolResult;
        for (int i = 1; i < argc - 1; ++i) {
            if (std::strcmp(argv[i], "--longctx") == 0)       longctxPath = argv[i + 1];
            if (std::strcmp(argv[i], "--chat-ids") == 0)      chatIdsPath = argv[i + 1];
            if (std::strcmp(argv[i], "--expert-ladder") == 0) ladderPath  = argv[i + 1];
            if (std::strcmp(argv[i], "--chat") == 0)          chatUser    = argv[i + 1];
            if (std::strcmp(argv[i], "--chat-file") == 0) {   // user turn from a file (long prompts)
                NSString * fc = [NSString stringWithContentsOfFile:@(argv[i + 1])
                                                          encoding:NSUTF8StringEncoding error:nil];
                if (!fc) { std::fprintf(stderr, "cannot read --chat-file %s\n", argv[i + 1]); return 1; }
                chatUser = std::string(fc.UTF8String);
            }
            if (std::strcmp(argv[i], "--system") == 0)        chatSystem  = argv[i + 1];
            if (std::strcmp(argv[i], "--tools") == 0)         toolsPath   = argv[i + 1];
            if (std::strcmp(argv[i], "--tool-result") == 0)   toolResult  = argv[i + 1];
        }
        bool chatThink = hasFlag(argc, argv, "--think");
        std::printf("path     : %s%s%s\n",
                    config.fused ? "FUSED (mx::fast / compile)" : "unfused (research)",
                    config.quantBits ? (std::string("  +Q") + std::to_string(config.quantBits)).c_str() : "",
                    config.quantEmbedBits ? (std::string("+eQ")+std::to_string(config.quantEmbedBits)).c_str() : "");
        if (config.quantKVBits) std::printf("kv-cache : Q%d (quantized_matmul attention)\n", config.quantKVBits);
        std::printf("config   : hidden=%d layers=%d qH=%d kvH(local/global)=%d/%d headDim(l/g)=%d/%d "
                    "softcap=%.1f embedScale=%.4f\n",
                    config.hiddenSize, config.numHiddenLayers, config.numAttentionHeads,
                    config.numKeyValueHeads, config.numGlobalKVHeads, config.headDim, config.globalHeadDim,
                    config.finalLogitSoftcapping, config.embedScale());

        std::printf("loading weights ...\n");
        es::ESWeightLoader weights(modelDir, config);
        std::printf("loaded %zu text tensors\n", weights.count());

        // ---- expert ladder: same prompt, GREEDY, sweeping the router's top-k (the only variable) ----
        // The dense expert compute evaluates all experts regardless; only how many the router keeps
        // (with renormalization) changes. Shows MoE sensitivity to over-/under-selecting experts.
        if (!ladderPath.empty()) {
            es::ESConformance ci(ladderPath);
            std::vector<int> ids = ci.ints("input_ids");
            es::ESTokenizer tokenizer(modelDir + "/tokenizer.json");
            int maxNew = decodeLen > 32 ? decodeLen : 200;
            std::printf("\n-- expert ladder (greedy, %zu prompt tokens, max %d, trained top_k=%d/%d) --\n",
                        ids.size(), maxNew, config.topKExperts, config.numExperts);
            for (int k : {128, 64, 32, 16, 8, 4}) {
                if (k > config.numExperts) continue;
                es::ESModelConfig c = config; c.topKExperts = k;
                es::ESGemma4TextForCausalLM lm(c, weights);   // shares weight arrays (refcounted)
                es::ESSamplingConfig sc; sc.greedy = true; sc.maxNewTokens = maxNew; sc.eosTokenId = 106;
                es::ESGenerationLoop loop(lm, sc);
                std::vector<int> gen = loop.generate(ids);
                std::printf("\n========== top_k = %d / %d ==========\n%s\n",
                            k, config.numExperts, tokenizer.decode(gen, /*skipSpecial=*/false).c_str());
            }
            return 0;
        }

        // ---- chat: build a Gemma-4 chat prompt with ESChatTemplate, generate, parse the response ----
        // Demonstrates the full pipeline: roles (system/user/model), the reasoning toggle (--think),
        // and response parsing (thought channel + tool calls separated from the visible answer).
        if (!chatUser.empty()) {
            es::ESTokenizer tokenizer(modelDir + "/tokenizer.json");
            es::ESChatTemplate chat(tokenizer);

            std::vector<es::ESChatMessage> msgs;
            if (!chatSystem.empty()) msgs.push_back({"system", chatSystem});
            msgs.push_back({"user", chatUser});

            // --tools <file>: one declaration body per non-empty line, reference grammar
            // (declaration:NAME{...} with <|"|> markers around strings).
            std::vector<std::string> toolDecls;
            if (!toolsPath.empty()) {
                FILE * f = std::fopen(toolsPath.c_str(), "r");
                if (!f) { std::fprintf(stderr, "cannot open --tools %s\n", toolsPath.c_str()); return 1; }
                char buf[16384];
                while (std::fgets(buf, sizeof buf, f)) {
                    std::string line(buf);
                    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
                    if (!line.empty()) toolDecls.push_back(line);
                }
                std::fclose(f);
            }

            std::vector<int> prompt = chat.build(msgs, /*enableThinking=*/chatThink,
                                                 /*addGenerationPrompt=*/true, toolDecls);
            std::printf("\n-- chat (%zu prompt tokens, thinking=%s, %zu tools, max %d) --\n",
                        prompt.size(), chatThink ? "on" : "off", toolDecls.size(), decodeLen);
            if (!chatSystem.empty()) std::printf("system: %s\n", chatSystem.c_str());
            if (chatUser.size() <= 2000) std::printf("user  : %s\n", chatUser.c_str());
            else std::printf("user  : %.200s... (%zu chars)\n", chatUser.c_str(), chatUser.size());
            if (!toolDecls.empty())
                std::printf("\n=== prompt (markers visible) ===\n%s\n=== end prompt ===\n",
                            tokenizer.decode(prompt, /*skipSpecial=*/false).c_str());

            es::ESGemma4TextForCausalLM lm(config, weights);
            es::ESSamplingConfig sc;
            sc.greedy      = !hasFlag(argc, argv, "--sample");
            sc.temperature = 1.0f; sc.topK = 64; sc.topP = 0.95f;
            if (std::string tk = argValue(argc, argv, "--top-k"); !tk.empty()) sc.topK = std::atoi(tk.c_str());
            if (std::string sd = argValue(argc, argv, "--seed"); !sd.empty()) sc.seed = std::strtoull(sd.c_str(), nullptr, 10);
            sc.maxNewTokens = decodeLen;
            sc.eosTokenId   = chat.stopToken();   // <turn|> = 106
            es::ESGenerationLoop loop(lm, sc);

            auto t0 = std::chrono::high_resolution_clock::now();
            double preS = 0;
            std::vector<int> gen;
            const std::string snapPath = argValue(argc, argv, "--kv-snapshot");
            if (!snapPath.empty()) {
                // Marker-snapshot mode: prime N-1 tokens (the last prompt token is always
                // tail, so every restore path produces logits naturally), save with
                // --kv-markers waypoints, and on later runs auto-restore the LONGEST
                // matching prefix — full prompt, or any marked depth with a new tail.
                const int N = (int) prompt.size(), Mp = N - 1;
                es::ESKVCache cache(config.numHiddenLayers);
                auto pf = [&](int p) { return apIdsFingerprint(modelDir, config, prompt, p); };
                int pos = cache.restoreSnapshot(snapPath, pf(Mp), apRawModeFor(config), pf);
                if (pos > 0)
                    std::printf("[kv] restored %d of %d prompt tokens from %s\n", pos, N, snapPath.c_str());
                if (pos < 0) {
                    pos = 0;
                    std::vector<int> marks;
                    std::string ms = argValue(argc, argv, "--kv-markers");
                    for (size_t s = 0; s < ms.size();) {
                        size_t comma = ms.find(',', s);
                        int v = std::atoi(ms.substr(s, comma == std::string::npos ? std::string::npos : comma - s).c_str());
                        if (v >= 4096 && v < Mp) marks.push_back(v);
                        if (comma == std::string::npos) break;
                        s = comma + 1;
                    }
                    std::sort(marks.begin(), marks.end());
                    marks.erase(std::unique(marks.begin(), marks.end()), marks.end());
                    for (int m : marks) {
                        mx::array ll = lm.lastLogits(std::vector<int>(prompt.begin() + pos, prompt.begin() + m),
                                                     &cache, pos);
                        mx::eval(ll); pos = m;
                        cache.markPrefix(m, pf(m));
                        std::printf("[kv] marker at %d\n", m);
                    }
                    if (pos < Mp) {
                        mx::array ll = lm.lastLogits(std::vector<int>(prompt.begin() + pos, prompt.begin() + Mp),
                                                     &cache, pos);
                        mx::eval(ll); pos = Mp;
                    }
                    std::printf("[kv] snapshot %s: %s (%zu markers)\n", snapPath.c_str(),
                                cache.saveSnapshot(snapPath, pf(Mp), Mp) ? "saved" : "SAVE FAILED",
                                marks.size());
                }
                mx::array ll = lm.lastLogits(std::vector<int>(prompt.begin() + pos, prompt.end()), &cache, pos);
                mx::eval(ll);
                preS = secsSince(t0);
                es::ESSampler sampler(sc);
                int p = N;
                int next = sampler.sample(ll); gen.push_back(next);
                for (int s = 1; s < sc.maxNewTokens; ++s) {
                    if (next == sc.eosTokenId) break;
                    ll = lm.lastLogits({next}, &cache, p); p += 1;
                    next = sampler.sample(ll); gen.push_back(next);
                }
            } else {
                gen = loop.generate(prompt, &preS);
            }
            double dt = secsSince(t0), decS = std::max(dt - preS, 1e-6);
            std::printf("prefill %zu tok in %.1fs (%.1f tok/s)   decode %zu tok in %.1fs (%.1f tok/s)\n",
                        prompt.size(), preS, prompt.size() / std::max(preS, 1e-6),
                        gen.size(), decS, gen.size() / decS);

            std::printf("\n=== raw (markers visible) ===\n%s\n",
                        tokenizer.decode(gen, /*skipSpecial=*/false).c_str());
            es::ESParsedResponse pr = chat.parse(gen);
            if (!pr.thought.empty())
                std::printf("\n=== thought (reasoning channel) ===\n%s\n", pr.thought.c_str());
            std::printf("\n=== answer ===\n%s\n", pr.answer.c_str());
            for (const auto & tc : pr.toolCalls)
                std::printf("\n=== tool_call ===\n%s(%s)\n", tc.name.c_str(), tc.args.c_str());

            // --tool-result "<key:val,...>": close the tool loop. Truncate the generation at the
            // first <tool_call|>, splice <|tool_response>response:NAME{...}<tool_response|> (the
            // reference grammar; the model turn stays OPEN), and let the model finish its answer.
            if (!toolResult.empty() && !pr.toolCalls.empty()) {
                size_t cut = 0;
                for (size_t k = 0; k < gen.size(); ++k)
                    if (gen[k] == chat.tokens().toolCallClose) { cut = k + 1; break; }
                std::vector<int> follow = prompt;
                follow.insert(follow.end(), gen.begin(), gen.begin() + cut);
                follow.push_back(chat.tokens().toolRespOpen);
                std::vector<int> resp = chat.encodeQuoted(
                    "response:" + pr.toolCalls[0].name + "{" + toolResult + "}");
                follow.insert(follow.end(), resp.begin(), resp.end());
                follow.push_back(chat.tokens().toolRespClose);

                std::printf("\n-- tool loop: feeding response, continuing (%zu prompt tokens) --\n",
                            follow.size());
                std::vector<int> gen2 = loop.generate(follow);
                std::printf("\n=== raw after tool_response (markers visible) ===\n%s\n",
                            tokenizer.decode(gen2, /*skipSpecial=*/false).c_str());
                es::ESParsedResponse pr2 = chat.parse(gen2);
                std::printf("\n=== final answer ===\n%s\n", pr2.answer.c_str());
            }
            return 0;
        }

        // ---- chat generation from pre-tokenized (chat-templated) ids; decode the response ----
        if (!chatIdsPath.empty()) {
            es::ESConformance ci(chatIdsPath);
            std::vector<int> ids = ci.ints("input_ids");
            es::ESGemma4TextForCausalLM lm(config, weights);
            es::ESTokenizer tokenizer(modelDir + "/tokenizer.json");
            std::printf("\n-- chat generation (%zu prompt tokens, max %d) --\n", ids.size(), decodeLen);

            es::ESSamplingConfig sc;
            sc.greedy = !hasFlag(argc, argv, "--sample");
            sc.temperature = 1.0f; sc.topK = 64; sc.topP = 0.95f;
            if (std::string tk = argValue(argc, argv, "--top-k"); !tk.empty()) sc.topK = std::atoi(tk.c_str());
            if (std::string sd = argValue(argc, argv, "--seed"); !sd.empty()) sc.seed = std::strtoull(sd.c_str(), nullptr, 10);
            sc.maxNewTokens = decodeLen;
            sc.eosTokenId = 106;  // <end_of_turn>
            es::ESGenerationLoop loop(lm, sc);

            auto t0 = std::chrono::high_resolution_clock::now();
            double preS = 0;
            std::vector<int> gen = loop.generate(ids, &preS);
            double dt = secsSince(t0), decS = std::max(dt - preS, 1e-6);
            std::printf("prefill %zu tok in %.1fs (%.1f tok/s)   decode %zu tok in %.1fs (%.1f tok/s)\n\n",
                        ids.size(), preS, ids.size() / std::max(preS, 1e-6),
                        gen.size(), decS, gen.size() / decS);
            std::printf("=== Isolde (via Apertura) ===\n%s\n", tokenizer.decode(gen, /*skipSpecial=*/false).c_str());
            { FILE* f = std::fopen("/tmp/apertura_ids.json", "w"); if (f) { std::fputc('[', f);
              for (size_t k = 0; k < gen.size(); ++k) std::fprintf(f, "%s%d", k ? "," : "", gen[k]);
              std::fputc(']', f); std::fclose(f); } }
            return 0;
        }

        // ---- long-context conformance: exercises the sliding-window boundary (>1024 tokens) ----
        if (!longctxPath.empty()) {
            es::ESConformance lc(longctxPath);
            es::ESGemma4TextForCausalLM lm(config, weights);
            std::vector<int> ids = lc.ints("input_ids");
            int seq = (int) ids.size();
            std::printf("\n-- long-context sliding-window conformance --\n");
            std::printf("seq=%d  window=%d  -> last query masks the first %d tokens in LOCAL layers "
                        "(visible in global)\n", seq, config.slidingWindow,
                        seq > config.slidingWindow ? seq - config.slidingWindow : 0);
            if (seq <= config.slidingWindow)
                std::printf("WARNING: seq <= window, sliding boundary NOT exercised!\n");

            mx::array logits = lm.forward(ids, nullptr, 0);  // [seq, vocab]
            mx::array mineLast = mx::reshape(mx::slice(logits, {seq - 1, 0}, {seq, logits.shape(1)}), {logits.shape(1)});
            mx::array refLast  = mx::astype(lc.get("logits_last"), mx::float32);
            int mineA = es::ESSampler::argmax(mineLast);
            int refA  = es::ESSampler::argmax(refLast);
            mx::array d = mx::max(mx::abs(mx::subtract(mx::astype(mineLast, mx::float32), refLast)));
            mx::eval(d);
            std::printf("argmax(last): mine=%d  oracle=%d  max|Δlogit|=%.3e  %s\n",
                        mineA, refA, d.item<float>(), (mineA == refA ? "MATCH ✅" : "MISMATCH ❌"));

            std::vector<int> refGreedy = lc.ints("greedy_tokens");
            es::ESSamplingConfig sc; sc.greedy = true; sc.maxNewTokens = (int) refGreedy.size(); sc.eosTokenId = -1;
            es::ESGenerationLoop loop(lm, sc);
            std::vector<int> mineGreedy = loop.generate(ids);
            bool gok = (mineGreedy.size() == refGreedy.size());
            for (size_t i = 0; gok && i < mineGreedy.size(); ++i) gok = (mineGreedy[i] == refGreedy[i]);
            std::printf("greedy mine  : "); for (int t : mineGreedy) std::printf("%d ", t); std::printf("\n");
            std::printf("greedy oracle: "); for (int t : refGreedy) std::printf("%d ", t); std::printf("\n");
            bool ok = (mineA == refA) && gok;
            std::printf("\n== LONG-CONTEXT %s (argmax=%s greedy=%s) ==\n",
                        ok ? "PASS" : "FAIL", mineA == refA ? "ok" : "FAIL", gok ? "ok" : "FAIL");
            return ok ? 0 : 1;
        }

        if (fusedLockstepFlag) {
            fusedLockstep(weights, config, prefillLen, decodeLen);
            return 0;
        }

        if (swaLockstepFlag) {
            swaLockstep(weights, config, prefillLen, decodeLen);
            return 0;
        }

        if (swaVerifyFlag) {
            swaVerify(weights, config, prefillLen, decodeLen);
            return 0;
        }

        if (cacheVerifyFlag) {
            cacheVerify(weights, config, prefillLen, decodeLen);
            return 0;
        }

        if (stepVerifyFlag) {
            stepVerify(weights, config, prefillLen, decodeLen);
            return 0;
        }

        if (benchStepFlag) {
            std::printf("\n-- compiled-step benchmark (prefill %d, decode %d) --\n", prefillLen, decodeLen);
            es::ESModelConfig cf = config; cf.fused = true;
            es::ESGemma4TextForCausalLM lmF(cf, weights);
            benchStep(lmF, prefillLen, decodeLen);
            return 0;
        }

        if (benchEagerFlag) {
            std::printf("\n-- eager fused benchmark (prefill %d, decode %d) --\n", prefillLen, decodeLen);
            es::ESModelConfig cf = config; cf.fused = true;
            es::ESGemma4TextForCausalLM lmF(cf, weights);
            std::string snap = argValue(argc, argv, "--kv-snapshot");
            std::string fp;
            if (!snap.empty()) {
                // Fingerprint: everything that changes the cached values or their layout.
                fp = modelDir + "|P=" + std::to_string(prefillLen) +
                     "|qb=" + std::to_string(cf.quantBits) +
                     "|qkv=" + std::to_string(cf.quantKVBits) + (cf.quantKVGlobalOnly ? "g" : "a") +
                     "|chunk=" + std::to_string(cf.prefillChunk);
            }
            benchEager(lmF, prefillLen, decodeLen, snap, fp);
            return 0;
        }

        if (stepLockstepFlag) {
            stepLockstep(weights, config, prefillLen, decodeLen);
            return 0;
        }

        if (headVerifyFlag) {
            headVerify(weights, config, prefillLen, decodeLen);
            return 0;
        }

        if (chunkVerifyFlag) {
            chunkVerify(weights, config, prefillLen, decodeLen,
                        config.prefillChunk > 0 ? config.prefillChunk : 512);
            return 0;
        }

        if (tiledVerifyFlag) {
            tiledVerify(weights, config, prefillLen, decodeLen,
                        config.tiledKChunk > 0 ? config.tiledKChunk : 1024);
            return 0;
        }

        if (tiledLockstepFlag) {
            tiledLockstep(weights, config, prefillLen, decodeLen,
                          config.tiledKChunk > 0 ? config.tiledKChunk : 1024);
            return 0;
        }

        if (rehydrateVerifyFlag) {
            rehydrateVerify(weights, config, prefillLen, decodeLen);
            return 0;
        }

        if (hasFlag(argc, argv, "--marker-verify")) {
            markerVerify(weights, config, modelDir, prefillLen, decodeLen > 0 ? decodeLen : 32);
            return 0;
        }

        if (hasFlag(argc, argv, "--excise-verify")) {
            exciseVerify(weights, config, decodeLen > 0 ? decodeLen : 32);
            return 0;
        }

        if (rawLockstepFlag) {
            rawLockstep(weights, config, prefillLen, decodeLen);
            return 0;
        }

        for (int i = 1; i < argc - 1; ++i)
            if (std::strcmp(argv[i], "--facade-verify") == 0)
                return facadeVerify(modelDir, argv[i + 1]);

        for (int i = 1; i < argc - 1; ++i)
            if (std::strcmp(argv[i], "--persist-verify") == 0)
                return persistVerify(modelDir, argv[i + 1]);

        if (hasFlag(argc, argv, "--tools-verify"))
            return toolsVerify(modelDir);

        if (benchAsync) {
            // Prototype: does keeping the sampled token on-device (no per-token host readback)
            // convert the decode-loop idle the Metal trace found into throughput? Fused path only.
            std::printf("\n-- async decode A/B (prefill %d, decode %d) --\n", prefillLen, decodeLen);
            es::ESModelConfig cf = config; cf.fused = true;
            es::ESGemma4TextForCausalLM lmF(cf, weights);
            benchOne(lmF, "fused", prefillLen, decodeLen);   // context: sync prefill+decode baseline
            benchAsyncOne(lmF, "fused", prefillLen, decodeLen);
            return 0;
        }

        if (bench) {
            std::printf("\n-- benchmark (prefill %d, decode %d) --\n", prefillLen, decodeLen);
            if (config.enableMoeBlock) {
                es::ESModelConfig cs = config; cs.moeSparse = true;
                es::ESGemma4TextForCausalLM lmS(cs, weights);
                char slbl[24]; std::snprintf(slbl, sizeof(slbl), "moe-sparse%s",
                                             config.quantBits ? ("-Q" + std::to_string(config.quantBits)).c_str() : "");
                if (config.quantBits == 0) {
                    // bf16: also bench dense for the comparison (shared weights). Quantized experts
                    // only live on the sparse path, so when quantizing we bench sparse alone.
                    es::ESModelConfig cd = config; cd.moeSparse = false;
                    es::ESGemma4TextForCausalLM lmD(cd, weights);
                    benchOne(lmD, "moe-dense", prefillLen, decodeLen);
                }
                benchOne(lmS, slbl, prefillLen, decodeLen);
            } else if (config.quantBits > 0) {
                es::ESGemma4TextForCausalLM lmq(config, weights);
                char lbl[24];
                std::snprintf(lbl, sizeof(lbl), "Q%d%s%s", config.quantBits,
                              config.quantEmbedBits ? "+e" : "", config.fused ? "-fused" : "");
                benchOne(lmq, lbl, prefillLen, decodeLen);
            } else {
                es::ESModelConfig cu = config; cu.fused = false;
                es::ESModelConfig cf = config; cf.fused = true;
                es::ESGemma4TextForCausalLM lmU(cu, weights);  // shares weight arrays (refcounted)
                es::ESGemma4TextForCausalLM lmF(cf, weights);
                benchOne(lmU, "unfused", prefillLen, decodeLen);
                benchOne(lmF, "fused", prefillLen, decodeLen);
            }
            return 0;
        }

        es::ESConformance conf(fixturesPath);
        es::ESGemma4TextForCausalLM lm(config, weights);

        std::vector<int> inputIds = conf.ints("input_ids");
        std::printf("prompt tokens: ");
        for (int t : inputIds) std::printf("%d ", t);
        std::printf("\n\n");

        int pass = 0, total = 0;
        // --dump-mlx <path>: also write our own tensors out, keyed by fixture name,
        // so the divergence can be localised offline (which layer, which element)
        // instead of only summarised as max/med/p99.
        std::string dumpPath = argValue(argc, argv, "--dump-mlx");
        std::unordered_map<std::string, mx::array> dumpMap;
        auto check = [&](const std::string & label, const mx::array & got, const std::string & ref,
                         float rel, float abs) {
            total++;
            if (!dumpPath.empty()) dumpMap.insert_or_assign(ref, mx::astype(got, mx::float32));
            if (conf.has(ref)) {
                // --ulp: scale-free per-op gate. Absolute tolerances are meaningless across
                // tensors whose magnitudes differ by orders of magnitude.
                bool ok = ulpFlag ? conf.compareUlp(label, got, ref)
                                  : conf.compare(label, got, ref, rel, abs);
                if (ok) pass++;
            }
            else std::printf("[%-26s] (no fixture '%s')\n", label.c_str(), ref.c_str());
        };

        // Fixtures carry a leading batch dim [1, seq, *]; reshape to drop it.
        auto fixt2d = [&](const std::string & name) {
            mx::array a = mx::astype(conf.get(name), config.computeDtype);
            auto sh = a.shape();
            if (sh.size() == 3 && sh[0] == 1) a = mx::reshape(a, {sh[1], sh[2]});
            return a;
        };

        // ---- op-level probes for a local layer (0) and a global layer (5) ----
        // Validates p-RoPE, dual head_dim, dual KV heads, k_eq_v, QK/V-norm at the op level.
        auto probeLayer = [&](int L) {
            std::printf("-- op probe: layer %d (%s) --\n", L, config.isSliding(L) ? "local" : "global");
            int seq = (int) inputIds.size();
            int hd = config.headDimFor(L), nQ = config.numAttentionHeads, nKV = config.kvHeadsFor(L);
            mx::array x = fixt2d("L" + std::to_string(L) + ".input_layernorm");  // attn input
            const auto & W = weights;
            es::ESRMSNorm qN(W.layer(L, "self_attn.q_norm.weight"), config.rmsNormEps);
            es::ESRMSNorm kN(W.layer(L, "self_attn.k_norm.weight"), config.rmsNormEps);
            es::ESRMSNorm vN(config.rmsNormEps);
            es::ESRotaryEmbedding rope(hd, config.isSliding(L) ? config.ropeThetaLocal : config.ropeThetaGlobal,
                                   config.isSliding(L) ? 1.0f : config.globalPartialRotaryFactor, config.computeDtype);
            auto cs = rope.cosSin(seq, 0);

            mx::array q  = qN.forward(mx::reshape(mx::matmul(x, mx::transpose(W.layer(L, "self_attn.q_proj.weight"))), {seq, nQ, hd}));
            mx::array kR = mx::reshape(mx::matmul(x, mx::transpose(W.layer(L, "self_attn.k_proj.weight"))), {seq, nKV, hd});
            mx::array k  = kN.forward(kR);
            mx::array vSrc = config.kEqVFor(L) ? kR
                          : mx::reshape(mx::matmul(x, mx::transpose(W.layer(L, "self_attn.v_proj.weight"))), {seq, nKV, hd});
            char p[40];
            std::snprintf(p, sizeof(p), "L%d.q_norm", L);  check(p, q, p, 5e-2f, 1.0e-2f);
            std::snprintf(p, sizeof(p), "L%d.k_norm", L);  check(p, k, p, 5e-2f, 1.0e-2f);
            std::snprintf(p, sizeof(p), "L%d.v_norm", L);  check(p, vN.forward(vSrc), p, 5e-2f, 1.0e-2f);
            std::snprintf(p, sizeof(p), "L%d.q_rope", L);  check(p, es::ESRotaryEmbedding::apply(q, cs.first, cs.second), p, 5e-2f, 1.0e-2f);
            std::snprintf(p, sizeof(p), "L%d.k_rope", L);  check(p, es::ESRotaryEmbedding::apply(k, cs.first, cs.second), p, 5e-2f, 1.0e-2f);
            // MLP from golden pre_feedforward_layernorm input.
            mx::array preFF = fixt2d("L" + std::to_string(L) + ".pre_feedforward_layernorm");
            es::ESMLPBlock mlp(es::esMakeLinear(W, W.layerKey(L, "mlp.gate_proj.weight"), 0, 64),
                               es::esMakeLinear(W, W.layerKey(L, "mlp.up_proj.weight"),   0, 64),
                               es::esMakeLinear(W, W.layerKey(L, "mlp.down_proj.weight"), 0, 64));
            std::snprintf(p, sizeof(p), "L%d.mlp", L);     check(p, mlp.forward(preFF), p, 2.5e-1f, 4.0e-2f);

            // Attention output after o_proj, before the residual add. This is the only gate that
            // sees the attention computation itself: everything above stops at q/k/v + RoPE, so
            // without it attention is only observable through layer_out and any divergence there
            // has to be attributed by elimination. It matters most on the FUSED path, where
            // mx::fast::scaled_dot_product_attention replaces the manual QK^T/mask/softmax/AV —
            // measured at 69.2% within 1 ULP on layer_out while every other probed op was 100%.
            // The mask comes from the model's own buildMask so the probe cannot drift from the
            // forward pass; pastLen 0 because this is a pure prefill probe with no cache.
            {
                es::ESAttention attn(config, L, W);
                mx::array mask = lm.model().buildMask(seq, /*pastLen=*/0, config.isSliding(L));
                mx::array attnOut = attn.forward(x, cs.first, cs.second, mask, nullptr, 0, nullptr);
                std::snprintf(p, sizeof(p), "L%d.attn_oproj", L);
                check(p, attnOut, p, 2.5e-1f, 4.0e-2f);
            }
        };
        if (probeAllFlag) {
            // Fixtures may carry op-level tensors for a subset of layers (the checked-in one
            // probes a spread; APERTURA_PROBE_LAYERS=all regenerates the full set). probeLayer
            // reads its inputs via conf.get(), which throws on a missing key, so skip layers
            // the fixture does not cover instead of aborting.
            int probed = 0, skipped = 0;
            for (int i = 0; i < config.numHiddenLayers; ++i) {
                if (!conf.has("L" + std::to_string(i) + ".input_layernorm")) { skipped++; continue; }
                probeLayer(i);
                probed++;
            }
            std::printf("-- op probes: %d layers covered by fixture, %d not present --\n", probed, skipped);
        } else {
            probeLayer(0);
            probeLayer(5);
        }
        std::printf("\n");

        // Per-layer isolation needs no cross-layer state — but the elastic models' PLE (per-layer
        // input) and shared-KV are exactly that, so isolation is meaningless there. The chained
        // forward + argmax + greedy is the gate for elastic models.
        if (config.hasPLE() || config.numKvSharedLayers > 0) {
            std::printf("-- per-layer ISOLATION skipped (elastic model: PLE/shared-KV need cross-layer state) --\n");
        } else {
            std::printf("-- per-layer ISOLATION conformance (golden input -> compare output; bf16 gate = abs p99) --\n");
            mx::array embed = fixt2d("embed_scaled");
            double accW1 = 0, accB4 = 0, worstW1 = 1e9; int worstLayer = -1, nAcc = 0;
            for (int i = 0; i < config.numHiddenLayers; ++i) {
                mx::array xIn = (i == 0) ? embed : fixt2d("layer_out." + std::to_string(i - 1));
                mx::array got = lm.model().isolatedLayer(i, xIn);
                char lbl[32]; std::snprintf(lbl, sizeof(lbl), "layer_out.%d", i);
                std::string ref = "layer_out." + std::to_string(i);
                if (ulpFlag) {
                    if (!dumpPath.empty()) dumpMap.insert_or_assign(ref, mx::astype(got, mx::float32));
                    total++;
                    if (conf.has(ref)) {
                        if (conf.compareUlp(lbl, got, ref)) pass++;
                        auto st = conf.ulpStats(got, ref);
                        accW1 += st.pctWithin1; accB4 += st.pctBeyond4; nAcc++;
                        if (st.pctWithin1 < worstW1) { worstW1 = st.pctWithin1; worstLayer = i; }
                    } else std::printf("[%-26s] (no fixture '%s')\n", lbl, ref.c_str());
                } else {
                    // bf16 floor: abs p99 ~ few e-3; rel p99 inflated by near-zero elements (informational).
                    check(lbl, got, ref, 8e-1f, 2.5e-2f);
                }
            }
            if (ulpFlag && nAcc > 0) {
                std::printf("\n  per-op ULP summary over %d layers: mean <=1ulp %.1f%%  mean >4ulp %.2f%%"
                            "  worst layer %d (<=1ulp %.1f%%)\n",
                            nAcc, accW1 / nAcc, accB4 / nAcc, worstLayer, worstW1);
            }
        }

        std::printf("\n-- chained forward (accumulation; gate = argmax) --\n");
        auto tr = lm.model().forwardTrace(inputIds, nullptr, 0);
        conf.compare("embed_scaled", tr.embed, "embed_scaled", 5e-3f, 5e-3f);
        conf.compare("final_norm(chained)", tr.finalNorm, "final_norm", 2.5e-1f, 2.5e-1f);
        mx::array logits = lm.forward(inputIds, nullptr, 0);
        conf.compare("logits(chained)", logits, "logits", 2.0e-1f, 2.0e-1f);

        if (!dumpPath.empty()) {
            dumpMap.insert_or_assign("embed_scaled", mx::astype(tr.embed, mx::float32));
            dumpMap.insert_or_assign("final_norm", mx::astype(tr.finalNorm, mx::float32));
            dumpMap.insert_or_assign("logits", mx::astype(logits, mx::float32));
            for (auto & kv : dumpMap) mx::eval(kv.second);
            mx::save_safetensors(dumpPath, dumpMap);
            std::printf("[dump] wrote %zu tensors -> %s\n", dumpMap.size(), dumpPath.c_str());
        }

        // ---- argmax gate ----
        int seq = logits.shape(0);
        mx::array lastRow = mx::reshape(mx::slice(logits, {seq - 1, 0}, {seq, logits.shape(1)}), {logits.shape(1)});
        int mineArgmax = es::ESSampler::argmax(lastRow);
        int refArgmax  = -1;
        if (conf.has("logits")) {
            mx::array refLogits = fixt2d("logits");  // [seq, vocab]
            int rseq = refLogits.shape(0);
            mx::array refLast = mx::reshape(mx::slice(refLogits, {rseq - 1, 0}, {rseq, refLogits.shape(1)}),
                                            {refLogits.shape(1)});
            refArgmax = es::ESSampler::argmax(mx::astype(refLast, mx::float32));
        }
        bool argmaxMatch = (mineArgmax == refArgmax);
        std::printf("\nargmax(last): mine=%d  oracle=%d  %s\n",
                    mineArgmax, refArgmax, (argmaxMatch ? "MATCH ✅" : "MISMATCH ❌"));
        std::printf("per-layer isolation: %d/%d layers passed\n\n", pass, total);
        bool greedyMatch = true;  // set below if a greedy fixture exists

        // ---- greedy generation conformance ----
        std::vector<int> refGreedy = conf.has("greedy_tokens") ? conf.ints("greedy_tokens") : std::vector<int>{};
        if (!refGreedy.empty()) {
            std::printf("-- greedy generation conformance --\n");
            es::ESSamplingConfig sc;
            sc.greedy = true;
            sc.maxNewTokens = (int) refGreedy.size();
            sc.eosTokenId = -1;  // don't early-stop; match PyTorch's fixed-length capture
            es::ESGenerationLoop loop(lm, sc);
            std::vector<int> mine = loop.generate(inputIds);

            bool ok = (mine.size() == refGreedy.size());
            for (size_t i = 0; ok && i < mine.size(); ++i) ok = (mine[i] == refGreedy[i]);
            std::printf("mine  : "); for (int t : mine) std::printf("%d ", t); std::printf("\n");
            std::printf("oracle: "); for (int t : refGreedy) std::printf("%d ", t); std::printf("\n");
            std::printf("greedy match: %s\n", ok ? "MATCH ✅" : "MISMATCH ❌");
            greedyMatch = ok;
        }

        // ---- tokenizer round-trip + free-form text generation ----
        try {
            es::ESTokenizer tokenizer(modelDir + "/tokenizer.json");

            std::printf("\n-- tokenizer conformance --\n");
            std::vector<int> enc = tokenizer.encode("The quick brown fox", /*addSpecialTokens=*/false);
            std::printf("encode(\"The quick brown fox\"): "); for (int t : enc) std::printf("%d ", t);
            bool tokOk = (enc == inputIds);
            std::printf(" %s\n", tokOk ? "MATCH ✅" : "(differs from fixture input_ids — check special-token policy)");
            std::printf("decode(input_ids): %s\n", tokenizer.decode(inputIds, true).c_str());

            // Optional: AperturaResearch --generate "prompt" [maxNew]
            for (int i = 1; i < argc; ++i) {
                if (std::string(argv[i]) == "--generate" && i + 1 < argc) {
                    std::string prompt = argv[i + 1];
                    int maxNew = (i + 2 < argc) ? std::atoi(argv[i + 2]) : 32;
                    std::printf("\n-- generate (prompt=%s, maxNew=%d) --\n", prompt.c_str(), maxNew);
                    std::vector<int> p = tokenizer.encode(prompt, false);
                    es::ESSamplingConfig sc; sc.greedy = true; sc.maxNewTokens = maxNew; sc.eosTokenId = 1;
                    es::ESGenerationLoop loop(lm, sc);
                    std::vector<int> gen = loop.generate(p);
                    std::printf("%s%s\n", prompt.c_str(), tokenizer.decode(gen, true).c_str());
                    break;
                }
            }
        } catch (const std::exception & e) {
            std::printf("\n[tokenizer] skipped: %s\n", e.what());
        }

        // Fused-path bit-stability: compare fused logits to the unfused (research) path,
        // per-position. Greedy on the degenerate fixture prompt can flip at bf16 near-ties
        // (the prompt self-repeats), so it is informational for the fused path, not a gate.
        bool fusedStable = true;
        if (config.fused) {
            std::printf("\n-- fused vs unfused logits (bit-stability) --\n");
            es::ESModelConfig cu = config; cu.fused = false;
            es::ESGemma4TextForCausalLM lmU(cu, weights);
            mx::array lu = lmU.forward(inputIds, nullptr, 0);   // [seq, vocab]
            int sN = lu.shape(0), agree = 0;
            for (int r = 0; r < sN; ++r) {
                mx::array a = mx::reshape(mx::slice(logits, {r, 0}, {r + 1, logits.shape(1)}), {logits.shape(1)});
                mx::array b = mx::reshape(mx::slice(lu, {r, 0}, {r + 1, lu.shape(1)}), {lu.shape(1)});
                if (es::ESSampler::argmax(a) == es::ESSampler::argmax(b)) agree++;
            }
            mx::array d = mx::max(mx::abs(mx::subtract(mx::astype(logits, mx::float32), mx::astype(lu, mx::float32))));
            mx::eval(d);
            fusedStable = (agree == sN);
            std::printf("per-position argmax agreement: %d/%d   max|Δlogit|=%.3e   %s\n",
                        agree, sN, d.item<float>(), fusedStable ? "STABLE ✅" : "DIVERGED ❌");
        }

        // Hard gate: argmax always. Secondary gate depends on path:
        //  quantized -> argmax only (quant introduces real error; greedy is informational)
        //  fused     -> per-position argmax agreement vs the research path
        //  unfused   -> exact greedy-token match
        std::string secondary;
        bool secondaryOk;
        if (config.quantBits > 0)     { secondaryOk = true;        secondary = "greedy=" + std::string(greedyMatch ? "ok" : "near-tie(info)"); }
        else if (config.fused)        { secondaryOk = fusedStable; secondary = "fused-stable=" + std::string(fusedStable ? "ok" : "FAIL"); }
        else                          { secondaryOk = greedyMatch; secondary = "greedy=" + std::string(greedyMatch ? "ok" : "FAIL"); }

        bool allPass = argmaxMatch && secondaryOk;
        std::printf("\n== CONFORMANCE %s (path=%s%s argmax=%s %s, %d/%d numeric gates) ==\n",
                    allPass ? "PASS" : "FAIL", config.fused ? "fused" : "unfused",
                    config.quantBits ? ("+Q" + std::to_string(config.quantBits)).c_str() : "",
                    argmaxMatch ? "ok" : "FAIL", secondary.c_str(), pass, total);
        return allPass ? 0 : 1;
    }
}
