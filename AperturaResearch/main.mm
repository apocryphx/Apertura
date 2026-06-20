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

static double secsSince(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

// Times prefill (process P tokens) and decode (generate D tokens) throughput for one LM.
static void benchOne(const es::ESGemma4TextForCausalLM & lm, const char * label, int P, int D) {
    std::vector<int> toks(P);
    for (int i = 0; i < P; ++i) toks[i] = 100 + i;  // arbitrary in-vocab ids

    // Prefill (1 warmup + best of 2).
    double bestPre = 1e9;
    for (int it = 0; it < 3; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        mx::array logits = lm.forward(toks, nullptr, 0);
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
        // Positional args (skip --flags and their values like the --generate operand).
        std::vector<std::string> pos;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--generate" || a == "--quant" || a == "--decode") { i++; continue; }
            if (a == "--quant-embed") { if (i + 1 < argc && std::atoi(argv[i + 1]) > 0) i++; continue; }
            if (a == "--longctx" || a == "--quant-kv" || a == "--prefill" || a == "--chat-ids"
                || a == "--expert-ladder" || a == "--chat" || a == "--system"
                || a == "--export" || a == "--quant-group" || a == "--verify-bundle"
                || a == "--vs-bf16" || a == "--prompt-file") { i++; continue; }
            if (a.rfind("--", 0) == 0) continue;
            pos.push_back(a);
        }
        std::string modelDir = pos.size() > 0 ? pos[0] : kDefaultModelDir;
        std::string fixturesPath =
            pos.size() > 1 ? pos[1]
                           : "/Users/apocryphx/Documents/GitHub/Apertura/aptransformerTests/Fixtures/fixtures.safetensors";

        std::printf("== Apertura conformance ==\n");
        std::printf("modelDir : %s\n", modelDir.c_str());
        std::printf("fixtures : %s\n", fixturesPath.c_str());

        es::ESModelConfig config = es::ESModelConfig::fromConfigJSON(modelDir + "/config.json");
        config.computeDtype = mx::bfloat16;
        bool useFused = hasFlag(argc, argv, "--fused");
        bool bench    = hasFlag(argc, argv, "--bench");
        config.fused  = useFused;
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

        int decodeLen = 32, prefillLen = 64;
        for (int i = 1; i < argc - 1; ++i) {
            if (std::strcmp(argv[i], "--decode") == 0)  decodeLen  = std::atoi(argv[i + 1]);
            if (std::strcmp(argv[i], "--prefill") == 0) prefillLen = std::atoi(argv[i + 1]);
        }
        std::string longctxPath, chatIdsPath, ladderPath, chatUser, chatSystem;
        for (int i = 1; i < argc - 1; ++i) {
            if (std::strcmp(argv[i], "--longctx") == 0)       longctxPath = argv[i + 1];
            if (std::strcmp(argv[i], "--chat-ids") == 0)      chatIdsPath = argv[i + 1];
            if (std::strcmp(argv[i], "--expert-ladder") == 0) ladderPath  = argv[i + 1];
            if (std::strcmp(argv[i], "--chat") == 0)          chatUser    = argv[i + 1];
            if (std::strcmp(argv[i], "--system") == 0)        chatSystem  = argv[i + 1];
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

            std::vector<int> prompt = chat.build(msgs, /*enableThinking=*/chatThink,
                                                 /*addGenerationPrompt=*/true);
            std::printf("\n-- chat (%zu prompt tokens, thinking=%s, max %d) --\n",
                        prompt.size(), chatThink ? "on" : "off", decodeLen);
            if (!chatSystem.empty()) std::printf("system: %s\n", chatSystem.c_str());
            std::printf("user  : %s\n", chatUser.c_str());

            es::ESGemma4TextForCausalLM lm(config, weights);
            es::ESSamplingConfig sc;
            sc.greedy      = !hasFlag(argc, argv, "--sample");
            sc.temperature = 1.0f; sc.topK = 64; sc.topP = 0.95f;
            sc.maxNewTokens = decodeLen;
            sc.eosTokenId   = chat.stopToken();   // <turn|> = 106
            es::ESGenerationLoop loop(lm, sc);

            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<int> gen = loop.generate(prompt);
            double dt = secsSince(t0);
            std::printf("generated %zu tokens in %.1fs (%.1f tok/s)\n",
                        gen.size(), dt, gen.size() / std::max(dt, 1e-6));

            es::ESParsedResponse pr = chat.parse(gen);
            if (!pr.thought.empty())
                std::printf("\n=== thought (reasoning channel) ===\n%s\n", pr.thought.c_str());
            std::printf("\n=== answer ===\n%s\n", pr.answer.c_str());
            for (const auto & tc : pr.toolCalls)
                std::printf("\n=== tool_call ===\n%s(%s)\n", tc.name.c_str(), tc.args.c_str());
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
            sc.maxNewTokens = decodeLen;
            sc.eosTokenId = 106;  // <end_of_turn>
            es::ESGenerationLoop loop(lm, sc);

            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<int> gen = loop.generate(ids);
            double dt = secsSince(t0);
            std::printf("generated %zu tokens in %.1fs (%.1f tok/s)\n\n", gen.size(), dt,
                        gen.size() / std::max(dt, 1e-6));
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
        auto check = [&](const std::string & label, const mx::array & got, const std::string & ref,
                         float rel, float abs) {
            total++;
            if (conf.has(ref)) { if (conf.compare(label, got, ref, rel, abs)) pass++; }
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
        };
        probeLayer(0);
        probeLayer(5);
        std::printf("\n");

        // Per-layer isolation needs no cross-layer state — but the elastic models' PLE (per-layer
        // input) and shared-KV are exactly that, so isolation is meaningless there. The chained
        // forward + argmax + greedy is the gate for elastic models.
        if (config.hasPLE() || config.numKvSharedLayers > 0) {
            std::printf("-- per-layer ISOLATION skipped (elastic model: PLE/shared-KV need cross-layer state) --\n");
        } else {
            std::printf("-- per-layer ISOLATION conformance (golden input -> compare output; bf16 gate = abs p99) --\n");
            mx::array embed = fixt2d("embed_scaled");
            for (int i = 0; i < config.numHiddenLayers; ++i) {
                mx::array xIn = (i == 0) ? embed : fixt2d("layer_out." + std::to_string(i - 1));
                mx::array got = lm.model().isolatedLayer(i, xIn);
                char lbl[32]; std::snprintf(lbl, sizeof(lbl), "layer_out.%d", i);
                // bf16 floor: abs p99 ~ few e-3; rel p99 inflated by near-zero elements (informational).
                check(lbl, got, "layer_out." + std::to_string(i), 8e-1f, 2.5e-2f);
            }
        }

        std::printf("\n-- chained forward (accumulation; gate = argmax) --\n");
        auto tr = lm.model().forwardTrace(inputIds, nullptr, 0);
        conf.compare("embed_scaled", tr.embed, "embed_scaled", 5e-3f, 5e-3f);
        conf.compare("final_norm(chained)", tr.finalNorm, "final_norm", 2.5e-1f, 2.5e-1f);
        mx::array logits = lm.forward(inputIds, nullptr, 0);
        conf.compare("logits(chained)", logits, "logits", 2.0e-1f, 2.0e-1f);

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
