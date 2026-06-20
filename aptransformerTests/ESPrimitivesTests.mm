//  ESPrimitivesTests — fast, weight-free unit tests for the MLX math primitives.
//
//  The full 31B forward-pass / generation conformance is exercised by the AperturaResearch
//  CLI driver (it loads the 58 GB model + PyTorch fixtures and gates on argmax/greedy match —
//  too heavy for a per-run XCTest). These tests cover the pure primitives that need no weights
//  and run in milliseconds: rotate_half, gelu_pytorch_tanh, RMSNorm, repeat_kv, RoPE structure,
//  and the proportional-RoPE (p-RoPE) zero-frequency tail.

#import <XCTest/XCTest.h>
#include "mlx/mlx.h"
#include "ESOps.h"
#include "ESRMSNorm.h"
#include "ESRotaryEmbedding.h"
#include "ESLinear.h"
#include "ESRouter.h"
#include "ESWeightLoader.h"
#include "mlx/random.h"
#import <Foundation/Foundation.h>

namespace mx = mlx::core;
using namespace es;

static float maxAbsDiff(const mx::array & a, const mx::array & b) {
    mx::array d = mx::max(mx::abs(mx::subtract(mx::astype(a, mx::float32), mx::astype(b, mx::float32))));
    mx::eval(d);
    return d.item<float>();
}

@interface ESPrimitivesTests : XCTestCase
@end

@implementation ESPrimitivesTests

- (void)testRotateHalf {
    // rotate_half([1,2,3,4]) = [-3,-4,1,2]
    float in[] = {1, 2, 3, 4};
    mx::array x(in, {1, 4}, mx::float32);
    mx::array got = rotateHalf(x);
    float exp[] = {-3, -4, 1, 2};
    mx::array ref(exp, {1, 4}, mx::float32);
    XCTAssertLessThan(maxAbsDiff(got, ref), 1e-6f);
}

- (void)testGeluTanhMatchesFormula {
    float in[] = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    mx::array x(in, {5}, mx::float32);
    mx::array got = geluTanh(x);
    mx::eval(got);
    const float * g = got.data<float>();
    for (int i = 0; i < 5; ++i) {
        double v = in[i];
        double inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
        double ref = 0.5 * v * (1.0 + std::tanh(inner));
        XCTAssertLessThanOrEqual(std::fabs(g[i] - ref), 1e-5, @"gelu mismatch at %d", i);
    }
}

- (void)testRMSNormUnitWeightNormalizes {
    // With weight=1, output should have unit RMS (within eps).
    int d = 8;
    std::vector<float> xv(d), wv(d, 1.0f);
    for (int i = 0; i < d; ++i) xv[i] = (float) (i - 3);  // some spread, nonzero
    mx::array x(xv.data(), {1, d}, mx::float32);
    mx::array w(wv.data(), {d}, mx::float32);
    ESRMSNorm norm(w, 1e-6f);
    mx::array y = norm.forward(x);
    mx::array ms = mx::mean(mx::multiply(y, y), -1, true);  // mean square of output
    mx::eval(ms);
    XCTAssertLessThanOrEqual(std::fabs(ms.item<float>() - 1.0f), 1e-3f);
}

- (void)testRMSNormWeightless {
    int d = 4;
    float xv[] = {2, 2, 2, 2};
    mx::array x(xv, {1, d}, mx::float32);
    ESRMSNorm norm(1e-6f);  // with_scale = false
    mx::array y = norm.forward(x);
    mx::eval(y);
    const float * p = y.data<float>();
    for (int i = 0; i < d; ++i) XCTAssertLessThanOrEqual(std::fabs(p[i] - 1.0f), 1e-3f);
}

- (void)testRepeatKVOrder {
    // [h0, h1] with nrep 2 -> [h0, h0, h1, h1]  (contiguous per kv head)
    float in[] = {0, 1};  // 2 kv heads, seq 1, dim 1
    mx::array x(in, {2, 1, 1}, mx::float32);
    mx::array got = repeatKV(x, 2);
    mx::eval(got);
    XCTAssertEqual(got.shape(0), 4);
    const float * p = got.data<float>();
    float exp[] = {0, 0, 1, 1};
    for (int i = 0; i < 4; ++i) XCTAssertEqual(p[i], exp[i]);
}

- (void)testLocalRoPEFullRotation {
    // Local RoPE (full rotation): at position 0, cos=1, sin=0 -> identity on any input.
    int hd = 8;
    ESRotaryEmbedding rope(hd, 10000.0f, /*partial=*/1.0f, mx::float32);
    auto cs = rope.cosSin(1, 0);
    mx::eval(cs.first, cs.second);
    // pos 0 -> all angles 0 -> cos all 1, sin all 0
    XCTAssertLessThan(maxAbsDiff(cs.first, mx::ones({1, hd}, mx::float32)), 1e-6f);
    XCTAssertLessThan(maxAbsDiff(cs.second, mx::zeros({1, hd}, mx::float32)), 1e-6f);
}

- (void)testFusedGeluMatchesManual {
    // Phase-2 fused micro-op must be bit-identical to the research path.
    mx::array x = mx::random::normal({64}, mx::float32);
    XCTAssertLessThan(maxAbsDiff(geluTanhFused(x), geluTanh(x)), 1e-6f);
}

- (void)testFusedRMSNormMatchesManual {
    // fused = mx::fast::rms_norm; unfused = manual f32. Agree within the bf16/f32 floor.
    int d = 5376;
    mx::array x = mx::random::normal({3, d}, mx::float32);
    mx::array w = mx::random::normal({d}, mx::float32);
    ESRMSNorm manual(w, 1e-6f, /*fused=*/false);
    ESRMSNorm fused(w, 1e-6f, /*fused=*/true);
    XCTAssertLessThan(maxAbsDiff(manual.forward(x), fused.forward(x)), 1e-4f);
}

- (void)testQuantizedLinearCloseToBf16 {
    // Phase-4: 8-bit affine quantized linear stays close to the bf16 matmul.
    int out = 128, in = 256;
    mx::array w = mx::random::normal({out, in}, mx::float32);
    mx::array x = mx::random::normal({4, in}, mx::float32);
    ESLinear full(w, /*quantBits=*/0, 64);
    ESLinear q8(w, /*quantBits=*/8, 64);
    mx::array yf = full.forward(x);
    mx::array yq = q8.forward(x);
    mx::array scale = mx::max(mx::abs(yf));
    mx::eval(scale);
    XCTAssertLessThan(maxAbsDiff(yf, yq), 0.05f * scale.item<float>());  // <5% of peak
}

- (void)testMoERouterTopKWeights {
    // Router output W [seq, E] must have exactly topK nonzeros per row, and (without per-expert
    // scaling, i.e. scale=ones) those weights renormalize to sum 1 per token.
    int H = 64, E = 16, K = 4, seq = 3;
    mx::array proj  = mx::random::normal({E, H}, mx::float32);
    mx::array scale = mx::ones({H}, mx::float32);
    mx::array pes   = mx::ones({E}, mx::float32);          // per-expert scale = 1 -> weights sum to 1
    es::ESRouter router(proj, scale, pes, E, K, H, 1e-6f, mx::float32);
    mx::array x = mx::random::normal({seq, H}, mx::float32);
    mx::array W = router.routeWeights(x);                  // [seq, E]
    XCTAssertEqual(W.shape(0), seq); XCTAssertEqual(W.shape(1), E);
    mx::array nonzero = mx::sum(mx::astype(mx::greater(W, mx::array(0.0f)), mx::int32), -1);  // per row
    mx::array rowsum  = mx::sum(W, -1);
    mx::eval(nonzero, rowsum);
    for (int s = 0; s < seq; ++s) {
        XCTAssertEqual(nonzero.data<int>()[s], K, @"row %d should route to exactly K experts", s);
        XCTAssertLessThanOrEqual(std::fabs(rowsum.data<float>()[s] - 1.0f), 1e-4, @"weights sum to 1");
    }
}

- (void)testProportionalRoPEZeroTail {
    // Global p-RoPE: headDim 512, partial 0.25 -> 64 real freqs, 192 zeros.
    // At a nonzero position, the zero-frequency dims must have cos=1, sin=0 (no rotation).
    int hd = 512;
    ESRotaryEmbedding rope(hd, 1000000.0f, 0.25f, mx::float32);
    auto cs = rope.cosSin(2, 0);
    mx::array cos = cs.first, sin = cs.second;  // [2, 512]
    mx::eval(cos, sin);
    const float * c = cos.data<float>();
    const float * s = sin.data<float>();
    // Row 1 (position 1). Layout: emb = cat(freqs, freqs). freqs[64:256] are zero ->
    // cos indices [64,256) and [320,512) == 1, sin == 0.
    int row = 1 * hd;
    for (int i = 64; i < 256; ++i) {
        XCTAssertLessThanOrEqual(std::fabs(c[row + i] - 1.0f), 1e-5f, @"cos tail %d", i);
        XCTAssertLessThanOrEqual(std::fabs(s[row + i]), 1e-5f, @"sin tail %d", i);
    }
    // The first rotated frequency at a nonzero position must actually rotate (sin != 0).
    XCTAssertGreaterThan(std::fabs(s[row + 0]), 1e-3f);
}

// --- Quantized .apml bundle export -----------------------------------------
// Synthesizes a tiny HF-style model snapshot, runs exportQuantizedBundle, and
// verifies the package: quantized projections become (weight u32 + scales +
// biases), the embedding is quantized at embed_bits, norms stay bf16, aux files
// are copied, and the manifest is self-describing. Tiny tensors -> runs fast.
- (void)testExportQuantizedBundle {
    @autoreleasepool {
        NSFileManager * fm = [NSFileManager defaultManager];
        NSString * base = [NSTemporaryDirectory() stringByAppendingPathComponent:
                           [@"apml-test-" stringByAppendingString:[[NSUUID UUID] UUIDString]]];
        NSString * modelDir = [base stringByAppendingPathComponent:@"model"];
        [fm createDirectoryAtPath:modelDir withIntermediateDirectories:YES attributes:nil error:nil];

        const int hidden = 128, vocab = 256;  // last dims divisible by group_size 64
        auto bf16 = [](const mx::Shape & shape) {
            return mx::astype(mx::random::normal(shape, mx::float32), mx::bfloat16);
        };
        const std::string P = "model.language_model.";
        std::unordered_map<std::string, mx::array> w = {
            {P + "embed_tokens.weight",                          bf16({vocab, hidden})},   // -> q (embed_bits)
            {P + "norm.weight",                                  bf16({hidden})},           // -> bf16
            {P + "layers.0.self_attn.q_proj.weight",             bf16({hidden, hidden})},   // -> q (bits)
            {P + "layers.0.mlp.gate_proj.weight",                bf16({hidden, hidden})},   // -> q (bits)
            {P + "layers.0.input_layernorm.weight",              bf16({hidden})},           // -> bf16
        };
        mx::save_safetensors([[modelDir stringByAppendingPathComponent:@"model.safetensors"] UTF8String], w);

        // Minimal aux files (exporter reads model_type and copies these verbatim).
        [@"{\"model_type\":\"gemma4\"}" writeToFile:[modelDir stringByAppendingPathComponent:@"config.json"]
                                         atomically:YES encoding:NSUTF8StringEncoding error:nil];
        [@"{\"version\":\"1.0\"}" writeToFile:[modelDir stringByAppendingPathComponent:@"tokenizer.json"]
                                   atomically:YES encoding:NSUTF8StringEncoding error:nil];

        NSString * outPath = [base stringByAppendingPathComponent:@"Tiny-Q4.apml"];
        es::ESBundleExportOptions opts;  // bits=4, group_size=64, embed_bits=8, variant "mlx-q4"
        std::string err;
        bool ok = es::exportQuantizedBundle([modelDir UTF8String], [outPath UTF8String], opts, &err);
        XCTAssertTrue(ok, @"export failed: %s", err.c_str());

        BOOL isDir = NO;
        XCTAssertTrue([fm fileExistsAtPath:outPath isDirectory:&isDir] && isDir, @"package missing");
        XCTAssertTrue([fm fileExistsAtPath:[outPath stringByAppendingPathComponent:@"tokenizer.json"]],
                      @"tokenizer.json not copied");

        // Manifest is self-describing.
        NSData * md = [NSData dataWithContentsOfFile:[outPath stringByAppendingPathComponent:@"manifest.json"]];
        NSDictionary * manifest = [NSJSONSerialization JSONObjectWithData:md options:0 error:nil];
        XCTAssertEqualObjects(manifest[@"kind"], @"apertura-model");
        XCTAssertEqualObjects(manifest[@"default_variant"], @"mlx-q4");
        XCTAssertEqualObjects(manifest[@"architecture"], @"gemma4");

        // Weights: quantized projections -> {u32 weight, scales, biases}; norms stay bf16.
        std::string st = [[outPath stringByAppendingPathComponent:@"weights/mlx-q4/model.safetensors"] UTF8String];
        auto loaded = mx::load_safetensors(st);
        auto & m = loaded.first;
        XCTAssertEqual(m.count("layers.0.self_attn.q_proj.weight"), 1u);
        XCTAssertEqual(m.count("layers.0.self_attn.q_proj.weight.scales"), 1u);
        XCTAssertEqual(m.count("layers.0.self_attn.q_proj.weight.biases"), 1u);
        XCTAssertTrue(m.at("layers.0.self_attn.q_proj.weight").dtype() == mx::uint32, @"packed weight must be u32");
        XCTAssertEqual(m.count("embed_tokens.weight.scales"), 1u, @"embedding should be quantized");
        XCTAssertEqual(m.count("norm.weight"), 1u);
        XCTAssertEqual(m.count("norm.weight.scales"), 0u, @"norm must stay bf16");
        XCTAssertTrue(m.at("norm.weight").dtype() == mx::bfloat16, @"norm must stay bf16");

        [fm removeItemAtPath:base error:nil];
    }
}

// --- Round-trip: reload-from-.apml == in-memory-quantize -------------------
// The acceptance bar for the bundle. Export a weight, reload it through the
// bundle-mode loader + factory, and assert the forward output is bit-identical
// to the path that quantizes the same bf16 weight in memory at load. (The full
// 31B argmax-stable conformance runs via the AperturaResearch driver; this is
// the same invariant on tiny tensors, runnable per-build.)
- (void)testReloadMatchesInMemoryQuant {
    @autoreleasepool {
        NSFileManager * fm = [NSFileManager defaultManager];
        NSString * baseRT = [NSTemporaryDirectory() stringByAppendingPathComponent:
                             [@"apml-rt-" stringByAppendingString:[[NSUUID UUID] UUIDString]]];
        NSString * modelDir = [baseRT stringByAppendingPathComponent:@"model"];
        [fm createDirectoryAtPath:modelDir withIntermediateDirectories:YES attributes:nil error:nil];

        const int hidden = 128, vocab = 256;
        auto bf16 = [](const mx::Shape & s) { return mx::astype(mx::random::normal(s, mx::float32), mx::bfloat16); };
        mx::array Wq = bf16({hidden, hidden});  // a q_proj
        mx::array E  = bf16({vocab, hidden});   // the embedding
        mx::array x  = bf16({4, hidden});
        mx::array h  = bf16({3, hidden});

        const std::string P = "model.language_model.";
        std::unordered_map<std::string, mx::array> w = {
            {P + "embed_tokens.weight",              E},
            {P + "layers.0.self_attn.q_proj.weight", Wq},
        };
        mx::save_safetensors([[modelDir stringByAppendingPathComponent:@"model.safetensors"] UTF8String], w);
        [@"{\"model_type\":\"gemma4\"}" writeToFile:[modelDir stringByAppendingPathComponent:@"config.json"]
                                         atomically:YES encoding:NSUTF8StringEncoding error:nil];

        NSString * apml = [baseRT stringByAppendingPathComponent:@"rt.apml"];
        es::ESBundleExportOptions opts;  // bits 4, group_size 64, embed_bits 8
        std::string err;
        XCTAssertTrue(es::exportQuantizedBundle([modelDir UTF8String], [apml UTF8String], opts, &err),
                      @"export failed: %s", err.c_str());

        // In-memory-quantize path (what load does today).
        ESLinear    linMem(Wq, opts.bits, opts.groupSize);
        ESEmbedding embMem(E, opts.embedBits, opts.groupSize);

        // Reload-from-bundle path.
        es::ESModelConfig cfg;
        es::ESWeightLoader loader([apml UTF8String], cfg);
        XCTAssertTrue(loader.isBundle(), @"loader should be in bundle mode");
        ESLinear    linDisk = es::esMakeLinear(loader, "layers.0.self_attn.q_proj.weight", 0, 0);
        ESEmbedding embDisk = es::esMakeEmbedding(loader, "embed_tokens.weight", 0, 0);

        // Same mx::quantize inputs in both paths -> identical packed weights -> identical forward.
        XCTAssertEqual(maxAbsDiff(linMem.forward(x), linDisk.forward(x)), 0.0f, @"q_proj reload != in-memory");
        XCTAssertEqual(maxAbsDiff(embMem.logits(h), embDisk.logits(h)), 0.0f, @"embed reload != in-memory");

        [fm removeItemAtPath:baseRT error:nil];
    }
}

@end
