//
//  ESVisionBudgetTests.mm
//  aptransformerTests
//
//  The visual-token budget (70/140/280/560/1120) through the image processor and the E2B
//  tower, gated bit-exact / at the per-op floor against the HF processors
//  (Tools/generate_budget_fixtures.py: both families, three pictures, every budget; the
//  tower+pooler at 70 and 560). The tower test needs the E2B source checkpoint and skips
//  without it.
//

#import <XCTest/XCTest.h>
#include "ESConformance.h"
#include "ESImagePreprocessor.h"
#include "ESModelConfig.h"
#include "ESWeightLoader.h"
#include "ESGemma4VisionModel.h"
#include "ESVisionPooler.h"
#include "ESMultimodalEmbedder.h"
#include <string>

namespace mx = mlx::core;

static NSString * APTFixturesDir(void) {
    return [[[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"Fixtures"];
}
static const char * kE2BSource = "/Users/apocryphx/.cache/huggingface/sources/gemma-4-E2B-it-qat-q4_0-unquantized";

@interface ESVisionBudgetTests : XCTestCase
@end

@implementation ESVisionBudgetTests {
    std::unique_ptr<es::ESConformance> _conf;
}

- (void)setUp {
    NSString * p = [APTFixturesDir() stringByAppendingPathComponent:@"fixtures_vision_budgets.safetensors"];
    if (![NSFileManager.defaultManager fileExistsAtPath:p])
        XCTSkip(@"fixtures_vision_budgets.safetensors missing — run Tools/generate_budget_fixtures.py");
    _conf = std::make_unique<es::ESConformance>(std::string(p.UTF8String));
}

// fixture[0, :rows] -> [rows, D]
- (mx::array)rows:(const std::string &)name count:(int)rows {
    mx::array a = _conf->get(name);
    auto sh = a.shape();
    mx::Shape start(sh.size(), 0), stop = sh; stop[1] = rows;
    return mx::reshape(mx::slice(a, start, stop), {rows, sh[2]});
}

- (double)exactPercent:(const mx::array &)got ref:(const mx::array &)ref {
    if (got.shape() != ref.shape()) return -1;
    mx::array eq = mx::sum(mx::astype(mx::equal(mx::astype(got, mx::float32), mx::astype(ref, mx::float32)), mx::float32));
    mx::eval(eq);
    return 100.0 * eq.item<float>() / (double) got.size();
}

- (void)checkFamily:(es::ESVisionKind)kind prefix:(const char *)fam {
    es::ESVisionConfig cfg;   // E2B/12B share patch 16, pooling 3, 280 default
    cfg.present = true; cfg.kind = kind;
    es::ESImagePreprocessor pre(cfg);
    const char * pics[] = {"e2b", "tall", "wide"};
    const char * files[] = {"fixtures_vision_e2b.png", "fixtures_budget_tall.png", "fixtures_budget_wide.png"};
    for (int b : es::ESImagePreprocessor::kSupportedBudgets) {
        for (int i = 0; i < 3; ++i) {
            std::string key = std::string(fam) + ".b" + std::to_string(b) + "." + pics[i] + ".";
            NSString * png = [APTFixturesDir() stringByAppendingPathComponent:@(files[i])];
            es::ESImagePatches p = pre.processFile(std::string(png.UTF8String), b);
            int n = _conf->ints(key + "num_soft_tokens")[0];
            XCTAssertEqual(p.numSoftTokens(), n, @"%s soft tokens", key.c_str());
            XCTAssertEqual(p.budget, b);
            const int rows = p.numPatches();
            XCTAssertEqual([self exactPercent:p.pixelValues ref:[self rows:key + "pixel_values" count:rows]], 100.0,
                           @"%s pixel_values", key.c_str());
            XCTAssertEqual([self exactPercent:p.positionIds ref:[self rows:key + "image_position_ids" count:rows]], 100.0,
                           @"%s positions", key.c_str());
            // The template's count must come out the same without touching pixels.
            CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:png], NULL);
            CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
            XCTAssertEqual(pre.softTokensForSize((int) CGImageGetHeight(img), (int) CGImageGetWidth(img), b), n);
            CGImageRelease(img); CFRelease(src);
        }
    }
}

- (void)testTowerProcessorAtEveryBudget   { [self checkFamily:es::ESVisionKind::tower prefix:"tower"]; }
- (void)testUnifiedProcessorAtEveryBudget { [self checkFamily:es::ESVisionKind::unified prefix:"unified"]; }

- (void)testUnsupportedBudgetIsRejected {
    es::ESVisionConfig cfg; cfg.present = true; cfg.kind = es::ESVisionKind::tower;
    es::ESImagePreprocessor pre(cfg);
    XCTAssertEqual(pre.resolveBudget(0), 280);
    XCTAssertThrows(pre.resolveBudget(100));
    XCTAssertThrows(pre.targetSize(640, 400, 300));
    XCTAssertTrue(es::ESImagePreprocessor::isSupportedBudget(1120));
    XCTAssertFalse(es::ESImagePreprocessor::isSupportedBudget(0));
}

// Pooler + embedder at 70 and 560, teacher-forced from the reference's own last encoder layer
// output (per-op floor: >= 85% within 1 bf16 ULP, <= 5% beyond 4 ULP — the decoder's per-layer
// thresholds). The chained whole-tower result is logged only: 16 bidirectional bf16 layers over
// thousands of patches drift from the CPU reference exactly as the text chain does, and the
// end-to-end gate for that is the argmax/greedy of --vision-conformance.
- (void)testPoolerAndEmbedderAtNonDefaultBudgets {
    if (![NSFileManager.defaultManager fileExistsAtPath:[@(kE2BSource) stringByAppendingPathComponent:@"model.safetensors"]])
        XCTSkip(@"E2B source checkpoint not present");
    es::ESModelConfig config = es::ESModelConfig::fromConfigJSON(std::string(kE2BSource) + "/config.json");
    config.computeDtype = mx::bfloat16;
    es::ESWeightLoader weights(kE2BSource, config);
    es::ESGemma4VisionModel vision(config, weights);
    es::ESImagePreprocessor pre(config.vision);
    es::ESVisionPooler pooler(config.vision.hiddenSize, config.vision.poolingKernelSize);
    es::ESMultimodalEmbedder emb(es::esMakeLinear(weights, "embed_vision.embedding_projection.weight", 0, 64),
                                 config.vision.rmsNormEps);
    NSString * png = [APTFixturesDir() stringByAppendingPathComponent:@"fixtures_vision_e2b.png"];
    for (int b : {70, 560}) {
        es::ESImagePatches p = pre.processFile(std::string(png.UTF8String), b);
        std::string key = "tower.b" + std::to_string(b) + ".e2b.";
        const int nSoft = p.numSoftTokens();
        XCTAssertEqual(nSoft, _conf->get(key + "tower_out").shape(0));
        // pooler: golden last-layer rows (valid patches) -> f32 pooled -> bf16 == tower_out
        mx::array last = mx::astype([self rows:key + "last_layer" count:p.numPatches()], mx::bfloat16);
        mx::array pooled = mx::astype(pooler.forward(last, p.positionIds, nSoft), mx::bfloat16);
        auto sp = es::ESConformance::ulpStatsArrays(pooled, _conf->get(key + "tower_out"));
        XCTAssertGreaterThanOrEqual(sp.pctWithin1, 85.0, @"budget %d pooler within 1 ULP", b);
        XCTAssertLessThanOrEqual(sp.pctBeyond4, 5.0, @"budget %d pooler beyond 4 ULP", b);
        // embedder: golden tower_out -> soft tokens
        mx::array soft = emb.forward(mx::astype(_conf->get(key + "tower_out"), mx::bfloat16));
        auto se = es::ESConformance::ulpStatsArrays(soft, _conf->get(key + "embed_vision"));
        XCTAssertGreaterThanOrEqual(se.pctWithin1, 85.0, @"budget %d embedder within 1 ULP", b);
        XCTAssertLessThanOrEqual(se.pctBeyond4, 5.0, @"budget %d embedder beyond 4 ULP", b);
        // chained (informational)
        auto tr = vision.forwardTrace(p);
        auto sc = es::ESConformance::ulpStatsArrays(tr.softTokens, _conf->get(key + "embed_vision"));
        NSLog(@"[budget %d] pooler <=1ulp %.1f%% | embedder <=1ulp %.1f%% | chained soft tokens <=1ulp %.1f%% (informational)",
              b, sp.pctWithin1, se.pctWithin1, sc.pctWithin1);
    }
}

@end
