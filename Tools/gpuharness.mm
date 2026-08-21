// gpuharness — standalone Metal harness for the Apertura decode-attention kernel.
// No MLX: compiles the kernel source directly, so MTLCaptureManager captures the real
// dispatch and `gpudebug` can replay + profile it headlessly (counters!).
//
//   ./gpuharness                 — time the kernel (GPU timestamps, median of 10)
//   ./gpuharness capture out.gputrace [kernel.metal] — capture one dispatch for gpudebug
//   ./gpuharness kernel.metal    — use an alternate kernel source file (tuning loop)
//   ./gpuharness compile kernel.metal — compile + pipeline only (no dispatch; safe
//                                  while the GPU is busy with something else)
//   ./gpuharness verify kernel.metal [--tg N] — run the BUILT-IN v3 and the variant,
//                                  compare out_m/out_l/out_acc (the register-resident
//                                  redesign gate: parity within reassociation noise)
//   any run accepts --tg N: threads per threadgroup for the VARIANT (default 256 —
//   v10-class dim-split kernels dispatch 128).
//
// Shapes fixed to the Gemma-4 global layer: KVH=4, NQ=8, HD=512, L=61440, S=256.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

static const char * kKernelSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_math>
using namespace metal;
typedef bfloat bfloat16_t;

[[kernel]] void apertura_decode_attn_g(
    device const bfloat16_t * q        [[buffer(0)]],
    device const bfloat16_t * kraw     [[buffer(1)]],
    device const bfloat16_t * kw       [[buffer(2)]],
    device const float * invfreq       [[buffer(3)]],
    device const int * params          [[buffer(4)]],
    device const float * fparams      [[buffer(5)]],
    device float * out_acc             [[buffer(6)]],
    device float * out_m               [[buffer(7)]],
    device float * out_l               [[buffer(8)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {

    const int split  = threadgroup_position_in_grid.x;
    const int kvh    = threadgroup_position_in_grid.y;
    const int tid    = thread_position_in_threadgroup.x;
    const int sg     = tid / 32;
    const int lane   = tid % 32;
    const int d0     = lane * 16;

    const int L      = params[0];
    const int pitch  = params[1];
    const int S      = params[2];
    const float eps  = fparams[0];
    const float scale = fparams[1];

    const int rows   = (L + S - 1) / S;
    const int r0     = split * rows;
    const int r1     = min(r0 + rows, L);

    float qreg[16];
    for (int i = 0; i < 16; ++i) qreg[i] = (float) q[(kvh * 8 + sg) * 512 + d0 + i];

    threadgroup bfloat16_t tile[4 * 512];
    threadgroup bfloat16_t kT[4 * 512];
    threadgroup float csC[4 * 64];
    threadgroup float csS[4 * 64];
    threadgroup float rinvT[4];

    float m = -3.0e38f, l = 0.0f;
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

    for (int t0 = r0; t0 < r1; t0 += 4) {
        const int nt = min(4, r1 - t0);

        if (sg < 4 && sg < nt) {
            const device bfloat16_t * src = kraw + ((size_t) kvh * pitch + t0 + sg) * 512;
            float ss = 0.0f;
            for (int i = 0; i < 16; ++i) {
                bfloat16_t x = src[lane * 16 + i];
                tile[sg * 512 + lane * 16 + i] = x;
                float xf = (float) x; ss += xf * xf;
            }
            ss = simd_sum(ss);
            if (lane == 0) rinvT[sg] = rsqrt(ss / 512.0f + eps);
        }
        if (sg >= 4) {
            const int basei = (tid - 128) * 2;
            for (int u = 0; u < 2; ++u) {
                const int idx = basei + u;
                const int rr = idx / 64, ff = idx % 64;
                if (rr < nt) {
                    float a = (float) (t0 + rr) * invfreq[ff];
                    csC[idx] = (float) (bfloat16_t) precise::cos(a);
                    csS[idx] = (float) (bfloat16_t) precise::sin(a);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int u = 0; u < 8; ++u) {
            const int idx = tid * 8 + u;
            const int rr = idx / 512, d = idx % 512;
            if (rr >= nt) break;
            const float rinv = rinvT[rr];
            float kn = (float) tile[idx] * rinv * (float) kw[d];
            float kd;
            if (d < 64) {
                float kn2 = (float) tile[rr * 512 + d + 256] * rinv * (float) kw[d + 256];
                kd = kn * csC[rr * 64 + d] - kn2 * csS[rr * 64 + d];
            } else if (d >= 256 && d < 320) {
                float kn2 = (float) tile[rr * 512 + d - 256] * rinv * (float) kw[d - 256];
                kd = kn * csC[rr * 64 + d - 256] + kn2 * csS[rr * 64 + d - 256];
            } else {
                kd = kn;
            }
            kT[idx] = (bfloat16_t) kd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int rr = 0; rr < nt; ++rr) {
            const threadgroup bfloat16_t * krow = kT + rr * 512;
            const threadgroup bfloat16_t * vraw = tile + rr * 512;
            float dot = 0.0f;
            for (int i = 0; i < 16; ++i) dot += qreg[i] * (float) krow[d0 + i];
            float score = simd_sum(dot) * scale;
            float mNew = max(m, score);
            float corr = exp(m - mNew);
            float e = exp(score - mNew);
            l = l * corr + e;
            const float p = e * rinvT[rr];
            for (int i = 0; i < 16; ++i)
                acc[i] = acc[i] * corr + p * (float) vraw[d0 + i];
            m = mNew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const size_t hoff = ((size_t) (kvh * 8 + sg) * S + split);
    for (int i = 0; i < 16; ++i) out_acc[hoff * 512 + d0 + i] = acc[i];
    if (lane == 0) { out_m[hoff] = m; out_l[hoff] = l; }
}
)METAL";

static uint16_t f2bf(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    bits += 0x8000;                     // round to nearest (ties toward +)
    return (uint16_t) (bits >> 16);
}

int main(int argc, char ** argv) {
    const int KVH = 4, NQ = 8, HD = 512, L = 61440, S = 256, ROT = 64;
    const float EPS = 1e-6f, THETA = 1e6f;

    std::string mode = argc > 1 ? argv[1] : "";
    const bool doCapture = mode == "capture";
    const bool doCompile = mode == "compile";
    const bool doVerify  = mode == "verify";
    std::string srcPath;
    NSString * capturePath = @"harness.gputrace";
    int tgW = 256;
    {
        std::vector<std::string> pos;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--tg") && i + 1 < argc) { tgW = std::atoi(argv[++i]); continue; }
            pos.push_back(argv[i]);
        }
        if (doCapture)                    { if (pos.size() > 1) capturePath = @(pos[1].c_str());
                                            if (pos.size() > 2) srcPath = pos[2]; }
        else if (doCompile || doVerify)   { if (pos.size() > 1) srcPath = pos[1]; }
        else if (!pos.empty())            srcPath = pos[0];
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        auto makePipeline = [&](NSString * src, const char * label) -> id<MTLComputePipelineState> {
            NSError * err = nil;
            MTLCompileOptions * opts = [MTLCompileOptions new];
            opts.mathMode = MTLMathModeFast;   // MLX's custom kernels compile fast-math; match it
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
            if (!lib) { fprintf(stderr, "[%s] compile failed: %s\n", label,
                                err.localizedDescription.UTF8String); return nil; }
            id<MTLFunction> fn = [lib newFunctionWithName:@"apertura_decode_attn_g"];
            if (!fn) { fprintf(stderr, "[%s] missing function apertura_decode_attn_g\n", label); return nil; }
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!p) { fprintf(stderr, "[%s] pipeline failed: %s\n", label,
                              err.localizedDescription.UTF8String); return nil; }
            printf("[%s] maxTotalThreadsPerThreadgroup=%lu  staticThreadgroupMemory=%lu\n",
                   label, (unsigned long) p.maxTotalThreadsPerThreadgroup,
                   (unsigned long) p.staticThreadgroupMemoryLength);
            return p;
        };

        NSString * variantSrc = nil;
        if (!srcPath.empty()) {
            variantSrc = [NSString stringWithContentsOfFile:@(srcPath.c_str())
                                                   encoding:NSUTF8StringEncoding error:nil];
            if (!variantSrc) { fprintf(stderr, "cannot read %s\n", srcPath.c_str()); return 1; }
            printf("kernel source: %s (tg %d)\n", srcPath.c_str(), tgW);
        }

        if (doCompile) {
            return makePipeline(variantSrc ? variantSrc : @(kKernelSource), "compile") ? 0 : 1;
        }

        id<MTLComputePipelineState> psoBuiltin = nil, psoVariant = nil;
        if (doVerify || variantSrc == nil) psoBuiltin = makePipeline(@(kKernelSource), "v3");
        if (variantSrc) psoVariant = makePipeline(variantSrc, "variant");
        if ((doVerify && (!psoBuiltin || !psoVariant)) ||
            (!doVerify && !(variantSrc ? psoVariant : psoBuiltin))) return 1;

        // buffers
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto bfBuf = [&](size_t n, float scale) {
            id<MTLBuffer> b = [dev newBufferWithLength:n * 2 options:MTLResourceStorageModeShared];
            uint16_t * p = (uint16_t *) b.contents;
            for (size_t i = 0; i < n; ++i) p[i] = f2bf(nd(rng) * scale);
            return b;
        };
        id<MTLBuffer> q    = bfBuf((size_t) KVH * NQ * HD, 0.05f);
        id<MTLBuffer> kraw = bfBuf((size_t) KVH * L * HD, 1.0f);
        id<MTLBuffer> kw   = bfBuf(HD, 0.1f);   // ~N(0, 0.1); offset below
        { uint16_t * p = (uint16_t *) kw.contents; for (int i = 0; i < HD; ++i) {
            uint32_t u = ((uint32_t) p[i]) << 16; float f; memcpy(&f, &u, 4); p[i] = f2bf(1.0f + f); } }

        std::vector<float> invf(256, 0.0f);
        for (int i = 0; i < ROT; ++i)
            invf[i] = (float) (1.0 / std::pow((double) THETA, (double) (2 * i) / (double) HD));
        id<MTLBuffer> invfreq = [dev newBufferWithBytes:invf.data() length:invf.size() * 4
                                                options:MTLResourceStorageModeShared];
        int ip[3] = {L, L, S};
        float fp[2] = {EPS, 1.0f};
        id<MTLBuffer> params  = [dev newBufferWithBytes:ip length:sizeof(ip) options:MTLResourceStorageModeShared];
        id<MTLBuffer> fparams = [dev newBufferWithBytes:fp length:sizeof(fp) options:MTLResourceStorageModeShared];
        auto outSet = [&] {
            std::array<id<MTLBuffer>, 3> o;
            o[0] = [dev newBufferWithLength:(size_t) KVH * NQ * S * HD * 4 options:MTLResourceStorageModeShared];
            o[1] = [dev newBufferWithLength:(size_t) KVH * NQ * S * 4 options:MTLResourceStorageModeShared];
            o[2] = [dev newBufferWithLength:(size_t) KVH * NQ * S * 4 options:MTLResourceStorageModeShared];
            return o;
        };
        auto outA = outSet();

        auto dispatchN = [&](id<MTLComputePipelineState> pso, int tgw,
                             std::array<id<MTLBuffer>, 3> & outs, int n) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            for (int r = 0; r < n; ++r) {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                id<MTLBuffer> bufs[9] = {q, kraw, kw, invfreq, params, fparams, outs[0], outs[1], outs[2]};
                for (int i = 0; i < 9; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
                [enc dispatchThreads:MTLSizeMake((NSUInteger) tgw * S, KVH, 1)
               threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
                [enc endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
            return (double) (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / n;  // ms per dispatch
        };

        if (doVerify) {
            auto outB = outSet();
            dispatchN(psoBuiltin, 256, outA, 1);
            dispatchN(psoVariant, tgW, outB, 1);
            const size_t nml = (size_t) KVH * NQ * S, nacc = nml * HD;
            const float * mA = (const float *) outA[1].contents, * mB = (const float *) outB[1].contents;
            const float * lA = (const float *) outA[2].contents, * lB = (const float *) outB[2].contents;
            const float * aA = (const float *) outA[0].contents, * aB = (const float *) outB[0].contents;
            double mMax = 0, lMax = 0, aMax = 0; size_t bad = 0;
            for (size_t i = 0; i < nml; ++i) {
                if (!std::isfinite(mB[i]) || !std::isfinite(lB[i])) { bad++; continue; }
                mMax = std::max(mMax, (double) std::fabs(mA[i] - mB[i]));
                lMax = std::max(lMax, std::fabs(lA[i] - lB[i]) / (std::fabs(lA[i]) + 1e-6));
            }
            // acc is compared NORMALIZED (acc/l = the split's O partial, bounded by the
            // V magnitudes): v10-class kernels legitimately differ from v3 in rounding
            // scheme (v3 stages K through bf16 with rinv inside the rounding; the
            // register design keeps f32), so raw-partial relative error explodes
            // wherever the weighted sum cancels. The normalized bound is the honest
            // gate: bf16-rounding-class differences stay ~1e-2 absolute.
            size_t aArg = 0;
            for (size_t i = 0; i < nacc; ++i) {
                if (!std::isfinite(aB[i])) { bad++; continue; }
                const size_t row = i / HD;
                const double nA = aA[i] / ((double) lA[row] + 1e-20);
                const double nB = aB[i] / ((double) lB[row] + 1e-20);
                const double d = std::fabs(nA - nB);
                if (d > aMax) { aMax = d; aArg = i; }
            }
            printf("verify vs v3: max |dm| %.3e   max rel dl %.3e   max |d(acc/l)| %.3e   non-finite %zu\n",
                   mMax, lMax, aMax, bad);
            {
                size_t d = aArg % HD, sp = (aArg / HD) % S, h = (aArg / HD / S) % NQ, kv = aArg / HD / S / NQ;
                printf("  worst normalized acc at kvh %zu head %zu split %zu dim %zu: v3 %.6f variant %.6f\n",
                       kv, h, sp, d, aA[aArg], aB[aArg]);
            }
            // Bounds: m/l in the bf16-rounding class (v3 stages K through bf16; a
            // register variant's scores differ by that epsilon), normalized acc ~1e-2.
            const bool pass = bad == 0 && mMax < 3e-2 && lMax < 3e-2 && aMax < 2e-2;
            printf("verify: %s\n", pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }

        id<MTLComputePipelineState> pso = variantSrc ? psoVariant : psoBuiltin;
        const int tgw = variantSrc ? tgW : 256;
        dispatchN(pso, tgw, outA, 1); dispatchN(pso, tgw, outA, 1);   // warmup

        if (doCapture) {
            MTLCaptureManager * mgr = [MTLCaptureManager sharedCaptureManager];
            MTLCaptureDescriptor * desc = [MTLCaptureDescriptor new];
            desc.captureObject = dev;
            desc.destination = MTLCaptureDestinationGPUTraceDocument;
            desc.outputURL = [NSURL fileURLWithPath:capturePath];
            NSError * cerr = nil;
            if (![mgr startCaptureWithDescriptor:desc error:&cerr]) {
                fprintf(stderr, "capture failed: %s\n", cerr.localizedDescription.UTF8String);
                return 1;
            }
            dispatchN(pso, tgw, outA, 1);
            [mgr stopCapture];
            printf("captured -> %s\n", capturePath.UTF8String);
            return 0;
        }

        std::vector<double> ts;
        for (int i = 0; i < 5; ++i) ts.push_back(dispatchN(pso, tgw, outA, 10));
        std::sort(ts.begin(), ts.end());
        double med = ts[ts.size() / 2];
        double gb = (double) KVH * L * HD * 2 / 1e9;
        printf("kernel GPU time: median %.3f ms (min %.3f)  ->  %.0f GB/s (%.2f GB stream)\n",
               med, ts.front(), gb / (med / 1e3), gb);

        // sanity: no NaNs in the partials
        float * lp = (float *) outA[2].contents;
        int bad = 0;
        for (size_t i = 0; i < (size_t) KVH * NQ * S; ++i) if (!std::isfinite(lp[i])) bad++;
        printf("partials: %d non-finite l-values\n", bad);
    }
    return 0;
}
