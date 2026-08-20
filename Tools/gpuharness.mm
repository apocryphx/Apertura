// gpuharness — standalone Metal harness for the Apertura decode-attention kernel.
// No MLX: compiles the kernel source directly, so MTLCaptureManager captures the real
// dispatch and `gpudebug` can replay + profile it headlessly (counters!).
//
//   ./gpuharness                 — time the kernel (GPU timestamps, median of 10)
//   ./gpuharness capture out.gputrace — capture one dispatch for gpudebug
//   ./gpuharness kernel.metal    — use an alternate kernel source file (tuning loop)
//
// Shapes fixed to the Gemma-4 global layer: KVH=4, NQ=8, HD=512, L=61440, S=256.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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

    bool doCapture = argc > 1 && std::strcmp(argv[1], "capture") == 0;
    NSString * capturePath = doCapture && argc > 2 ? @(argv[2]) : @"harness.gputrace";
    std::string srcPath = doCapture ? (argc > 3 ? argv[3] : "") : (argc > 1 ? argv[1] : "");

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        NSString * src;
        if (!srcPath.empty()) {
            src = [NSString stringWithContentsOfFile:@(srcPath.c_str())
                                            encoding:NSUTF8StringEncoding error:nil];
            if (!src) { fprintf(stderr, "cannot read %s\n", srcPath.c_str()); return 1; }
            printf("kernel source: %s\n", srcPath.c_str());
        } else {
            src = @(kKernelSource);
        }

        NSError * err = nil;
        MTLCompileOptions * opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeFast;   // MLX's custom kernels compile fast-math; match it
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"apertura_decode_attn_g"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("pipeline: maxTotalThreadsPerThreadgroup=%lu  staticThreadgroupMemory=%lu\n",
               (unsigned long) pso.maxTotalThreadsPerThreadgroup,
               (unsigned long) pso.staticThreadgroupMemoryLength);

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
        id<MTLBuffer> outAcc = [dev newBufferWithLength:(size_t) KVH * NQ * S * HD * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> outM   = [dev newBufferWithLength:(size_t) KVH * NQ * S * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> outL   = [dev newBufferWithLength:(size_t) KVH * NQ * S * 4 options:MTLResourceStorageModeShared];

        auto dispatchN = [&](int n) {   // n back-to-back dispatches in ONE command buffer
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            for (int r = 0; r < n; ++r) {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                id<MTLBuffer> bufs[9] = {q, kraw, kw, invfreq, params, fparams, outAcc, outM, outL};
                for (int i = 0; i < 9; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
                [enc dispatchThreads:MTLSizeMake(256 * S, KVH, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
            return (double) (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / n;  // ms per dispatch
        };
        auto dispatch = [&] { return dispatchN(1); };

        // warmup
        dispatch(); dispatch();

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
            dispatch();
            [mgr stopCapture];
            printf("captured -> %s\n", capturePath.UTF8String);
            return 0;
        }

        std::vector<double> ts;
        for (int i = 0; i < 5; ++i) ts.push_back(dispatchN(10));
        std::sort(ts.begin(), ts.end());
        double med = ts[ts.size() / 2];
        double gb = (double) KVH * L * HD * 2 / 1e9;
        printf("kernel GPU time: median %.3f ms (min %.3f)  ->  %.0f GB/s (%.2f GB stream)\n",
               med, ts.front(), gb / (med / 1e3), gb);

        // sanity: no NaNs in the partials
        float * lp = (float *) outL.contents;
        int bad = 0;
        for (size_t i = 0; i < (size_t) KVH * NQ * S; ++i) if (!std::isfinite(lp[i])) bad++;
        printf("partials: %d non-finite l-values\n", bad);
    }
    return 0;
}
