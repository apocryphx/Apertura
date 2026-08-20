#include "ESDecodeAttn.h"
#include "mlx/fast.h"

#include <cassert>
#include <stdexcept>
#include <tuple>

namespace es {

// Kernel v3 of the 2026-08-20 probe session (see the scratchpad decode_kernel_probe arc:
// v2 tg-sharing 25 GB/s -> v3 114 GB/s; batched updates, vector loads, f32 staging,
// larger tiles, and software prefetch all REGRESSED — this simple form is the champion).
// Per 4-row tile: A: simdgroups 0..3 stage one row each + its rms, simdgroups 4..7 build
// the 4x64 cos/sin table; B: reconstruct K once, cooperatively; C: one simdgroup per
// query head — dot, online softmax, accumulate V from the raw tile.
static const char * kDecodeAttnSource = R"(
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
)";

// Pass 2, fused: the online-softmax merge of the split partials. One threadgroup per
// query head reads acc [numQ, S, 512] + m/l [numQ, S] exactly once and writes O — the
// op-level version streamed the 512-wide partials through ~5 separate MLX kernels per
// layer per token, which cost more than the main kernel's half-bytes win (measured
// 2026-08-20: raw path +2.6 ms/token of epilogue against a -1.9 ms/token kernel).
// Empty tail splits carry m = -3e38, l = 0: their weight underflows to exactly 0.
static const char * kDecodeCombineSource = R"(
    const int h   = threadgroup_position_in_grid.x;   // query head
    const int tid = thread_position_in_threadgroup.x; // 0..255
    const int S   = params[0];

    threadgroup float wS[256];   // S <= 256 (S is 128; the static assert below guards growth)

    float mMax = -3.0e38f;
    for (int s = 0; s < S; ++s) mMax = max(mMax, m[h * S + s]);
    for (int s = tid; s < S; s += 256) wS[s] = exp(m[h * S + s] - mMax);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lTot = 0.0f;
    for (int s = 0; s < S; ++s) lTot += l[h * S + s] * wS[s];

    for (int d = tid; d < 512; d += 256) {
        float a = 0.0f;
        for (int s = 0; s < S; ++s) a += acc[((size_t) h * S + s) * 512 + d] * wS[s];
        out[h * 512 + d] = (T) (a / lTot);
    }
)";

mx::array esDecodeAttnGlobal(const mx::array & q, const mx::array & rawBuf, int len, int pitch,
                             const mx::array & kw, float eps, const std::vector<float> & invFreq) {
    const int numQ = q.shape(0), hd = q.shape(1);
    const int numKV = rawBuf.shape(0);
    if (hd != 512 || numQ != numKV * 8 || rawBuf.shape(2) != 512)
        throw std::invalid_argument("esDecodeAttnGlobal: expects headDim 512 with 8 query heads per KV head");
    if ((int) invFreq.size() < 64)
        throw std::invalid_argument("esDecodeAttnGlobal: invFreq table too short");

    static auto kernel = mx::fast::metal_kernel(
        "apertura_decode_attn_g",
        {"q", "kraw", "kw", "invfreq", "params", "fparams"},
        {"out_acc", "out_m", "out_l"},
        kDecodeAttnSource,
        "#include <metal_simdgroup>\n#include <metal_math>\nusing namespace metal;\n",
        /*ensure_row_contiguous=*/false);

    const int S = 256;   // in-model needs 256 (see doc); keep <= 256: combine's wS staging
    std::vector<int> ip = {len, pitch, S};
    std::vector<float> fp = {eps, 1.0f};   // scaling 1.0 — QK-norm absorbs 1/sqrt(d)
    mx::array params(ip.data(), {3}, mx::int32);
    mx::array fparams(fp.data(), {2}, mx::float32);
    mx::array invf(invFreq.data(), {(int) invFreq.size()}, mx::float32);

    auto outs = kernel(
        {q, rawBuf, kw, invf, params, fparams},
        {{numKV, 8, S, hd}, {numKV, 8, S}, {numKV, 8, S}},
        {mx::float32, mx::float32, mx::float32},
        std::make_tuple(256 * S, numKV, 1),
        std::make_tuple(256, 1, 1),
        {}, std::nullopt, false, {});

    // pass 2: fused split combine (see kDecodeCombineSource) — one dispatch, partials
    // streamed exactly once, output written directly in q's dtype.
    static auto combine = mx::fast::metal_kernel(
        "apertura_decode_attn_combine",
        {"acc", "m", "l", "params"},
        {"out"},
        kDecodeCombineSource,
        "#include <metal_math>\nusing namespace metal;\n",
        /*ensure_row_contiguous=*/false);

    std::vector<int> cp = {S};
    mx::array cparams(cp.data(), {1}, mx::int32);
    auto merged = combine(
        {outs[0], outs[1], outs[2], cparams},
        {{numQ, hd}},
        {q.dtype()},
        std::make_tuple(256 * numQ, 1, 1),
        std::make_tuple(256, 1, 1),
        {{"T", q.dtype()}}, std::nullopt, false, {});
    return merged[0];
}

}  // namespace es
