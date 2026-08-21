#include "ESDecodeAttn.h"
#include "mlx/fast.h"

#include <cassert>
#include <cstdlib>
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

// v10-dimsplit (probe, env APERTURA_DECODE_V10=1): register-resident hot loop — the
// occupancy redesign from REGISTER_RESIDENT_DECODE.md. One 4-simdgroup threadgroup per
// (split, kvh) handles all 8 query heads; the row is striped mod-4 across simdgroups in
// 16-dim chunks (lane owns 4 dims), which makes every rope pair partner exactly
// simd_shuffle_xor(., 16). rinv is deferred to the score scalar (rope is linear), so
// the hot loop is registers + one 36-float cross-sg exchange per row (144 B static
// threadgroup memory vs v3's ~10 KB). Mirror of Tools/v10_dimsplit.metal — keep in sync.
static const char * kDecodeAttnV10Source = R"(
    const int split  = threadgroup_position_in_grid.x;
    const int kvh    = threadgroup_position_in_grid.y;
    const int tid    = thread_position_in_threadgroup.x;
    const int sg     = tid / 32;
    const int lane   = tid % 32;

    const int L      = params[0];
    const int pitch  = params[1];
    const int S      = params[2];
    const float eps  = fparams[0];
    const float scale = fparams[1];

    const int rows   = (L + S - 1) / S;
    const int r0     = split * rows;
    const int r1     = min(r0 + rows, L);

    const int chunk = sg + 4 * (lane / 4);
    const int base  = chunk * 16 + (lane % 4) * 4;
    const bool ropeLo = chunk < 4;
    const bool ropeHi = chunk >= 16 && chunk < 20;

    float kwreg[4];
    for (int j = 0; j < 4; ++j) kwreg[j] = (float) kw[base + j];
    float freq[4];
    for (int j = 0; j < 4; ++j) {
        int ff = ropeLo ? (base + j) : (ropeHi ? (base - 256 + j) : 0);
        freq[j] = invfreq[ff];
    }

    float qreg[8][4];
    for (int h = 0; h < 8; ++h)
        for (int j = 0; j < 4; ++j)
            qreg[h][j] = (float) q[(kvh * 8 + h) * 512 + base + j];

    float m[8], l[8], acc[8][4];
    for (int h = 0; h < 8; ++h) {
        m[h] = -3.0e38f; l[h] = 0.0f;
        for (int j = 0; j < 4; ++j) acc[h][j] = 0.0f;
    }

    threadgroup float stg[4][9];

    for (int t = r0; t < r1; ++t) {
        const device bfloat16_t * src = kraw + ((size_t) kvh * pitch + t) * 512 + base;
        float raw[4], rk[4];
        for (int j = 0; j < 4; ++j) {
            raw[j] = (float) src[j];
            rk[j]  = raw[j] * kwreg[j];
        }
        if (ropeLo || ropeHi) {
            float cs[4] = {0, 0, 0, 0}, sn[4] = {0, 0, 0, 0};
            if (ropeLo) {
                for (int j = 0; j < 4; ++j) {
                    float a = (float) t * freq[j];
                    cs[j] = (float) (bfloat16_t) precise::cos(a);
                    sn[j] = (float) (bfloat16_t) precise::sin(a);
                }
            }
            float pk0 = simd_shuffle_xor(rk[0], 16), pk1 = simd_shuffle_xor(rk[1], 16);
            float pk2 = simd_shuffle_xor(rk[2], 16), pk3 = simd_shuffle_xor(rk[3], 16);
            float pc0 = simd_shuffle_xor(cs[0], 16), pc1 = simd_shuffle_xor(cs[1], 16);
            float pc2 = simd_shuffle_xor(cs[2], 16), pc3 = simd_shuffle_xor(cs[3], 16);
            float ps0 = simd_shuffle_xor(sn[0], 16), ps1 = simd_shuffle_xor(sn[1], 16);
            float ps2 = simd_shuffle_xor(sn[2], 16), ps3 = simd_shuffle_xor(sn[3], 16);
            float pk[4] = {pk0, pk1, pk2, pk3};
            if (ropeLo) {
                for (int j = 0; j < 4; ++j) rk[j] = rk[j] * cs[j] - pk[j] * sn[j];
            } else {
                float pc[4] = {pc0, pc1, pc2, pc3};
                float ps[4] = {ps0, ps1, ps2, ps3};
                for (int j = 0; j < 4; ++j) rk[j] = rk[j] * pc[j] + pk[j] * ps[j];
            }
        } else {
            (void) simd_shuffle_xor(rk[0], 16);
        }

        float d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0, ss = 0;
        for (int j = 0; j < 4; ++j) {
            ss += raw[j] * raw[j];
            d0 += qreg[0][j] * rk[j]; d1 += qreg[1][j] * rk[j];
            d2 += qreg[2][j] * rk[j]; d3 += qreg[3][j] * rk[j];
            d4 += qreg[4][j] * rk[j]; d5 += qreg[5][j] * rk[j];
            d6 += qreg[6][j] * rk[j]; d7 += qreg[7][j] * rk[j];
        }
        d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
        d4 = simd_sum(d4); d5 = simd_sum(d5); d6 = simd_sum(d6); d7 = simd_sum(d7);
        ss = simd_sum(ss);
        if (lane < 9) {
            float v = lane == 0 ? d0 : lane == 1 ? d1 : lane == 2 ? d2 : lane == 3 ? d3
                    : lane == 4 ? d4 : lane == 5 ? d5 : lane == 6 ? d6 : lane == 7 ? d7 : ss;
            stg[sg][lane] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score[8], sstot = 0;
        for (int h = 0; h < 8; ++h) score[h] = stg[0][h] + stg[1][h] + stg[2][h] + stg[3][h];
        sstot = stg[0][8] + stg[1][8] + stg[2][8] + stg[3][8];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float rinv = rsqrt(sstot / 512.0f + eps);
        for (int h = 0; h < 8; ++h) {
            const float sc = score[h] * rinv * scale;
            const float mNew = max(m[h], sc);
            const float corr = exp(m[h] - mNew);
            const float e = exp(sc - mNew);
            l[h] = l[h] * corr + e;
            const float p = e * rinv;
            for (int j = 0; j < 4; ++j)
                acc[h][j] = acc[h][j] * corr + p * raw[j];
            m[h] = mNew;
        }
    }

    for (int h = 0; h < 8; ++h) {
        const size_t hoff = ((size_t) (kvh * 8 + h) * S + split);
        for (int j = 0; j < 4; ++j) out_acc[hoff * 512 + base + j] = acc[h][j];
        if (sg == 0 && lane == 0) { out_m[hoff] = m[h]; out_l[hoff] = l[h]; }
    }
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

// q8 variant of the v3 kernel: identical structure, but tile rows are affine u8 —
// dequant (x = q * scale + bias, per-row scale/bias staged alongside rinv) happens at
// every use: staging rms, K reconstruction, V accumulation. 512 B + 8 B per row from
// device instead of 1 KB.
static const char * kDecodeAttnQ8Source = R"(
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

    threadgroup uchar tile[4 * 512];
    threadgroup bfloat16_t kT[4 * 512];
    threadgroup float csC[4 * 64];
    threadgroup float csS[4 * 64];
    threadgroup float rinvT[4];
    threadgroup float scT[4];
    threadgroup float bsT[4];

    float m = -3.0e38f, l = 0.0f;
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

    for (int t0 = r0; t0 < r1; t0 += 4) {
        const int nt = min(4, r1 - t0);

        if (sg < 4 && sg < nt) {
            const size_t row = (size_t) kvh * pitch + t0 + sg;
            const device uchar * src = kq + row * 512;
            const float sc = ksc[row];
            const float bs = kbs[row];
            float ss = 0.0f;
            for (int i = 0; i < 16; ++i) {
                uchar x = src[lane * 16 + i];
                tile[sg * 512 + lane * 16 + i] = x;
                float xf = (float) x * sc + bs; ss += xf * xf;
            }
            ss = simd_sum(ss);
            if (lane == 0) {
                rinvT[sg] = rsqrt(ss / 512.0f + eps);
                scT[sg] = sc; bsT[sg] = bs;
            }
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
            const float sc = scT[rr], bs = bsT[rr];
            float kn = ((float) tile[idx] * sc + bs) * rinv * (float) kw[d];
            float kd;
            if (d < 64) {
                float kn2 = ((float) tile[rr * 512 + d + 256] * sc + bs) * rinv * (float) kw[d + 256];
                kd = kn * csC[rr * 64 + d] - kn2 * csS[rr * 64 + d];
            } else if (d >= 256 && d < 320) {
                float kn2 = ((float) tile[rr * 512 + d - 256] * sc + bs) * rinv * (float) kw[d - 256];
                kd = kn * csC[rr * 64 + d - 256] + kn2 * csS[rr * 64 + d - 256];
            } else {
                kd = kn;
            }
            kT[idx] = (bfloat16_t) kd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int rr = 0; rr < nt; ++rr) {
            const threadgroup bfloat16_t * krow = kT + rr * 512;
            const threadgroup uchar * vraw = tile + rr * 512;
            const float sc = scT[rr], bs = bsT[rr];
            float dot = 0.0f;
            for (int i = 0; i < 16; ++i) dot += qreg[i] * (float) krow[d0 + i];
            float score = simd_sum(dot) * scale;
            float mNew = max(m, score);
            float corr = exp(m - mNew);
            float e = exp(score - mNew);
            l = l * corr + e;
            const float p = e * rinvT[rr];
            for (int i = 0; i < 16; ++i)
                acc[i] = acc[i] * corr + p * ((float) vraw[d0 + i] * sc + bs);
            m = mNew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const size_t hoff = ((size_t) (kvh * 8 + sg) * S + split);
    for (int i = 0; i < 16; ++i) out_acc[hoff * 512 + d0 + i] = acc[i];
    if (lane == 0) { out_m[hoff] = m; out_l[hoff] = l; }
)";

// Shared pass 2: fuse the flash-decoding split merge into one dispatch (see
// kDecodeCombineSource).
static mx::array combineSplits(const std::vector<mx::array> & outs, int numQ, int hd, int S,
                               mx::Dtype dtype) {
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
        {dtype},
        std::make_tuple(256 * numQ, 1, 1),
        std::make_tuple(256, 1, 1),
        {{"T", dtype}}, std::nullopt, false, {});
    return merged[0];
}

mx::array esDecodeAttnGlobal(const mx::array & q, const mx::array & rawBuf, int len, int pitch,
                             const mx::array & kw, float eps, const std::vector<float> & invFreq) {
    const int numQ = q.shape(0), hd = q.shape(1);
    const int numKV = rawBuf.shape(0);
    if (hd != 512 || numQ != numKV * 8 || rawBuf.shape(2) != 512)
        throw std::invalid_argument("esDecodeAttnGlobal: expects headDim 512 with 8 query heads per KV head");
    if ((int) invFreq.size() < 64)
        throw std::invalid_argument("esDecodeAttnGlobal: invFreq table too short");

    // v10 probe toggle: APERTURA_DECODE_V10=1 selects the register-resident dim-split
    // kernel (4 simdgroups / 128 threads per threadgroup) — in-model arms for the
    // occupancy redesign, gated standalone by Tools/gpuharness verify.
    static const bool useV10 = std::getenv("APERTURA_DECODE_V10") != nullptr;
    static auto kernel = mx::fast::metal_kernel(
        useV10 ? "apertura_decode_attn_g_v10" : "apertura_decode_attn_g",
        {"q", "kraw", "kw", "invfreq", "params", "fparams"},
        {"out_acc", "out_m", "out_l"},
        useV10 ? kDecodeAttnV10Source : kDecodeAttnSource,
        "#include <metal_simdgroup>\n#include <metal_math>\nusing namespace metal;\n",
        /*ensure_row_contiguous=*/false);
    const int tgw = useV10 ? 128 : 256;

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
        std::make_tuple(tgw * S, numKV, 1),
        std::make_tuple(tgw, 1, 1),
        {}, std::nullopt, false, {});

    return combineSplits(outs, numQ, hd, S, q.dtype());
}

mx::array esDecodeAttnGlobalQ8(const mx::array & q, const mx::array & qBuf,
                               const mx::array & scBuf, const mx::array & bsBuf,
                               int len, int pitch, const mx::array & kw, float eps,
                               const std::vector<float> & invFreq) {
    const int numQ = q.shape(0), hd = q.shape(1);
    const int numKV = qBuf.shape(0);
    if (hd != 512 || numQ != numKV * 8 || qBuf.shape(2) != 512)
        throw std::invalid_argument("esDecodeAttnGlobalQ8: expects headDim 512 with 8 query heads per KV head");
    if (qBuf.dtype() != mx::uint8)
        throw std::invalid_argument("esDecodeAttnGlobalQ8: qBuf must be uint8");
    if ((int) invFreq.size() < 64)
        throw std::invalid_argument("esDecodeAttnGlobalQ8: invFreq table too short");

    static auto kernel = mx::fast::metal_kernel(
        "apertura_decode_attn_g_q8",
        {"q", "kq", "ksc", "kbs", "kw", "invfreq", "params", "fparams"},
        {"out_acc", "out_m", "out_l"},
        kDecodeAttnQ8Source,
        "#include <metal_simdgroup>\n#include <metal_math>\nusing namespace metal;\n",
        /*ensure_row_contiguous=*/false);

    const int S = 256;   // same split count as the bf16 kernel (see its comment)
    std::vector<int> ip = {len, pitch, S};
    std::vector<float> fp = {eps, 1.0f};   // scaling 1.0 — QK-norm absorbs 1/sqrt(d)
    mx::array params(ip.data(), {3}, mx::int32);
    mx::array fparams(fp.data(), {2}, mx::float32);
    mx::array invf(invFreq.data(), {(int) invFreq.size()}, mx::float32);

    auto outs = kernel(
        {q, qBuf, scBuf, bsBuf, kw, invf, params, fparams},
        {{numKV, 8, S, hd}, {numKV, 8, S}, {numKV, 8, S}},
        {mx::float32, mx::float32, mx::float32},
        std::make_tuple(256 * S, numKV, 1),
        std::make_tuple(256, 1, 1),
        {}, std::nullopt, false, {});

    return combineSplits(outs, numQ, hd, S, q.dtype());
}

// One simdgroup per row: lane holds 16 elements, simd_min/simd_max give the row range,
// then pack. round() is half-away-from-zero, matching mx::round in the ops reference.
static const char * kQuantizeRawSource = R"(
    const int row  = threadgroup_position_in_grid.x;
    const int kvh  = threadgroup_position_in_grid.y;
    const int lane = thread_position_in_threadgroup.x;
    const int n    = params[0];

    const size_t off = ((size_t) kvh * n + row) * 512;
    float x[16];
    float mn = 3.0e38f, mxv = -3.0e38f;
    for (int i = 0; i < 16; ++i) {
        x[i] = (float) src[off + lane * 16 + i];
        mn = min(mn, x[i]); mxv = max(mxv, x[i]);
    }
    mn = simd_min(mn); mxv = simd_max(mxv);
    const float sc = max((mxv - mn) / 255.0f, 1e-8f);
    for (int i = 0; i < 16; ++i)
        outq[off + lane * 16 + i] = (uchar) min(round((x[i] - mn) / sc), 255.0f);
    if (lane == 0) { outs[(size_t) kvh * n + row] = sc; outb[(size_t) kvh * n + row] = mn; }
)";

std::array<mx::array, 3> esQuantizeRawRows(const mx::array & kNew) {
    const int kvH = kNew.shape(0), n = kNew.shape(1), hd = kNew.shape(2);
    if (hd != 512)
        throw std::invalid_argument("esQuantizeRawRows: expects headDim 512");

    static auto kernel = mx::fast::metal_kernel(
        "apertura_quantize_raw_rows",
        {"src", "params"},
        {"outq", "outs", "outb"},
        kQuantizeRawSource,
        "#include <metal_simdgroup>\n#include <metal_math>\nusing namespace metal;\n",
        /*ensure_row_contiguous=*/true);   // kNew is a transpose view — copy it packed

    std::vector<int> ip = {n};
    mx::array params(ip.data(), {1}, mx::int32);
    auto outs = kernel(
        {kNew, params},
        {{kvH, n, hd}, {kvH, n, 1}, {kvH, n, 1}},
        {mx::uint8, mx::float32, mx::float32},
        std::make_tuple(32 * n, kvH, 1),
        std::make_tuple(32, 1, 1),
        {}, std::nullopt, false, {});
    return {outs[0], outs[1], outs[2]};
}

}  // namespace es
