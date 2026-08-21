// v10-dimsplit — register-resident decode attention (REGISTER_RESIDENT_DECODE.md).
//
// One threadgroup = one (split, kvh) = 4 simdgroups = one row team, ALL 8 query heads.
// The 512-dim row is split into 32 chunks of 16 dims, striped mod-4 across simdgroups
// (sg s owns chunks {s, s+4, ..., s+28}); each lane owns 4 dims of one chunk. Striping
// makes every RoPE pair partner exactly simd_shuffle_xor(., 16): chunk c pairs with
// c+16, which is 4 chunk-slots (= 16 lanes) away in the same simdgroup.
//
// The hot loop is register-only:
//   load 4 bf16 -> rkw = raw*kw -> shuffle-rope (rinv-free: rope is LINEAR, so
//   K = rinv * rope(raw*kw) and rinv defers to the score scalar) -> 8 dot partials.
// Cross-simdgroup traffic per row is ONE 36-float exchange (4 sgs x [8 dots + sumsq])
// through threadgroup memory — ~64x less staging traffic than v3's 2 KB row tiles,
// which is exactly what the occupancy manager was throttling. Output dims are owned
// per simdgroup, so the epilogue writes 128-dim slices with no cross-sg combine.
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
    const int sg     = tid / 32;          // 0..3
    const int lane   = tid % 32;

    const int L      = params[0];
    const int pitch  = params[1];
    const int S      = params[2];
    const float eps  = fparams[0];
    const float scale = fparams[1];

    const int rows   = (L + S - 1) / S;
    const int r0     = split * rows;
    const int r1     = min(r0 + rows, L);

    // Lane -> dims: chunk slot i = lane/4 (8 chunks per sg), chunk id c = sg + 4*i,
    // dim base = c*16 + (lane%4)*4, covering dims [base, base+4).
    const int chunk = sg + 4 * (lane / 4);
    const int base  = chunk * 16 + (lane % 4) * 4;
    const bool ropeLo = chunk < 4;              // dims 0..63   (lanes 0..3 of each sg)
    const bool ropeHi = chunk >= 16 && chunk < 20;  // dims 256..319 (lanes 16..19)

    // Per-lane constants: kw slice and the rope frequencies for this lane's dims.
    float kwreg[4];
    for (int j = 0; j < 4; ++j) kwreg[j] = (float) kw[base + j];
    float freq[4];
    for (int j = 0; j < 4; ++j) {
        int ff = ropeLo ? (base + j) : (ropeHi ? (base - 256 + j) : 0);
        freq[j] = invfreq[ff];
    }

    // Query slices: 8 heads x 4 dims.
    float qreg[8][4];
    for (int h = 0; h < 8; ++h)
        for (int j = 0; j < 4; ++j)
            qreg[h][j] = (float) q[(kvh * 8 + h) * 512 + base + j];

    float m[8], l[8], acc[8][4];
    for (int h = 0; h < 8; ++h) {
        m[h] = -3.0e38f; l[h] = 0.0f;
        for (int j = 0; j < 4; ++j) acc[h][j] = 0.0f;
    }

    // Cross-sg exchange: [sg][8 dots + sumsq]. 144 bytes total — the ONLY threadgroup
    // memory in the kernel.
    threadgroup float stg[4][9];

    for (int t = r0; t < r1; ++t) {
        // ---- load + rkw + rope, all registers ----
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
            // Partner values and (for the hi half) the lo half's cos/sin via xor-16.
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
            // Uniform shuffles keep the simdgroup convergent; results unused.
            (void) simd_shuffle_xor(rk[0], 16);
        }

        // ---- per-sg partials: 8 dots + sumsq ----
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
            const float p = e * rinv;                 // V = raw * rinv
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
}
