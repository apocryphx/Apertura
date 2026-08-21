// v10d-dimsplit — v10c + 4-row tiles (v10c processed ONE row per barrier pair:
// 240 serialized load->sync->compute epochs per split with tiny work between
// syncs — latency-bound at 59 GB/s. Tiling restores 4-wide memory-level
// parallelism and cuts barriers 4x while keeping the exchange ~1.2 KB.)
// Retains from v10c: incremental-rotation cos/sin (no transcendental calls
// in the hot loop: v10b still spilled 96 B/thread because precise::cos/sin are
// CALLS whose clobbers force the live q/acc arrays to spill around them — the
// reason v3 computed its table in dedicated simdgroups. Here cos/sin advance by
// a per-lane 4-FMA complex rotation, initialized once per split; drift over 240
// rows is ~1e-5, far inside the bf16 rounding class the verify gate allows).
// (v10a at 128 threads spilled 176 B/thread — 4-dim lanes need ~110 registers —
// and the spill traffic re-tripped the occupancy manager: 23% occupancy,
// 78 GB/s of WRITE bandwidth. Here 64 chunks of 8 dims stripe mod-8 across
// 8 simdgroups; a lane owns 2 dims, ~60 registers, no spills. Chunk c pairs
// with c+32 = slot i+4 = 16 lanes away, so the xor-16 rope trick survives.)
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
    const int chunk = sg + 8 * (lane / 4);
    const int base  = chunk * 8 + (lane % 4) * 2;
    const bool ropeLo = chunk < 8;                  // dims 0..63   (lanes 0..3)
    const bool ropeHi = chunk >= 32 && chunk < 40;  // dims 256..319 (lanes 16..19)

    // Per-lane constants: kw slice and the rope frequencies for this lane's dims.
    float kwreg[2];
    for (int j = 0; j < 2; ++j) kwreg[j] = (float) kw[base + j];
    float freq[2];
    for (int j = 0; j < 2; ++j) {
        int ff = ropeLo ? (base + j) : (ropeHi ? (base - 256 + j) : 0);
        freq[j] = invfreq[ff];
    }

    // Query slices: 8 heads x 4 dims.
    // Incremental rotation state: (c, s) = (cos, sin)(t * freq), advanced by the
    // constant per-row rotation (cF, sF) = (cos, sin)(freq). Non-rope lanes idle at
    // the identity rotation — uniform code, no divergence in the maintenance.
    float cF[2], sF[2], cR[2], sR[2];
    for (int j = 0; j < 2; ++j) {
        if (ropeLo || ropeHi) {
            cF[j] = precise::cos(freq[j]);
            sF[j] = precise::sin(freq[j]);
            float a0 = (float) r0 * freq[j];
            cR[j] = precise::cos(a0);
            sR[j] = precise::sin(a0);
        } else { cF[j] = 1.0f; sF[j] = 0.0f; cR[j] = 1.0f; sR[j] = 0.0f; }
    }

    float qreg[8][2];
    for (int h = 0; h < 8; ++h)
        for (int j = 0; j < 2; ++j)
            qreg[h][j] = (float) q[(kvh * 8 + h) * 512 + base + j];

    float m[8], l[8], acc[8][2];
    for (int h = 0; h < 8; ++h) {
        m[h] = -3.0e38f; l[h] = 0.0f;
        for (int j = 0; j < 2; ++j) acc[h][j] = 0.0f;
    }

    // Cross-sg exchange: [sg][8 dots + sumsq]. 144 bytes total — the ONLY threadgroup
    // memory in the kernel.
    threadgroup float stg[8][4][9];

    for (int t0 = r0; t0 < r1; t0 += 4) {
        const int nt = min(4, r1 - t0);

        // ---- load nt rows (independent -> memory-level parallelism), rope in regs ----
        float raw[4][2], rk[4][2];
        for (int r = 0; r < 4; ++r) {
            if (r < nt) {
                const device bfloat16_t * src = kraw + ((size_t) kvh * pitch + (t0 + r)) * 512 + base;
                raw[r][0] = (float) src[0];
                raw[r][1] = (float) src[1];
            } else { raw[r][0] = 0.0f; raw[r][1] = 0.0f; }
        }
        for (int r = 0; r < 4; ++r) {
            float a0 = raw[r][0] * kwreg[0], a1 = raw[r][1] * kwreg[1];
            float cs0 = cR[0], cs1 = cR[1], sn0 = sR[0], sn1 = sR[1];
            float pk0 = simd_shuffle_xor(a0, 16), pk1 = simd_shuffle_xor(a1, 16);
            float pc0 = simd_shuffle_xor(cs0, 16), pc1 = simd_shuffle_xor(cs1, 16);
            float ps0 = simd_shuffle_xor(sn0, 16), ps1 = simd_shuffle_xor(sn1, 16);
            if (ropeLo) {
                rk[r][0] = a0 * cs0 - pk0 * sn0;
                rk[r][1] = a1 * cs1 - pk1 * sn1;
            } else if (ropeHi) {
                rk[r][0] = a0 * pc0 + pk0 * ps0;
                rk[r][1] = a1 * pc1 + pk1 * ps1;
            } else {
                rk[r][0] = a0; rk[r][1] = a1;
            }
            // advance the rotation to the next row (identity for non-rope lanes)
            const float c2 = cR[0] * cF[0] - sR[0] * sF[0];
            sR[0] = sR[0] * cF[0] + cR[0] * sF[0]; cR[0] = c2;
            const float c3 = cR[1] * cF[1] - sR[1] * sF[1];
            sR[1] = sR[1] * cF[1] + cR[1] * sF[1]; cR[1] = c3;
        }

        // ---- per-sg partials for all nt rows, one exchange ----
        for (int r = 0; r < nt; ++r) {
            float d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0, ss = 0;
            ss += raw[r][0] * raw[r][0] + raw[r][1] * raw[r][1];
            d0 += qreg[0][0] * rk[r][0] + qreg[0][1] * rk[r][1];
            d1 += qreg[1][0] * rk[r][0] + qreg[1][1] * rk[r][1];
            d2 += qreg[2][0] * rk[r][0] + qreg[2][1] * rk[r][1];
            d3 += qreg[3][0] * rk[r][0] + qreg[3][1] * rk[r][1];
            d4 += qreg[4][0] * rk[r][0] + qreg[4][1] * rk[r][1];
            d5 += qreg[5][0] * rk[r][0] + qreg[5][1] * rk[r][1];
            d6 += qreg[6][0] * rk[r][0] + qreg[6][1] * rk[r][1];
            d7 += qreg[7][0] * rk[r][0] + qreg[7][1] * rk[r][1];
            d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
            d4 = simd_sum(d4); d5 = simd_sum(d5); d6 = simd_sum(d6); d7 = simd_sum(d7);
            ss = simd_sum(ss);
            if (lane < 9) {
                float v = lane == 0 ? d0 : lane == 1 ? d1 : lane == 2 ? d2 : lane == 3 ? d3
                        : lane == 4 ? d4 : lane == 5 ? d5 : lane == 6 ? d6 : lane == 7 ? d7 : ss;
                stg[sg][r][lane] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- combine + online softmax per row (registers; identical order to v3) ----
        for (int r = 0; r < nt; ++r) {
            float score[8], sstot = 0;
            for (int h = 0; h < 8; ++h) {
                float sh = 0;
                for (int g = 0; g < 8; ++g) sh += stg[g][r][h];
                score[h] = sh;
            }
            for (int g = 0; g < 8; ++g) sstot += stg[g][r][8];
            const float rinv = rsqrt(sstot / 512.0f + eps);
            for (int h = 0; h < 8; ++h) {
                const float sc = score[h] * rinv * scale;
                const float mNew = max(m[h], sc);
                const float corr = exp(m[h] - mNew);
                const float e = exp(sc - mNew);
                l[h] = l[h] * corr + e;
                const float p = e * rinv;
                acc[h][0] = acc[h][0] * corr + p * raw[r][0];
                acc[h][1] = acc[h][1] * corr + p * raw[r][1];
                m[h] = mNew;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int h = 0; h < 8; ++h) {
        const size_t hoff = ((size_t) (kvh * 8 + h) * S + split);
        for (int j = 0; j < 2; ++j) out_acc[hoff * 512 + base + j] = acc[h][j];
        if (sg == 0 && lane == 0) { out_m[hoff] = m[h]; out_l[hoff] = l[h]; }
    }
}
