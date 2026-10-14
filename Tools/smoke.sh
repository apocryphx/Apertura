#!/bin/bash
# smoke.sh — one-command health check of this machine's Apertura setup (~1-2 min on E2B).
#   Tools/smoke.sh [model.apml]           default: the E2B bundle on the "Gemma 4" volume
#   APERTURA_SMOKE_DD=<dir>               DerivedData location (default: $TMPDIR/apertura-smoke)
#
# Checks, in order (each PASS / FAIL / SKIP; exit status 1 if anything FAILs):
#   1. toolchain   Metal compiler present (without it MLX silently builds no kernels / no NAX)
#   2. mlx         ../mlx at the Tools/mlx.pin commit, libmlx.a built, NAX kernels compiled in
#   3. build       AperturaResearch (Release, arm64)
#   4. kit-tests   AperturaKit XCTests with APERTURAKIT_TEST_MODEL = the model (end-to-end
#                  session + checkpoint round trip included)
#   5. persist     --persist-verify: snapshot restore byte-identical to a fresh prime
#   6. fixtures    PyTorch fixture conformance, if fixtures_e2b_qat + its QAT source exist
#   7. bench       warm prefill 2048 + decode 64 (reported, not gated; runs right after the
#                  other steps without a cool-down, so expect a few % below README numbers)
# Written 2026-10-13: the E2B/E4B checkpoint bug sat unnoticed because the model tests skip
# without APERTURAKIT_TEST_MODEL — this script always sets it.

set -u
cd "$(dirname "$0")/.."                      # repo root
ROOT="$(pwd)"
MLX="$(cd .. && pwd)/mlx"
MODELS="/Volumes/Gemma 4"                  # model volume, same layout on every machine (SYNC.md)
MODEL="${1:-$MODELS/gemma-4-E2B-it-qat-q4.apml}"
DD="${APERTURA_SMOKE_DD:-${TMPDIR:-/tmp}/apertura-smoke}"
LOG="$DD/logs"; mkdir -p "$LOG"
QAT_SRC="$MODELS/sources/gemma-4-E2B-it-qat-q4_0-unquantized"
FIXTURE="$ROOT/aptransformerTests/Fixtures/fixtures_e2b_qat.safetensors"

RESULTS=(); FAILED=0
record() {  # name status detail
    RESULTS+=("$(printf '%-10s %-5s %s' "$1" "$2" "$3")")
    [ "$2" = FAIL ] && FAILED=1
    printf '%-10s %-5s %s\n' "$1" "$2" "$3"
}

echo "== Apertura smoke — $(sysctl -n machdep.cpu.brand_string), macOS $(sw_vers -productVersion)"
echo "   model: $MODEL"
echo "   logs : $LOG"

# 1. toolchain
if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    record toolchain PASS "$(xcrun -sdk macosx metal --version 2>&1 | head -1)"
else
    record toolchain FAIL "Metal Toolchain missing: xcodebuild -downloadComponent MetalToolchain"
fi

# 2. mlx
PIN="$(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' Tools/mlx.pin | head -1 | tr -d '[:space:]')"
if [ ! -f "$MLX/build/libmlx.a" ]; then
    record mlx FAIL "no $MLX/build/libmlx.a — run: sh Tools/build_mlx.sh"
elif [ "$(git -C "$MLX" rev-parse HEAD 2>/dev/null)" != "$PIN" ]; then
    record mlx FAIL "../mlx is not at the pin ${PIN:0:9} — run: sh Tools/build_mlx.sh"
else
    NAX=$(nm "$MLX/build/libmlx.a" 2>/dev/null | grep -ci nax)
    if [ "$NAX" -gt 0 ]; then record mlx PASS "pin ${PIN:0:9}, NAX kernels compiled in"
    else record mlx FAIL "built WITHOUT NAX kernels (toolchain missing at build time?) — rebuild"; fi
fi

# 3. build
BUILD_ARGS=(-workspace Apertura.xcworkspace -configuration Release -derivedDataPath "$DD"
            ARCHS=arm64 ONLY_ACTIVE_ARCH=YES CODE_SIGNING_ALLOWED=NO)
R="$DD/Build/Products/Release/AperturaResearch"
if xcodebuild "${BUILD_ARGS[@]}" -scheme AperturaResearch build > "$LOG/build.log" 2>&1; then
    record build PASS "AperturaResearch"
else
    record build FAIL "see $LOG/build.log: $(grep -m1 ' error:' "$LOG/build.log" | cut -c1-120)"
fi

if [ ! -d "$MODEL" ]; then
    record model FAIL "bundle not found: $MODEL"
    for c in kit-tests persist fixtures bench; do record $c SKIP "no model"; done
else
    # 4. AperturaKit tests (Debug, test-host derived data separate from the Release build)
    if TEST_RUNNER_APERTURAKIT_TEST_MODEL="$MODEL" xcodebuild -workspace Apertura.xcworkspace \
           -scheme AperturaKit -derivedDataPath "$DD-test" ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
           test > "$LOG/kit-tests.log" 2>&1; then
        record kit-tests PASS "$(grep -E 'Executed [0-9]+ tests' "$LOG/kit-tests.log" | tail -1 | sed 's/^[[:space:]]*//' | cut -c1-70)"
    else
        record kit-tests FAIL "$(grep -m1 'error: -\[' "$LOG/kit-tests.log" | sed 's/.*error: //' | cut -c1-110)"
    fi

    if [ -x "$R" ]; then
        # 5. persist-verify with a fixed persona slice of the repo's War and Peace text
        head -c 6000 "$ROOT/leo tolstoy - war and peace.txt" | tail -c 4000 > "$LOG/persona.txt"
        "$R" "$MODEL" --persist-verify "$LOG/persona.txt" > "$LOG/persist.log" 2>&1
        if grep -q "byte-identity of both turns: PASS" "$LOG/persist.log"; then
            record persist PASS "$(grep -o '[0-9]*x faster' "$LOG/persist.log") restore, tamper re-primes"
        else
            record persist FAIL "see $LOG/persist.log"
        fi

        # 6. fixtures (optional: needs the generated fixture + the bf16 QAT source)
        if [ -f "$FIXTURE" ] && [ -d "$QAT_SRC" ]; then
            ok=1
            for mode in "" "--fused"; do
                "$R" "$QAT_SRC" "$FIXTURE" $mode > "$LOG/fixtures${mode}.log" 2>&1
                grep -q "CONFORMANCE PASS" "$LOG/fixtures${mode}.log" || ok=0
            done
            if [ $ok = 1 ]; then record fixtures PASS "E2B QAT vs PyTorch, unfused + fused"
            else record fixtures FAIL "see $LOG/fixtures*.log"; fi
        else
            record fixtures SKIP "no fixtures_e2b_qat / QAT source (Tools/generate_fixtures.py)"
        fi

        # 7. warm bench (informational)
        line=$(APERTURA_BENCH_WARM=16 "$R" "$MODEL" --bench-eager --prefill 2048 --decode 64 2>&1 | grep '\[eager')
        pf=$(echo "$line" | sed -nE 's/.*prefill 2048 tok: *([0-9.]+) tok\/s.*/\1/p')
        dc=$(echo "$line" | sed -nE 's/.*decode 64 tok: *([0-9.]+) tok\/s.*/\1/p')
        if [ -n "$pf" ]; then record bench PASS "prefill 2048: $pf tok/s, decode: $dc tok/s (warm)"
        else record bench FAIL "bench produced no result"; fi
    else
        for c in persist fixtures bench; do record $c SKIP "no AperturaResearch binary"; done
    fi
fi

echo; echo "== summary"
for r in "${RESULTS[@]}"; do echo "$r"; done
[ $FAILED = 0 ] && echo "SMOKE PASS" || echo "SMOKE FAIL"
exit $FAILED
