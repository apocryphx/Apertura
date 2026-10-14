# Memo — from the M4 Max session to the M6 session (2026-10-13, evening)

Your day's work was pulled and validated on the M4 Max (40-core GPU, 128 GB, no NAX). It holds up.
`origin/main` is now `0a59295`, eight commits past your `0de16da`. Details are in
`SESSION-2026-10-13.md` under "M4 Max validation", "Unified bundle naming" and "ObjCTokenizer moved
to a sibling folder". What you need to know and do:

## Do on the M6
1. `git pull`. Then delete the empty `External/` folder the old submodule leaves behind.
2. Check that `~/Developer/ObjCTokenizer` is at `cfc3b6b` or later. Apertura now builds against
   that sibling, and `cfc3b6b` is the version verified to give Gemma identical token IDs.
3. Rename the local bundles: `/Volumes/Gemma 4/gemma-4-E2B-it-q4.apml` →
   `gemma-4-E2B-it-qat-q4.apml`, same for E4B. Then re-add both in the app's Settings, since the
   app stores models by path. `Tools/smoke.sh` now defaults to the new E2B name.
4. Run `Tools/smoke.sh`, then run the AperturaKit tests once on the 26B:
   `TEST_RUNNER_APERTURAKIT_TEST_MODEL="/Volumes/Gemma 4/gemma-4-26b-a4b-it-qat-q4.apml" xcodebuild -workspace Apertura.xcworkspace -scheme AperturaKit test`

## What changed
- **26B-A4B could never run through AperturaKit, so not in the app or MCP either (fixed, `24f6dde`).**
  `APModel` never set `config.moeSparse`, so a quantized 26B bundle took the dense all-experts
  path. That path needs bf16 expert tensors a `.apml` doesn't carry, so every forward threw MLX
  shape errors (16 test failures, also at `7f3b235`). Your 14/14 runs on E2B and the 12B didn't
  reach this path. `APModel` now picks the sparse path for the MoE family, and so does
  `--facade-verify`'s reference. After the fix, the 26B passes 14/14 and both verify checks.
- **One naming rule:** `gemma-4-<variant>-it-qat-q4` when built from Google's QAT checkpoint,
  `gemma-4-<variant>-it-q4` when built from the regular one. On HF it's `apocryphx/<name>-apml`;
  locally it's `/Volumes/Gemma 4/<name>.apml`. The bf16 sources go in `/Volumes/Gemma 4/sources/`.
  - HF E2B/E4B renamed to `…-it-qat-q4-apml`. Their weights were already the QAT exports
    (sha256-identical); the old names redirect.
  - The plain-q4 26B repo was deleted.
  - `SYNC.md` has the table.
- **ObjCTokenizer is now a sibling** (`../ObjCTokenizer`), as in ES-Archive `f10126e`.
  `Apertura.xcodeproj` references it as a subproject, so the bare project builds on its own.
  Gemma token IDs are identical between `dd3b653` and `cfc3b6b` on 11,971 texts.

## M4 Max results for your open items
- **26B vs bf16:** 69/69 on the probes and 254/256 on a 2,149-token prompt. Old and new builds
  give the same result.
- **E2B / E4B / 12B:** same as the archive, and old = new on the long prompt.
- **31B:** 60/60 on the probes, 251/256 on the long prompt (old = new); facade and persist checks
  pass. Prefill 190 → 204 tok/s at 2K.
- **Prefill, old → new, cold (tok/s):**
  - E2B: 1,511 → 5,737 at 2K; 2,449 → 6,823 at 4K. About 8,100 warm, roughly matching your
    M6 with NAX.
  - E4B: 1,172 → 2,208 at 2K.
  - 26B: 463 → 1,366 at 2K.
  - 12B: unchanged. Decode unchanged everywhere.
- **Elastic chunk 512 vs 1024:** within ±0.5% on the M4 Max, so 1024 is neutral here and the
  shared default stands. No per-machine profile is needed.

## Not done
- These results aren't in the ES Archive yet. Your entry "The day an entry-level Mac mini tripled
  Apertura's prefill…" lists the 26B vs-bf16 and the M4 Max chunk re-measure as open; both are
  answered above.
- The fixtures step still skips on the M4 Max: `fixtures_e2b_qat.safetensors` is gitignored and
  has no PyTorch environment here.
