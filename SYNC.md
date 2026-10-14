# SYNC — keeping two machines in step

The project's resources fall into three tiers with different sync mechanisms.
The design rule, learned the hard way (2026-08-12, three destruction events in
one day): **verified-but-unsynced is the most expensive state anything can be
in.** Small and precious goes in private git; huge and static goes in iCloud;
huge and regenerable is regenerated.

## Tier 1 — git (small, precious, versioned)

| What | Repo | Notes |
|---|---|---|
| Code | `apocryphx/Apertura` (+ `ObjCTokenizer`, `ml-explore/mlx` plain upstream, …) | public/existing |
| Isolde's scripture + `persona_history/` | `apocryphx/isolde-scripture` **PRIVATE** | repo lives at `/Volumes/Macintosh HD/Users/apocryphx/Models`; `.gitignore` admits only text. Commit + push after any session where the persona changed (inscriptions!). |
| Claude project memory | `apocryphx/apertura-memory` **PRIVATE** | lives in the Claude memory dir |
| Workspace | materialized by `Tools/bootstrap.sh` | the workspace file itself stays outside the repo; the script writes it deterministically |

Git working trees stay under `*.nosync` folders — **never inside an
iCloud-synced path** (sync races corrupt `.git`, conflicts fork files).

## Tier 2 — Hugging Face + local Resources folder (huge, regenerable-by-download)

All custom conversions are PUBLIC on Hugging Face with Gemma-license
passthrough cards — that is the durable, versioned source of truth.

**Naming rule (2026-10-13):** one `<name>` per bundle, everywhere —
`apocryphx/<name>-apml` on HF, `/Volumes/Gemma 4/<name>.apml` locally.
`<name>` = `gemma-4-<variant>-it-<qat->q4`, where `qat-` means *built from Google's
`…-it-qat-q4_0-unquantized` checkpoint* (lattice-exact g32) and its absence means built from
the regular `…-it` checkpoint (plain affine). The manifest's `source.model_id` is the ground
truth; the name must agree with it.

| `<name>` | Source | Notes |
|---|---|---|
| `gemma-4-31b-it-qat-q4` | QAT | lattice-exact since 2026-10-09 (overwritten in place; earlier revisions were affine g64; the `…-q4-g32-apml` repo was deleted) |
| `gemma-4-12b-it-qat-q4` | QAT | lattice-exact since 2026-10-09 |
| `gemma-4-26b-a4b-it-qat-q4` | QAT | lattice-exact since 2026-10-09 |
| `gemma-4-E2B-it-qat-q4` | QAT | lattice-exact, Q4 head + Q4 PLE table, Apertura ≥ 2026-10-09; renamed from `…-E2B-it-q4-apml` 2026-10-13 (HF redirects the old id) |
| `gemma-4-E4B-it-qat-q4` | QAT | same as E2B; renamed from `…-E4B-it-q4-apml` |

No regular-source (non-`qat-`) bundle is published at present: `gemma-4-26b-a4b-it-q4-apml`
(plain affine g64) was deleted 2026-10-13, superseded by the QAT bundle. Plus
`apocryphx/embeddinggemma-300m-qat-q4_0-coreml`. Every bundle passed
`--verify-bundle` bit-exact before publishing.

Working copies live on each machine's internal **`Gemma 4`** APFS volume (not
iCloud-synced — HF makes file-level cloud sync of these redundant), with the same
layout on both Macs so repo paths match:

```
/Volumes/Gemma 4/<name>.apml       the Apertura bundles (Tools/smoke.sh and the AperturaKit
                                   scheme's APERTURAKIT_TEST_MODEL point here)
/Volumes/Gemma 4/sources/<google repo name>   bf16 QAT sources for fixtures / --vs-bf16
                                   (M4 Max: symlinks into ~/.cache/huggingface/hub snapshots)
/Volumes/Gemma 4/mlx-community/…   reference checkpoints for mlx-lm comparisons
```

The older `~/Documents/GitHub.nosync/Resources/Models/…` paths may remain as symlinks into the
volume (the M6 Mac mini has them since 2026-10-13). Bundles the app or CLI expect under
`…/Users/apocryphx/Models/` are symlinked from there, e.g.:

```sh
ln -s "/Volumes/Gemma 4/gemma-4-31b-it-qat-q4.apml"     "/Volumes/Macintosh HD/Users/apocryphx/Models/"
```

Provision any machine with `hf download apocryphx/<name>-apml --local-dir "/Volumes/Gemma 4/<name>.apml"`.
The iCloud "ES Resources" folder is RETIRED (2026-08-12) — iCloud remains an
option only for future artifacts that are both huge and NOT publishable.

## Tier 3 — regenerate, never sync

| What | Why not sync | How to get it on the other machine |
|---|---|---|
| HF originals (`gemma-4-31B*`, ~116 GB) | pure cache; HF's CDN beats iCloud | `hf download` / conversion inputs |
| KV snapshots (`isolde-kv*.safetensors`, ~2 GB each) | rewritten on every persona edit → constant churn | app rebuilds one in ~90 s per mode, then it's cached |
| Test fixtures (`aptransformerTests/Fixtures/*.safetensors`) | regenerable | `Tools/generate_fixtures.py` + friends (needs HF snapshots) |
| Core Data store (chat history) | live SQLite + WAL; file sync corrupts it | see CloudKit below |
| DerivedData, `libmlx.a` | machine-specific; metallib path bakes in absolutely | rebuild per machine: `sh Tools/build_mlx.sh` (checks out the commit in `Tools/mlx.pin`, so machines converge by each re-running it) |

## Chat history: CloudKit, not file sync

The app already uses `NSPersistentCloudKitContainer` and the `CDChatSession`
schema is CloudKit-clean (all attributes optional, external-storage blob).
Turning on the iCloud/CloudKit capability + container in the target gives
cross-machine chat sync through Apple's channel — the plan for persona-in-
Core-Data rides the same rail. Until then, per-machine history; transcripts
are portable JSON (`transcriptJSON`) if one ever needs moving by hand.

## Formerly unprotected repos — resolved 2026-08-12

`code/SDPABench` and `code/ES-Capture` now have private remotes
(`apocryphx/SDPABench`, `apocryphx/ES-Capture`). SDPABench's source was
replayed back to its 2026-08-11 state from session transcripts after the
working copy was destroyed — the third artifact recovered that way in one
day. If a repo exists without a remote, that is a standing bug.

## New machine, from zero

1. `git clone https://github.com/apocryphx/Apertura.git` into `~/Developer/`
   (ObjCTokenizer is a sibling, `~/Developer/ObjCTokenizer`, shared with ES-Archive)
2. `sh Tools/bootstrap.sh` (clones `../ObjCTokenizer` if missing + persona-repo hint)
3. `sh Tools/build_mlx.sh` — clones `ml-explore/mlx` (plain upstream, no
   fork) as `../mlx`, checks out the commit in `Tools/mlx.pin`, and builds
   `libmlx.a` + the metallib
   (needs Xcode's Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain` —
   without it the build quietly drops the M5/M6 Neural Accelerator kernels)
4. `hf download` the `.apml` bundles you need (see Tier 2) — at least E2B, which
   `Tools/smoke.sh` uses (if `hf` hangs, check IPv6: `curl -6 https://www.google.com`)
5. `Tools/smoke.sh` — about a minute; every line should read PASS (fixtures may SKIP)
6. First app launch re-primes each reasoning mode once (~90 s), then snapshots
