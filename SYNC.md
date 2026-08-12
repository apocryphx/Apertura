# SYNC — keeping two machines in step

The project's resources fall into three tiers with different sync mechanisms.
The design rule, learned the hard way (2026-08-12, three destruction events in
one day): **verified-but-unsynced is the most expensive state anything can be
in.** Small and precious goes in private git; huge and static goes in iCloud;
huge and regenerable is regenerated.

## Tier 1 — git (small, precious, versioned)

| What | Repo | Notes |
|---|---|---|
| Code | `apocryphx/Apertura` (+ `ObjCTokenizer`, `mlx` fork, …) | public/existing |
| Isolde's scripture + `persona_history/` | `apocryphx/isolde-scripture` **PRIVATE** | repo lives at `/Volumes/Macintosh HD/Users/apocryphx/Models`; `.gitignore` admits only text. Commit + push after any session where the persona changed (inscriptions!). |
| Claude project memory | `apocryphx/apertura-memory` **PRIVATE** | lives in the Claude memory dir |
| Workspace | materialized by `Tools/bootstrap.sh` | the workspace file itself stays outside the repo; the script writes it deterministically |

Git working trees stay under `*.nosync` folders — **never inside an
iCloud-synced path** (sync races corrupt `.git`, conflicts fork files).

## Tier 2 — iCloud Drive (huge, write-once)

Candidates: the converted `.apml` bundles (~36 GB total). One-time upload,
then both laptops have them.

Setup (once, per machine — **quit Apertura first**, it mmaps the bundles):

```sh
mkdir -p ~/Documents/Apertura\ Resources
mv "/Volumes/Macintosh HD/Users/apocryphx/Models/gemma-4-31b-it-qat-q4.apml"     ~/Documents/Apertura\ Resources/
mv "/Volumes/Macintosh HD/Users/apocryphx/Models/gemma-4-31b-it-qat-q4-g32.apml" ~/Documents/Apertura\ Resources/
ln -s ~/Documents/Apertura\ Resources/gemma-4-31b-it-qat-q4.apml     "/Volumes/Macintosh HD/Users/apocryphx/Models/"
ln -s ~/Documents/Apertura\ Resources/gemma-4-31b-it-qat-q4-g32.apml "/Volumes/Macintosh HD/Users/apocryphx/Models/"
```

The symlinks keep every existing path (app defaults, CLI invocations) working.

**Hard requirements:**
- System Settings → Apple ID → iCloud Drive → **"Optimize Mac Storage" OFF**
  on BOTH machines (or right-click the folder → "Keep Downloaded"). MLX mmaps
  the weights; an evicted stub stalls the app mid-load for a 17 GB download.
- Before relying on the second machine, confirm the download completed
  (`brctl download`, or just check the folder shows no cloud badges).
- iCloud's scheduler is opaque: the first 36 GB upload can take hours even on
  fiber. It is a background convenience, not a deadline mechanism.

## Tier 3 — regenerate, never sync

| What | Why not sync | How to get it on the other machine |
|---|---|---|
| HF originals (`gemma-4-31B*`, ~116 GB) | pure cache; HF's CDN beats iCloud | `hf download` / conversion inputs |
| KV snapshots (`isolde-kv*.safetensors`, ~2 GB each) | rewritten on every persona edit → constant churn | app rebuilds one in ~90 s per mode, then it's cached |
| Test fixtures (`aptransformerTests/Fixtures/*.safetensors`) | regenerable | `Tools/generate_fixtures.py` + friends (needs HF snapshots) |
| Core Data store (chat history) | live SQLite + WAL; file sync corrupts it | see CloudKit below |
| DerivedData, `libmlx.a` | machine-specific; metallib path bakes in absolutely | rebuild per machine |

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

1. `git clone https://github.com/apocryphx/Apertura.git` into `…/code/`
2. `sh Tools/bootstrap.sh` (workspace + ObjCTokenizer + persona-repo hint)
3. Clone the mlx fork, check out the pin, build `libmlx.a` (see
   `aptransformer/PERFORMANCE_ROADMAP.md` / memory for the colocate phases)
4. Wait for / download the `.apml` bundles via iCloud (badges gone)
5. First app launch re-primes each reasoning mode once (~90 s), then snapshots
