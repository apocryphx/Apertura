# The `.apml` Bundle — Apertura Model Package

**Status:** design spec (pre-implementation) · **Format version:** 1 · **Owner:** Apertura

A self-contained, MLX-native, on-disk package for an Apertura model: quantized
weights + tokenizer + config + a self-describing manifest, presented to the
Finder as a single file.

---

## 1. Why a package (and why not the alternatives)

The goal: **one artifact you hand the app (or AirDrop to another Mac) and it
just runs**, with all "pertinent information" inside and the internal layout
free to evolve.

- **Not GGUF.** GGUF is ggml's quantization + tensor conventions + embedded
  tokenizer. MLX can't consume its K-quants, and even unquantized GGUF needs
  tensor-name remap / un-permute. Incompatible at the *semantics* layer, not
  just the container.
- **Not a single fat `.safetensors` byte-stream.** That only buys
  transmissibility, and nobody emails an 18 GB model. The one realistic
  transport — Mac↔Mac AirDrop — preserves *packages* faithfully (same as
  `.pages`, `.rtfd`, `.xcodeproj`). So a directory package loses nothing and
  avoids header-bloat hacks (e.g. embedding a 32 MB `tokenizer.json` as a
  header string or a `uint8` tensor).
- **Not Core AI's `.aimodel`.** Core AI (WWDC 2026, iOS/macOS 27.0+) is the
  *sealed, fast* path: quantization is baked at specialization, inference is a
  compiled `InferenceFunction` graph, sampling is fixed (`greedy/topK/topP`).
  Apertura is the *open, observable* instrument — freeze mid-thought, read every
  layer, coarsen precision step by step. `.aimodel` cannot express that. Core AI
  is a future **deployment tier**, not a replacement (see §7).

A macOS **package** gives us the single-file ergonomics (Finder, copy,
duplicate, AirDrop) with native members in their natural formats, each read by
the loader we already have (MLX safetensors, `OCTTokenizer`, BOM-safe NSJSON).

> Apple's own `.aimodel` is *also* a package — queryable `metadata`, a
> `summary`/statistics view, and `AIModelCache` holding device-specialized
> variants. The convergence is a strong signal this shape is right; this spec
> deliberately mirrors that pattern (manifest-as-`AIModelAsset.metadata`,
> `variants`-as-specialized-assets).

---

## 2. macOS package registration

A directory becomes a Finder-opaque single file when its extension is a UTI
conforming to `com.apple.package`. Declared in **Apertura's** `Info.plist`:

```xml
<!-- Exported type: the .apml package -->
<key>UTExportedTypeDeclarations</key>
<array>
  <dict>
    <key>UTTypeIdentifier</key>            <string>com.elarity.apertura.model</string>
    <key>UTTypeDescription</key>           <string>Apertura Model</string>
    <key>UTTypeConformsTo</key>
    <array>
      <string>com.apple.package</string>
      <string>public.data</string>
    </array>
    <key>UTTypeTagSpecification</key>
    <dict>
      <key>public.filename-extension</key> <array><string>apml</string></array>
    </dict>
  </dict>
</array>

<!-- Document type: let Apertura open .apml -->
<key>CFBundleDocumentTypes</key>
<array>
  <dict>
    <key>CFBundleTypeName</key>            <string>Apertura Model</string>
    <key>LSItemContentTypes</key>          <array><string>com.elarity.apertura.model</string></array>
    <key>CFBundleTypeRole</key>            <string>Viewer</string>
    <key>LSHandlerRank</key>               <string>Owner</string>
  </dict>
</array>
```

Once Launch Services registers Apertura, `Gemma-31B-Q4.apml` shows as one file;
"Show Package Contents" reveals the directory. **Faithfulness note:** the bytes
are always intact; the single-file *appearance* requires the receiver to have
the UTI registered. Without Apertura installed it shows as a folder and still
loads — no data loss, only cosmetics. On Linux/HF it is simply a directory (a
feature: Python tools can read members directly).

> Modern equivalent: declare the same via the `UniformTypeIdentifiers`
> framework (`UTType(exportedAs: "com.elarity.apertura.model", conformingTo:
> .package)`). The Info.plist keys remain the registration source of truth.

---

## 3. On-disk layout

```
Gemma-Apertura-31B-Q4.apml/          ← Finder: one file
  manifest.json                      # self-describing trust anchor (see §4)
  config.json                        # ESModelConfig source (HF text_config)
  tokenizer.json                     # consumed directly by OCTTokenizer (BOM-safe)
  chat_template.jinja                # optional
  generation_config.json             # optional
  weights/
    mlx-q4/                          # one directory per runtime variant (§7)
      model.safetensors              # (or sharded model-00001-of-000NN + index)
      quantization.json              # bits, group_size, embed_bits, per-tensor scheme
```

Rules:
- The **directory is the unit**; the loader entry point is the package URL.
- Weights live under `weights/<variant-id>/` so adding/removing a variant never
  touches the top-level contract.
- Keep members in their *natural* format; do not embed large blobs into JSON.

---

## 4. `manifest.json` schema

The manifest is the self-identifier and trust anchor — Apertura's analogue of
`AIModelAsset.metadata` + `summary`.

```jsonc
{
  "format_version": 1,                       // loader refuses unknown MAJOR
  "kind": "apertura-model",                  // disambiguates from raw HF safetensors
  "name": "Gemma 4 31B (QAT q4_0) — Apertura Q4",
  "created": "2026-06-19T00:00:00Z",         // stamp at write time (not in-script)

  "architecture": "gemma4",                  // selects the ESGemma4* assembly
  "config": "config.json",                   // path within the package

  "tokenizer": { "file": "tokenizer.json", "kind": "huggingface-tokenizers" },
  "chat_template": "chat_template.jinja",

  "source": {                                // provenance
    "model_id": "google/gemma-4-31B-it-qat-q4_0-unquantized",
    "revision": "<hf commit sha>",
    "recipe": "mlx affine quant, group_size=64, bits=4, embed_bits=8"
  },

  "calibration": {                           // the instrument's self-check
    "reference": "google/gemma-4-31b-it",
    "byte_identity": "argmax-stable",        // claim this artifact was certified against
    "hash": "<sha256 of canonical logits or id stream over the conformance corpus>"
  },

  "default_variant": "mlx-q4",
  "variants": [
    {
      "id": "mlx-q4",
      "runtime": "mlx",                      // "mlx" now; reserved: "coreai"
      "path": "weights/mlx-q4",
      "precision": "q4",
      "quantization": {                      // also duplicated in the variant's quantization.json
        "scheme": "mlx-affine",
        "bits": 4, "group_size": 64, "embed_bits": 8
      },
      "files": ["model.safetensors"],        // or the shard list + index
      "approx_bytes": 18000000000
    }
    // future: { "id": "mlx-bf16", ... }, { "id": "coreai", "runtime": "coreai", "files": ["model.aimodelc"] }
  ]
}
```

Validation on load:
1. `format_version` major must be known, else refuse.
2. `kind == "apertura-model"`, else treat as a foreign directory.
3. Resolve `default_variant` (or a caller-requested id) → load that variant only.
4. Optionally verify `calibration.hash` after load (instrument self-test).

---

## 5. Reading & writing the package

Use **`NSFileWrapper`** (directory wrapper) or plain URL-relative file reads —
**not `NSBundle`** (`NSBundle` is for *code* bundles: `.app`/`.framework`; it
expects `Contents/MacOS`/Info.plist and will fight a data package).

- **Read:** treat the package as a URL; read members directly.
  - `mx::load_safetensors((pkg / "weights/mlx-q4/model.safetensors").c_str())`
  - `OCTTokenizer` on `pkg/"tokenizer.json"`
  - `config.json` via the BOM-safe NSJSON path
- **Write (atomic):** build the package in a temp directory, then
  `[NSFileManager replaceItemAtURL:withItemAtURL:...]` (or `NSFileWrapper`'s
  atomic write) so a half-written `.apml` never appears valid.

---

## 6. Mapping to the existing code (as implemented)

The model already *runs* quantized (`ESModelConfig.quantBits`/`quantEmbedBits`;
`ESLinear`/`ESExperts`/`ESEmbedding` quantize at construction). What was missing
was *persistence*. Implemented as follows.

**Export — `es::exportQuantizedBundle(modelDir, outPath, opts)`** in
[`ESWeightLoader.mm`](aptransformer/ESWeightLoader.mm):
1. Load BF16 via `ESWeightLoader` (HF mode) and iterate `loader.all()`.
2. Quantize by a precise **name rule** (`octIsLayerProjQuant`: q/k/v/o, gate/up/down,
   `experts.gate_up_proj`/`down_proj` at `bits`; `embed_tokens.weight` at `embedBits`),
   emitting `name` / `name.scales` / `name.biases`; everything else passes through bf16.
   This is drift-proof because the loader map reflects the *real* tensor set (no
   `v_proj` for global `k_eq_v`, MoE only where present, tied embed) — and §6's
   round-trip test catches any rule/runtime divergence. (Simpler than a
   `serialize()`-per-layer pass and needs zero layer-code changes for export.)
3. `mx::save_safetensors` + write `manifest.json`/`quantization.json`, copy
   `config.json`/`tokenizer.json`/`chat_template.jinja`, assemble atomically (§5).

**Reload — bundle-aware `ESWeightLoader`:**
1. The constructor auto-detects a package (`manifest.json` with
   `kind == "apertura-model"`) and dispatches to `loadBundle`, which loads the
   default variant's `model.safetensors` **verbatim** — packed `uint32` weights
   are never `astype`-d (the HF path's cast at [`ESWeightLoader.mm`](aptransformer/ESWeightLoader.mm) would destroy them).
2. Already-quantized constructors on `ESLinear`/`ESEmbedding`/`ESExperts` adopt
   `(packed, scales, biases)` and **skip** `mx::quantize`.
3. Three factories — `esMakeLinear` / `esMakeEmbedding` / `esMakeExperts` — are
   the single routing point: pre-quantized from the bundle when `hasQuantized(name)`,
   else the bf16/quantize-now path. Construction sites (`ESAttention`, `ESMLPBlock`
   via `ESDecoderLayer`, `ESExperts`, `ESGemma4TextModel` embed) call the factories.

**QAT lattice-exact recipe — `opts.lattice` / `--export-lattice` (2026-10-09).**
Google's `*-qat-q4_0-unquantized` checkpoints are bf16 weights that already sit on an int4
lattice: per 32-element block along the input dim, `w = bf16(k·d)`, `k ∈ [-8, 7]`, and `d`
is the **QAT-learned** per-block step — *not* ggml's `absmax/-8` and not `absmax/7`; the
extreme code present is 8 in ~60% of blocks, 7 in ~38%, smaller in the rest (measured on
the 31B). Running `mx::quantize` over that is a *second* quantization with a min/max scale
and a zero point it was never trained for — g64 even straddles two trained blocks — and
lands most weights off their trained values (llama.cpp's naive Q4_0 conversion matches the
lattice in only ~25% of bytes; Unsloth's Gemma 4 QAT notes). `es::quantizeQ4Lattice`
recovers `d` per block from the lattice structure (smallest `K ≤ 8` with `absmax/K` a
consistent integer lattice, least-squares refined, snapped to the bf16 step that reproduces
the most stored weights) and writes MLX's **ordinary affine format** with `scale = d`,
`bias = -8·d`, `code = k + 8`, g32, 4 bits — so `mx::dequantize`/`quantized_matmul`
reproduce `k·d` in float, no loader or kernel change. Ceiling: the checkpoint stores bf16
*roundings* of `k·d`, so one bf16 step cannot hit every element — **~90.5% of weights
bit-exact, 100% within one bf16 ulp** on the 31B, i.e. the Q4 weights are as close to the
trained lattice as the bf16 checkpoint itself is. Per tensor the exporter measures the fit
(`ESLatticeFit`); a tensor off the lattice falls back to `mx::quantize` g32 and is listed.
`embed_tokens` / the PLE table are stored exact at 4 bits when on the grid (the manifest's
`embed_bits`/`ple_bits` record what was written; `quantization.json` gains
`lattice: "qat-int4-g32"` + fit stats). Gates: `--lattice-scan <snapshot>` (per-class fit,
no export — run first), `--verify-lattice <o.apml>` (dequantize every bundle tensor vs the
source; PASS ≥ 99.9% within one ulp), `--vs-bf16` (forward-level top-1 vs the bf16 QAT
reference), and `testQ4LatticeExactRoundTrip` (synthetic lattice → export → reload bit-exact;
shows `mx::quantize` g32 is *not*). Loading a lattice bundle with `--quant-embed` re-quantizes
the exact Q4 head to the requested width (lossy); the default (flag absent) keeps it verbatim.

**Acceptance test (implemented, runnable per-build):**
`ESPrimitivesTests/testReloadMatchesInMemoryQuant` asserts reload-from-`.apml`
forward output is **bit-identical** (maxAbsDiff == 0) to the in-memory-quantize
path, for the linear and embedding projections, on tiny tensors.
`testExportQuantizedBundle` validates the package shape (u32 weights + scales +
biases, norms bf16, manifest, copied aux). The full-31B **argmax-stable**
conformance is the same invariant at scale via the AperturaResearch driver;
stamp its hash into `manifest.calibration.hash`.

> Note: a quantized bundle supports MoE only via `moeSparse` (gather_qmm) — the
> bf16 dense expert `forward()` path needs un-quantized expert weights the bundle
> doesn't carry. Irrelevant to the dense 31B (no experts).

---

## 7. Variants & forward compatibility

`variants[]` mirrors `AIModelCache`'s specialized assets — one package, multiple
runnable representations, selected by id at load:

- **`mlx-q4`** (now) — the inspectable instrument, Tier 1/2.
- **`mlx-bf16`** (optional) — full-precision reference for calibration.
- **`coreai`** (future, gated on 27.0) — a `.aimodelc` produced via
  `coreai_torch`/`torch.export` from the HF PyTorch Gemma-4, wrapped through
  Foundation Models (`LanguageModel`/`LanguageModelExecutor`) for a **ship-it**
  tier. Distinct artifact; **loses inspectability + custom sampling** — a
  deployment option, not the research path.

The three-tier story: **MLX-unfused (microscope) · MLX-fused (fast research) ·
Core AI (ship-it)** — co-packageable, manifest-selected.

---

## 8. Naming — collisions to avoid

Core AI owns these; do not reuse:
- Extensions `.aimodel` / `.aimodelc`.
- Types `AIModel`, `AIModelAsset`, `AIModelCache`, `NDArray`, `InferenceFunction`,
  `ComputeUnitKind`, `SpecializationOptions`.

Apertura uses `.apml`, UTI `com.elarity.apertura.model`, and the existing `ES*`
type prefix — all clear of the above.

---

## 9. Open questions

- Sharding threshold for `model.safetensors` (single file vs N×5 GB shards).
- Whether `calibration.hash` covers logits or the argmax id-stream (id-stream is
  cheaper and matches the byte-identity contract).
- Whether to copy the full `tokenizer.json` or also record its source hash for
  provenance.
