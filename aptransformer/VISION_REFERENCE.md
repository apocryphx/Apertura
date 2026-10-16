# Apertura Vision Reference — Gemma-4 image input (annotated)

A standing explanation of how a picture reaches the Gemma-4 text decoder in Apertura: the image
processor, the vision tower, the multimodal embedder, and the soft-token substitution at prefill.
Source of truth is the code in `ESImagePreprocessor.{h,mm}`, `ESGemma4VisionModel.{h,mm}` and the
`ESVision*` classes, plus the `ESSoftTokens` path in `ESGemma4TextModel.mm`; this file mirrors the
key pieces with commentary. Conformance is gated against the PyTorch reference by
`AperturaResearch --vision-conformance` on `Tools/generate_vision_fixtures.py`'s fixture.

Verified on the E2B QAT checkpoint (2026-10-15): image processor bit-exact, every tower stage
within the bf16 per-op floor (teacher-forced isolation: mean 95.8% of elements within 1 ULP, worst
layer 92.3%), text argmax and 20/20 greedy tokens MATCH end to end (own pixels → own tower → own
prefill), and the same on the lattice-Q4 `.apml` export.

---

## 0. The big picture

```
 CGImage ──ESImagePreprocessor──► patches f32 [n, 768] + positions (x,y) [n, 2]
                                      │
 ESGemma4VisionModel:  patch_embedder → 16 × encoder layer (axial 2-D RoPE) → pooler (3×3, f32·√768)
                                      │                                   → cast bf16 [tokens, 768]
                       embed_vision:  rmsnorm(no scale) → Linear 768→1536   → soft tokens [tokens, 1536]
                                      │
 ESChatTemplate:   <|turn>user\n <|image> tokens × <|image|> <image|> {text} <turn|>\n …
                                      │
 ESGemma4TextModel::forward(ids, …, soft):  h = embed(ids)·√H ;  h[image positions] = soft tokens
                                            PLE token-identity rows for those positions = pad (id 0)
                                            masks: causal (E2B) — or OR(same image block) on local
                                            layers when text_config.use_bidirectional_attention=="vision"
```

The number of soft tokens is **not** the constant 280: it is `(h/48)·(w/48)` of the resized picture
(the processor fits the largest grid of 16-px patches ≤ 2520 whose sides are multiples of 48). The
640×400 fixture picture resizes to 1008×624 → 63×39 patches → **273** soft tokens. The chat
template therefore expands each image from its own count (`ESChatMessage::imageSoftTokenCounts`,
`ESImagePreprocessor::softTokensForSize`).

---

## 1. The image processor — the place ports go wrong  (ESImagePreprocessor)

`Gemma4ImageProcessor` (torchvision backend), reproduced bit-for-bit:

1. **Decode to RGB8.** PIL `convert("RGB")`. We draw the CGImage into an 8-bit RGB bitmap in the
   image's own RGB colour space (no colour matching). Alpha is composited over black — PIL drops
   the channel instead; opaque photos are unaffected.
2. **Aspect-preserving resize** (`get_aspect_ratio_preserving_size`): `factor = sqrt(2520·256 /
   (h·w))`, sides floored to multiples of 48, with the degenerate-side fixups.
3. **Bicubic, antialias=True — the uint8 integer path.** torchvision hands the uint8 tensor to
   aten, whose uint8 antialiased resampler is PIL's integer algorithm, not the float one:
   - weights in double: Keys cubic `a = −0.5`, `support = 2·scale`, `center = scale·(i+0.5)`,
     `xmin/xsize` as in `_compute_indices_min_size_weights_aa`, normalised per output pixel;
   - quantised to int16 at a per-pass `precision` such that the largest weight fits 15 bits
     (`_compute_index_ranges_int16_weights`), rounding `(int)(0.5 + v)` / `(int)(−0.5 + v)`;
   - `out = (1 << (prec−1)) + Σ src·w) >> prec`, clamped to [0, 255]; **W pass then H pass, uint8
     intermediate.**
   The float32 separable path (what a uint8→float cast would run) rounds ~1.1% of pixels one
   level differently — measured on the fixture; that was the first FAIL of this port.
4. **Rescale** `x · float32(1/255)`. No mean/std normalisation (the model does `2(x−0.5)`).
5. **Patchify** `(C,H,W) → (nh·nw, 16·16·3)`, per-patch vector ordered `(py, px, c)`, patches
   row-major; positions `(x, y)`. The reference pads to 2520 with `(−1, −1)`; the engine runs the
   valid patches only (padding is masked out of attention and zeroed before pooling there).

Gate: `[pixel_values] PASS bit-exact 100.0000%`.

---

## 2. The vision tower  (ESGemma4VisionModel, ESVision*)

Config (E2B `vision_config`): hidden 768, 16 layers, 12 heads × 64, intermediate 3072,
`rms_norm_eps 1e-6`, `rope_theta 100`, `use_clipped_linears true`, `standardize false`.

**Patch embedder.** `h = input_proj(bf16(2·(px − 0.5))) + table[0][x] + table[1][y]`; the learned
table is `[2, 10240, 768]`.

**Clipped linears** (`ESClippableLinear`): QAT bounds stored with the checkpoint as bf16 scalars —
`clamp(clamp(x, in_min, in_max) @ Wᵀ, out_min, out_max)`. Every q/k/v/o and gate/up/down of the
tower has them; the patch embedder's `input_proj` and `embed_vision`'s projection are plain.

**Axial 2-D RoPE** (`ESVisionRotaryEmbedding`): `inv_freq = 1/100^(arange(0,32,2)/32)` (16
entries); angles `x·inv_freq` and `y·inv_freq`; tables laid out `[c_x, c_x, c_y, c_y]` (64 wide);
`apply_multidimensional_rope` splits the head into two 32-wide halves and runs NEOX `rotate_half`
on each with its own coordinate. cos/sin computed in f32, cast to bf16.

**Attention** (`ESVisionAttention`): `q_norm`/`k_norm` with scale, weightless `v_norm`, scaling
1.0, no mask (bidirectional over all valid patches), f32 softmax → bf16 → `@ V`.

**Layer** (`ESVisionEncoderLayer`): the text decoder's 4-norm sandwich without `layer_scalar`.

**Pooler** (`ESVisionPooler`): kernel index `x/3 + (maxX/3)·(y/3)`, one-hot `[n, tokens]/9` in
f32, `pooled = Wᵀ @ h.float()`, **cast back to bf16**, then `.float() · √768` — returned in f32.
`Gemma4VisionModel.forward` then (optionally) standardises and casts to bf16.

**Multimodal embedder** (`ESMultimodalEmbedder`): weightless RMSNorm (eps 1e-6) → `Linear(768→1536)`.

Gate (teacher-forced isolation, per layer): `>= 85% within 1 ULP, <= 5% beyond 4 ULP` — the same
thresholds the text decoder uses; measured 92.3–97.7% / 0.35–1.83%. The chained 16-layer run is
printed for information only (bf16 accumulation over 2457 bidirectional tokens drifts from the CPU
reference exactly as the text chain does); the end-to-end gate is the text argmax/greedy.

---

## 3. Into the text model  (ESSoftTokens, ESGemma4TextModel::forward)

`Gemma4Model.forward` does three things with an image, reproduced as:

1. **Scatter.** `inputs_embeds.masked_scatter(image_mask, soft)` → `ESSoftTokens::apply`:
   `mx::scatter` of the `[k, 1536]` soft tokens over the `<|image|>` positions of the scaled
   embedding. The placeholder ids stay in the token vector.
2. **PLE (E2B/E4B).** The reference forms `per_layer_inputs` from `where(image_mask, pad_token_id,
   input_ids)` — the pad row (id 0) for every soft token — while the context projection sees the
   scattered embeddings. `computePerLayerInputs(pleTokenIds(tokens, soft, padTokenId), h)`.
3. **Masks.** `text_config.use_bidirectional_attention` is `null` on E2B/E4B → plain causal
   (`create_masks_for_generate`). When it is `"vision"` (`config.useBidirectionalVisionAttention`)
   the local (sliding) layers get `AND(sliding, OR(causal, same image block))`; global layers stay
   causal (`create_masks_for_vision_model`). `buildMask(…, blockIds)` implements the overlay for
   keys of the current forward; **no fixture exercises it** (no such checkpoint on this Mac).

Chunked prefill (`ESGemma4TextForCausalLM::lastLogits`) slices the substitutions per chunk
(`ESSoftTokens::slice`). Text-only calls pass no soft tokens and are byte-identical to before
(`fixtures_e2b_qat` re-gated: argmax + greedy MATCH, 14/14).

Gate: `inputs_embeds` bit-exact with the reference's own soft tokens; `argmax MATCH`;
`greedy 20/20 MATCH` — both with golden soft tokens (text side alone) and end to end.

---

## 4. Prompt grammar and the session  (ESChatTemplate, APLocalSession)

`<|image|>` in the jinja becomes `<|image>` (255999) + N × `<|image|>` (258880) + `<image|>`
(258882) via `Gemma4Processor.replace_image_token`. `ESChatTemplate::imageTokens(n)` renders that;
`build()` and `APLocalSession`'s turn delta put the images **before** the text, as the template
renders `[image…, text]` content parts. Gate: the fixture's 291 ids reproduced exactly.

Thinking off, generation prompt: whether an EMPTY thought channel is pre-closed after
`<|turn>model\n` is a per-checkpoint fact (`ESModelConfig::preclosesEmptyThought`, read from the
checkpoint's `chat_template.jinja`; without one, `!hasPLE`). The 12B/26B/31B templates emit
`<|channel>thought\n<channel|>` and those models produce the tags themselves; the E2B/E4B templates
do not, and pre-closing there made E2B emit a stray `<channel|>` and run on (model card, "Disabled
Thinking"). The gate now requires the exact id count.

`APLocalSession` encodes each `APImage` (from `fileURL`) once per turn through
`[APModel internalVision]` and prefills the delta with the `ESSoftTokens`; image tokens are ordinary
cache positions, so rewind marks, excision, checkpoints and snapshots are unchanged — except that
the prime-snapshot fingerprint now also hashes the image identifiers (two pictures of one size
render identical placeholder ids).

---

## 4b. The unified family (12B QAT) — encoder-free embedder  (ESUnifiedVisionEmbedder)

`Gemma4UnifiedForConditionalGeneration` (`model_type gemma4_unified`, `vision_config.model_type
gemma4_unified_vision`) has **no tower**. `ESVisionConfig::kind == unified` switches three things:

- **Processor** (`Gemma4UnifiedImageProcessor`): same resize rule and integer bicubic; then
  `patches_merge` folds each 3×3 group of 16-px patches into one 48×48×3 = 6912 vector per soft
  token, in plain row-major pixel order (y, x, c) over the block, tokens row-major over the block
  grid, positions `(x/3, y/3)`. `ESImagePreprocessor` cuts 48-px patches directly
  (`ESImagePatches::merged`). No padding (reference pads to 280 with `(−1,−1)`).
- **Embedder** (`ESUnifiedVisionEmbedder`): `bf16(pixels)` — **no** `2(x−0.5)` scaling —
  `LN1(6912) → Dense 6912→3840 + bias (one fused addmm) → LN2 → + pos[x,0] + pos[y,1]
  (table [1120, 2, 3840]) → LN3 (pos_norm) → RMSNorm(no scale, 1e-6) → Linear 3840→3840`.
  `nn.LayerNorm` (eps 1e-5) is computed in f32 and rounded once to bf16. Tensors:
  `vision_embedder.{patch_ln1, patch_dense, patch_ln2, pos_embedding, pos_norm}` +
  `embed_vision.embedding_projection.weight`.
- **Masks**: `text_config.use_bidirectional_attention == "vision"` on the 12B, so the blockwise
  overlay of §3 is live there (sliding window 1024, 48 layers, no PLE).

Gate: `AperturaResearch <12B source> fixtures_vision_unified_12b.safetensors --vision-conformance --ulp`
(`Tools/generate_unified_vision_fixtures.py`, embedder-only: the 10 vision tensors loaded into a bare
`Gemma4UnifiedVisionEmbedder`). Measured 2026-10-15: pixel_values + positions **bit-exact**; ln1 100%
exact; dense 98.1% exact / 99.6% within 1 ULP; ln2 / pos_added / pos_norm ≥ 99.2% exact; soft tokens
93.0% exact / 98.2% within 1 ULP chained, 100% exact teacher-forced from the reference's pos_norm.

**PENDING** — the end-to-end text gate (argmax + 20-token greedy with the image, and the bidirectional
overlay) for the unified family: a full 12B bf16 forward is 23.9 GB and does not fit this 24 GB Mac.
On the 128 GB machine (M4 Max):

```sh
APERTURA_MODEL="/Volumes/Gemma 4/sources/gemma-4-12B-it-qat-q4_0-unquantized" \
APERTURA_FIXTURES=fixtures_vision_unified_12b_full \
~/.venvs/apertura-torch/bin/python Tools/generate_vision_fixtures.py     # needs the Unified model class — see TODO in that script
./AperturaResearch "$SNAP_12B" aptransformerTests/Fixtures/fixtures_vision_unified_12b_full.safetensors --vision-conformance --ulp
```
(The generator currently instantiates `Gemma4ForConditionalGeneration`; for the unified source swap
in `Gemma4UnifiedForConditionalGeneration` + `AutoProcessor` and hook `model.model.embed_vision`
instead of the tower — a small edit, left for the machine that can run it. The driver's unified
branch then needs the text section of the tower gate wired in, which is a copy of the existing block.)

## 5. Switches and capabilities

| Where | What |
|---|---|
| `config.json` `vision_config` | `ESModelConfig::vision.present` — the tower is loaded iff present and the tensors exist |
| `ESWeightLoader::hasVisionTower()` | `embed_vision.embedding_projection.weight` plus the tower's `patch_embedder.input_proj.weight` OR the unified `vision_embedder.patch_dense.weight` |
| `APModel.hasVisionTower` / `supportsImageInput` | the loaded weights carry the tower |
| `+[APModel modelAtURLHasVisionTower:]` | pre-flight: HF index names the tensors (either family; single-file checkpoints assumed complete), or the `.apml` manifest lists `"vision"` in `modalities` |
| `APSession.acceptsImageContent` (public) | Google: always; local: `model.hasVisionTower` |
| `--export` / `--export-lattice` | vision tensors (either family) ride along **bf16** (never quantized); manifest `modalities: ["text","vision"]`; the export streams tensors (each source freed after quantization) so a 24 GB source exports on a 24 GB Mac |
| `AperturaResearch --chat … --image <file> [--image-budget N]` | one image on the user turn, soft tokens prefilled; N ∈ {70, 140, 280, 560, 1120} |
| Visual-token budget | `ESImagePreprocessor::process(…, budget)` / `softTokensForSize(…, budget)` (0 = config default 280; anything outside the five values throws); `APGenerationOptions.visualTokenBudget` per turn (local backend only — Gemini ignores it); `-[APModel imageTokenCountForPixelSize:budget:]`; app: Settings ▸ Generation ▸ Image detail (`AperturaGenVisualTokensKey`). The budget scales the resize target (budget·9 patches); the tower's pooler stays 3×3 (reference derives k from patches/tokens, both scaled). Gated bit-exact for both families × 5 budgets × 3 pictures and at the per-op floor for the E2B tower at 70/560 (`aptransformerTests/ESVisionBudgetTests.mm`, `Tools/generate_budget_fixtures.py`). |
| `AperturaResearch <snap> fixtures_vision_e2b.safetensors --vision-conformance [--ulp]` | the gate |

Not done: audio; vision-tower quantization; the bidirectional block overlay is unverified; images
with alpha are flattened over black rather than alpha-dropped.
