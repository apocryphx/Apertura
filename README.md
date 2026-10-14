# Apertura

**A from-scratch Objective-C++ / MLX rebuild of Google's Gemma-4 for Apple Silicon — built to be inspected, observed, and experimented with.**

Most language models are run behind glass: you send a prompt and get an answer, with no way to watch the machine think. Apertura is the opposite. It's a faithful, class-for-class, op-for-op re-implementation of the HuggingFace PyTorch `Gemma4TextForCausalLM`, written in readable Objective-C++ on top of [MLX](https://github.com/ml-explore/mlx), that runs the whole Gemma-4 text family **natively and entirely on a Mac** — no cloud, no Python at inference time. Every layer is a small, inspectable object you can trace, freeze mid-forward, quantize, or take apart. (The `Apertura` app also offers an optional cloud backend over the Gemini API for comparison — see Features — but the engine itself never leaves the machine.)

It is a research instrument first and a runtime second: correctness is gated by **bit-exact conformance against the PyTorch reference**, not by vibes.

---

## What it is

- **Faithful.** A clean rewrite from `modeling_gemma4.py`, conformance-tested op-by-op against PyTorch. Greedy generation matches the reference token-for-token down to the floating-point-order floor (e.g. 80/80 and 89/89 matching tokens on real prompts before the first bf16 near-tie diverges).
- **Universal.** One codebase plays the **entire Gemma-4 text family**, switched by `config.json` alone — no code changes.
- **Local.** Runs offline on Apple Silicon via MLX. A 31B-parameter model holds a conversation on a laptop.
- **Observable.** A conformance trace exposes the scaled embedding, every decoder-layer output, and the final norm; the CLI driver lets you sweep experts, coarsen precision, and toggle reasoning, then watch what changes.

## Supported models

All four architectures are verified to match the PyTorch reference (argmax + greedy) at bf16, selectable purely by the model's `config.json`:

| Model | Architecture | Notes |
|---|---|---|
| Gemma-4 31B | Dense | 60 layers, hybrid local/global attention |
| Gemma-4 26B | Mixture-of-Experts | 128 experts, top-8 routing (dense or sparse path) |
| Gemma-4 E2B / E4B | Elastic | Per-Layer Embeddings (PLE) + shared-KV layers |
| Gemma-4 QAT (31B, 12B, 26B, E2B, E4B) | Quantization-aware-trained | run at bf16, or as **lattice-exact Q4 bundles** (below) |

The faithful Gemma-4 details ported exactly include: hybrid 5:1 local/global attention, dual head_dim (256 local / 512 global), partial RoPE on global layers, QK-norm before RoPE, weightless V-norm with `attention_k_eq_v`, the 4-norm sandwich + per-layer `layer_scalar`, tied embeddings, the bf16-rounded embedding scale, and the final-logit softcap (30.0). See [`aptransformer/MoE_REFERENCE.md`](aptransformer/MoE_REFERENCE.md) for the annotated MoE path and config switches.

## Layout

```
aptransformer/          The MLX compute framework (the model itself) — core ES* classes
  ESModelConfig          parses config.json — one config drives the whole family
  ESExecutionProfile     per-family execution policy (chunk size, prefill skips, MoE sort, warm-up)
  ESWeightLoader         sharded safetensors -> mx::array, cast to compute dtype
  ESEmbedding ESLinear ESRMSNorm ESRotaryEmbedding ESMLPBlock
  ESAttention ESKVCache ESRouter ESExperts          attention, cache, MoE
  ESDecoderLayer ESGemma4TextModel ESGemma4TextForCausalLM
  ESSampler ESGenerationLoop                        sampling + prefill/decode loop
  ESConformance                                     fixture loading + deviation stats
AperturaKit/             App-facing SDK framework — a pure Objective-C surface; no C++ types cross the boundary
  APSession              abstract session: transcript, tool registry, delegate callbacks, reasoning toggle
  APLocalSession         on-device backend — wraps the engine's persistent KV-cache session, snapshot restore
  APGoogleSession        cloud backend — Gemma over the Gemini API (streaming, native function calling)
  APModel                loads/holds weights, wraps aptransformer's model classes for the local backend
  APMessage APResponse APContent APGenerationOptions APModelConfiguration APTool …   session value types
  ESTokenizer ESChatTemplate   tokenizer wrapper + Gemma-4 chat grammar (shared by AperturaKit and AperturaResearch)
Apertura/                macOS chat app built on AperturaKit
  ViewController.m       model load, persona priming, streaming replies, local/cloud backend switch, Keychain
  Core Data/CDChatSession   Core Data + CloudKit persisted transcript (capture-only, versioned JSON)
AperturaResearch/        Command-line driver, reusing AperturaKit's core headers
  main.mm                conformance sweep, generation, benchmarks, expert ladder
aptransformerTests/      XCTest conformance + primitive tests
AperturaKitTests/        Unit tests for APLocalSession, APGoogleSession, APMessage, APGenerationOptions, ...
AperturaTests/           Xcode-generated app test stubs (not yet filled in)
AperturaUITests/         Xcode-generated UI test stubs (not yet filled in)
Tools/                   PyTorch fixture generators (run once, in a torch env)
```

`aptransformer`, `AperturaKit`, and `AperturaResearch` each compile the shared `ES*` core sources directly as their own target sources — they don't link a shared `aptransformer.framework`. Editing an `ES*` file affects all three targets' next build; there's no separate library to relink.

## Requirements

- **macOS on Apple Silicon** (Metal), with Xcode's **Metal Toolchain** component installed (`xcodebuild -downloadComponent MetalToolchain`) — without it MLX can't compile its kernels and drops the M5/M6 Neural Accelerator path (see [GPU Neural Accelerators](#gpu-neural-accelerators-m5--m6)).
- **[MLX](https://github.com/ml-explore/mlx)** — plain upstream, no fork, pinned to the commit in `Tools/mlx.pin` (the single source of truth; bump it deliberately after validating a newer upstream commit — an unpinned experiment was reverted on day one when upstream `main` failed to compile against the macOS 27 SDK). Checked out as a sibling directory (`../mlx`) and built locally as a static library; `sh Tools/build_mlx.sh` does clone + checkout of the pin + the CMake build (`cmake -B build -DBUILD_SHARED_LIBS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF && cmake --build build --target mlx`), producing `build/libmlx.a` and `build/mlx/backend/metal/kernels/mlx.metallib`. It never touches a dirty checkout away from the pin, so local experiments survive; `--status` reports pin-vs-checkout drift. The Xcode project points at it via `MLX_ROOT`/`MLX_BUILD`/`MLX_METALLIB`. **Do not rely on `brew install mlx`** — a Homebrew mlx install could shadow the local checkout's headers while `-lmlx` links the local library, producing ABI-mismatch link errors; Homebrew's mlx isn't a build dependency of this project. (A fork was maintained here previously for iOS/Catalyst XCFramework packaging scripts and a smoke test; both were dropped once Apertura settled on macOS-only — the smoke test now lives at `Tools/MLXTest.h`/`.mm`.)
- **[ObjCTokenizer](https://github.com/apocryphx/ObjCTokenizer)** — a pure-Objective-C, byte-identical HuggingFace tokenizer, used for encode/decode. Checked out as a sibling folder, `../ObjCTokenizer` (shared with ES-Archive; `sh Tools/bootstrap.sh` clones it), and referenced by `Apertura.xcodeproj` as a subproject.
- **Model weights** — a Gemma-4 HuggingFace snapshot (`config.json` + sharded safetensors + `tokenizer.json`).
- **(Conformance only)** a Python env with `torch` + `transformers` to regenerate fixtures from the reference implementation. Not needed to run inference.

## Build

**Xcode** (primary): open `Apertura.xcworkspace` or the bare `Apertura.xcodeproj` (it references `../ObjCTokenizer` as a subproject, so either builds) and build whichever target you need: `aptransformer` (the compute framework alone), `AperturaResearch` (the CLI driver), `AperturaKit` (the app SDK) + `Apertura` (the macOS app), or the `*Tests` targets. The `aptransformer` folder is a synchronized group, so new `.mm` files are picked up automatically. Requires `../mlx` built first (`sh Tools/build_mlx.sh`, see Requirements above). Command-line Release builds need `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES` — `libmlx.a` is arm64-only and a universal link fails on x86_64.

**Direct clang** (research/dev build): with `mlx` and `ObjCTokenizer` checked out as siblings and `mlx` built, compile the framework sources, the driver, the tokenizer wrapper, and the prebuilt ObjCTokenizer objects together:

```sh
MLX=../mlx
OCT=../ObjCTokenizer/ObjCTokenizer
clang++ -std=gnu++20 -fobjc-arc -ObjC++ -O2 \
  -I"$MLX" -Iaptransformer -IAperturaKit -I"$OCT/.." -I"$OCT" \
  aptransformer/ES*.mm AperturaResearch/main.mm AperturaKit/ESTokenizer.mm \
  AperturaKit/ESChatTemplate.mm build/oct/*.o \
  -L"$MLX/build" -lmlx -licucore \
  -framework Foundation -framework Metal -framework Accelerate \
  -framework QuartzCore -framework MetalPerformanceShaders \
  -o build/AperturaResearch
```

## Run

The driver takes a model snapshot directory and a mode. A few examples:

```sh
SNAP=~/.cache/huggingface/hub/models--google--gemma-4-31b-it/snapshots/<hash>

# Chat (reasoning off): build a Gemma-4 prompt, generate, parse the answer
./build/AperturaResearch "$SNAP" --chat "Name three primary colors." --decode 40

# Chat with the reasoning channel exposed
./build/AperturaResearch "$SNAP" --think --chat "A bat and ball cost \$1.10..." --decode 400

# Quantized inference (4-bit weights, 8-bit embedding) + operator fusion
./build/AperturaResearch "$SNAP" --quant 4 --quant-embed 8 --fused --generate "..." 200

# Experiment: sweep an MoE model's active experts 128 -> 4 and watch the output shift
./build/AperturaResearch "$SNAP_26B" --expert-ladder /path/to/prompt_ids.safetensors

# Conformance + throughput
./build/AperturaResearch "$SNAP" /path/to/fixtures.safetensors    # per-op + argmax/greedy gate
./build/AperturaResearch "$SNAP" --bench --prefill 512 --decode 128
```

Key flags: `--chat` / `--system` / `--think` / `--sample`, `--quant N` / `--quant-embed [N]` / `--quant-kv N`, `--fused`, `--moe-sparse`, `--expert-ladder`, `--generate`, `--decode` / `--prefill`, `--longctx`, `--bench`, `--export <o.apml>` / `--export-lattice` / `--lattice-scan`.

## Features

- **Quantization** — 4/8-bit weights, independent embedding/LM-head bits, and a quantized KV cache. QAT checkpoints (`google/*-qat-q4_0-unquantized`) export **lattice-exact** (`--export-lattice`): the bundle's scales are the QAT-learned int4 steps, so the Q4 weights match the checkpoint to bf16 precision (see [`BUNDLE.md`](BUNDLE.md) and the fidelity table below).
- **Operator fusion** — `mx::fast` kernels and `mx::compile` for RMSNorm, RoPE, SDPA, GeLU.
- **Sparse MoE routing** — `gather_mm` / `gather_qmm` so only the selected experts are computed.
- **Gemma-4 chat grammar** (`ESChatTemplate`) — turns/roles, the on/off reasoning channel, and tool-call parsing, built at the token-id level to match the reference exactly.
- **Sampling** — greedy plus temperature / top-k / top-p.
- **App** — `AperturaKit` exposes a pure Objective-C `APSession` contract with two swappable backends: `APLocalSession` (on-device, the engine above) and `APGoogleSession` (Gemma over the Gemini API). The `Apertura` macOS app is a chat UI on top of it, with Core Data + CloudKit transcript persistence and Keychain-backed API keys for the cloud backend.

Decode is memory-bandwidth-bound: bf16 on the 31B runs at roughly the same throughput as llama.cpp, and the quantization + fusion + sparse-MoE levers scale it up substantially. Numbers depend on the machine. As of 2026-07-21 the Q4 engine measures at **llama.cpp parity on both decode and prefill** at practical context lengths (94-99.5% decode, prefill parity through ~10K-token prompts), with no custom Metal kernels — measured standing, per-lever record, and the benchmark methodology (thermal gating, process hygiene) live in [`aptransformer/PERFORMANCE_ROADMAP.md`](aptransformer/PERFORMANCE_ROADMAP.md).

## Fidelity of the published Q4 bundles

Google's Gemma 4 QAT checkpoints are bf16 weights that already sit on an int4 lattice (per
32-block, `w = bf16(k·d)`, `k ∈ [-8, 7]`, `d` the QAT-learned step). A generic min/max quantizer
run over them is a *second*, misaligned quantization — the failure Unsloth documented for naive
Q4_0 GGUFs. Apertura's `--export-lattice` recovers `d` per block and stores it as the affine
scale, so the Q4 bundle *is* the trained lattice to bf16 precision. Every published bundle
(`apocryphx/gemma-4-*-apml`) has used this recipe since 2026-10-09; measured against each
model's bf16 QAT checkpoint on an M4 Max:

| Model | weights bit-exact | within 1 bf16 ulp | top-1 vs bf16, 2176-tok prompt ×256 | 9907-tok prompt ×256 |
|---|---:|---:|---:|---:|
| 31B | 90.7% | 100% | 255/256 | **256/256** |
| 12B | 90.3% | 100% | 254/256 | 252/256 |
| 26B-A4B | 93.5% | 100% | 254/256 | 250/256 |
| E2B | 90.4% | 100% | 251/256 | 254/256 |
| E4B | 90.4% | 100% | 251/256 | 252/256 |

The weights that are not bit-identical differ by one bf16 rounding step — the checkpoint stores
bf16 roundings of `k·d`, so no single bf16 step can hit every element; this is the ceiling for a
4-bit affine format. The previous affine group-64 bundles measured 12-28% bit-exact and 87-95%
top-1 on the same prompts; the lattice recipe costs ~5% decode throughput vs group 64 on the 31B.
Top-1 is teacher-forced: both models see the bf16 answer as context, so 256/256 means free greedy
generation would reproduce bf16's answer exactly. Full tables and method: roadmap §1d.

## GPU Neural Accelerators (M5 / M6)

Apple's M5-generation GPUs and later carry matrix units ("Neural Accelerators") in every GPU
core; MLX enables them on Metal 4 at GPU generation ≥ 17 (the M6 reports `applegpu_g18g`). MLX drives them through its NAX kernels — quantized matmul, GEMM and fused attention — and
picks them at runtime (`is_nax_available()`), so the engine needs no code change: the pinned MLX
already has them. Measured 2026-10-13 on an **Apple M6** (12-core GPU, 24 GB), same binary
linked against two builds of the pinned MLX, NAX on vs forced off (`-DMLX_METAL_NO_NAX`);
`--bench-eager`, one arm per process, cold-gated (die ≤ 52 °C), prefill tok/s:

| Model | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|
| E2B | 2218 vs 1541 (1.4×) | 3788 vs 2355 (1.6×) | 5154 vs 2591 (2.0×) | 5932 vs 2719 (2.2×) | 6237 vs 2689 (2.3×) | |
| E4B | 1107 vs 676 (1.6×) | 1795 vs 793 (2.3×) | 2072 vs 824 (2.5×) | 2252 vs 835 (2.7×) | 2317 vs 830 (2.8×) | |
| 12B | | 431 vs 186 (2.3×) | | 591 vs 186 (3.2×) | | 576 vs 180 (3.2×) |
| 26B-A4B | | | | | 626 vs 391 (1.6×) | |

E2B/E4B include the shared-KV prefill skip and 26B-A4B the expert-sorted MoE prefill (both
below); before them, E2B measured 2477 vs 943 @2048 and 26B-A4B ~140 tok/s at every length.
26B-A4B is 15.8 GB on a 24 GB machine — only 4K is reported, shorter prompts were dominated by
swap and a fixed per-run cost.

**Prefill skips that matter more than kernels.** (1) *Shared-KV layers* (E2B 20/35, E4B 18/42)
write no cache, so in prefill only the last position needs them — `lastLogits` runs them on one
row (E2B 2477 → 5932 tok/s @2048; greedy output identical, fixtures pass). mlx-lm gets the same
effect implicitly from lazy evaluation, which is why its benchmark reports ~9.4K tok/s for E2B on
this M6 while its full forward runs at ~2.7K. (2) *MoE prefill sorts (token, expert) pairs by
expert* before `gather_qmm(sorted_indices=true)`, turning seq×k matvecs into one grouped GEMM per
expert (26B-A4B ~140 → 626 tok/s @4096; the sorted kernel is also closer to an fp32 reference,
rel-L2 4.7e-3 vs 7.9e-3). (3) *Middle prefill chunks skip the shared-KV layers entirely* (only
the final chunk's last row needs them), and elastic models default to 1024-token chunks (dense
and MoE stay at 512). Warm (`APERTURA_BENCH_WARM=1`), median of 3: E2B 7973 / 7783 / 7065 tok/s
@2K / 4K / 8K, E4B 2702 / 2660 / 2500. The table above is cold single-process runs, which include
MLX's one-time pipeline-state builds (~150 ms of GPU idle in a Metal System Trace) — the app
should run a short warm-up prefill at model load.

Decode is unchanged (E2B 68-78, E4B 36-41, 12B 15-18 tok/s in both builds): single-token decode
runs the matrix-vector kernels, which are bandwidth-bound and don't use the accelerators.
Fidelity is unchanged too: E2B `--vs-bf16` 115/116 + 255/256 (1653-tok prompt) with NAX vs
116/116 + 253/256 without, and fixture conformance against the E2B QAT checkpoint passes 14/14
gates (unfused and fused) on both. A kernel probe against an fp32 CPU reference shows NAX
quantized matmul with identical error to the classic kernel and NAX attention at the bf16
rounding floor (rel-L2 1.6e-3 vs 3.6e-3). Greedy outputs of the two builds fork only at bf16
near-ties. The very first NAX prefill in a process pays a one-time kernel JIT (E2B @256: 0.6 s
instead of 0.16 s) — warm up once at model load.

**Build trap:** MLX compiles the NAX kernels only when the Metal compiler works and the
deployment target is ≥ 26.2. If Xcode's Metal Toolchain component is missing (it is a separate
download in recent Xcode: `xcodebuild -downloadComponent MetalToolchain`), CMake prints a warning
and silently builds **without** NAX — a working but 2-3× slower prefill on M5/M6.

## Conformance

**Machine check:** `Tools/smoke.sh [bundle.apml]` (default: the E2B bundle) verifies a machine end to end in about a minute — Metal toolchain, MLX at the pin with NAX compiled in, the CLI build, the AperturaKit tests with a real model (session + checkpoint round trip), snapshot restore byte-identity (`--persist-verify`), PyTorch fixture conformance when the fixture and its source are present, and a warm benchmark. Run it after setting up a machine, bumping MLX, or touching the engine.

`Tools/generate_fixtures.py` (run once in a torch env) captures the reference's intermediate tensors and greedy token sequence. `ESConformance` loads them and reports per-op deviation (max / median / p99); the acceptance gate is **exact per-position argmax and greedy token-id match**, with numeric tolerances set to the bf16 floor. Cross-engine divergence at near-ties (Metal/MLX vs MPS/PyTorch vs llama.cpp) is the expected floating-point-order floor, not a correctness gap.

## Acknowledgements

- The HuggingFace `transformers` Gemma-4 reference (`modeling_gemma4.py`) — the authoritative oracle.
- [MLX](https://github.com/ml-explore/mlx) — the Apple Silicon array framework.
- [ObjCTokenizer](https://github.com/apocryphx/ObjCTokenizer) — the tokenizer.
- [Gemini API](https://ai.google.dev/) — the optional cloud backend behind `APGoogleSession`.

## License

[MIT](LICENSE) © 2026 Kolja Wawrowsky.
