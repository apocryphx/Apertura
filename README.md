# Apertura

**A from-scratch Objective-C++ / MLX rebuild of Google's Gemma-4 for Apple Silicon — built to be inspected, observed, and experimented with.**

Most language models are run behind glass: you send a prompt and get an answer, with no way to watch the machine think. Apertura is the opposite. It's a faithful, class-for-class, op-for-op re-implementation of the HuggingFace PyTorch `Gemma4TextForCausalLM`, written in readable Objective-C++ on top of [MLX](https://github.com/ml-explore/mlx), that runs the whole Gemma-4 text family **natively and entirely on a Mac** — no cloud, no Python at inference time. Every layer is a small, inspectable object you can trace, freeze mid-forward, quantize, or take apart.

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
| Gemma-4 31B QAT | Quantization-aware-trained | runs faithfully at bf16 |

The faithful Gemma-4 details ported exactly include: hybrid 5:1 local/global attention, dual head_dim (256 local / 512 global), partial RoPE on global layers, QK-norm before RoPE, weightless V-norm with `attention_k_eq_v`, the 4-norm sandwich + per-layer `layer_scalar`, tied embeddings, the bf16-rounded embedding scale, and the final-logit softcap (30.0). See [`aptransformer/MoE_REFERENCE.md`](aptransformer/MoE_REFERENCE.md) for the annotated MoE path and config switches.

## Layout

```
aptransformer/         The MLX compute framework (the model itself)
  ESModelConfig        parses config.json — one config drives the whole family
  ESWeightLoader       sharded safetensors -> mx::array, cast to compute dtype
  ESEmbedding ESLinear ESRMSNorm ESRotaryEmbedding ESMLPBlock
  ESAttention ESKVCache ESRouter ESExperts          attention, cache, MoE
  ESDecoderLayer ESGemma4TextModel ESGemma4TextForCausalLM
  ESSampler ESGenerationLoop                        sampling + prefill/decode loop
  ESConformance                                     fixture loading + deviation stats
AperturaResearch/      Command-line driver
  main.mm              conformance sweep, generation, benchmarks, expert ladder
  ESTokenizer          thin wrapper over ObjCTokenizer
  ESChatTemplate       Gemma-4 chat grammar: roles, reasoning channel, tool calls
aptransformerTests/    XCTest conformance + primitive tests
Tools/                 PyTorch fixture generators (run once, in a torch env)
```

## Requirements

- **macOS on Apple Silicon** (Metal).
- **MLX** — `brew install mlx` (developed against 0.31.2). Headers in `/opt/homebrew/include`, lib `-lmlx`.
- **[ObjCTokenizer](https://github.com/apocryphx/ObjCTokenizer)** — a pure-Objective-C, byte-identical HuggingFace tokenizer, used for encode/decode. Checked out as a sibling directory.
- **Model weights** — a Gemma-4 HuggingFace snapshot (`config.json` + sharded safetensors + `tokenizer.json`).
- **(Conformance only)** a Python env with `torch` + `transformers` to regenerate fixtures from the reference implementation. Not needed to run inference.

## Build

**Xcode** (primary): open `Apertura.xcodeproj` and build the `aptransformer` framework, the `AperturaResearch` CLI target, and the `aptransformerTests` test target. The `aptransformer` folder is a synchronized group, so new `.mm` files are picked up automatically.

**Direct clang** (research/dev build): with MLX in `/opt/homebrew` and `ObjCTokenizer` checked out as a sibling, compile the framework sources, the driver, the tokenizer wrapper, and the prebuilt ObjCTokenizer objects together:

```sh
OCT=../ObjCTokenizer/ObjCTokenizer
clang++ -std=gnu++20 -fobjc-arc -ObjC++ -O2 \
  -I/opt/homebrew/include -Iaptransformer -IAperturaResearch -I"$OCT/.." -I"$OCT" \
  aptransformer/ES*.mm AperturaResearch/main.mm AperturaResearch/ESTokenizer.mm \
  AperturaResearch/ESChatTemplate.mm build/oct/*.o \
  -L/opt/homebrew/lib -lmlx -licucore \
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

Key flags: `--chat` / `--system` / `--think` / `--sample`, `--quant N` / `--quant-embed [N]` / `--quant-kv N`, `--fused`, `--moe-sparse`, `--expert-ladder`, `--generate`, `--decode` / `--prefill`, `--longctx`, `--bench`.

## Features

- **Quantization** — 4/8-bit weights, independent embedding/LM-head bits, and a quantized KV cache.
- **Operator fusion** — `mx::fast` kernels and `mx::compile` for RMSNorm, RoPE, SDPA, GeLU.
- **Sparse MoE routing** — `gather_mm` / `gather_qmm` so only the selected experts are computed.
- **Gemma-4 chat grammar** (`ESChatTemplate`) — turns/roles, the on/off reasoning channel, and tool-call parsing, built at the token-id level to match the reference exactly.
- **Sampling** — greedy plus temperature / top-k / top-p.

Decode is memory-bandwidth-bound: bf16 on the 31B runs at roughly the same throughput as llama.cpp, and the quantization + fusion + sparse-MoE levers scale it up substantially. Numbers depend on the machine.

## Conformance

`Tools/generate_fixtures.py` (run once in a torch env) captures the reference's intermediate tensors and greedy token sequence. `ESConformance` loads them and reports per-op deviation (max / median / p99); the acceptance gate is **exact per-position argmax and greedy token-id match**, with numeric tolerances set to the bf16 floor. Cross-engine divergence at near-ties (Metal/MLX vs MPS/PyTorch vs llama.cpp) is the expected floating-point-order floor, not a correctness gap.

## Acknowledgements

- The HuggingFace `transformers` Gemma-4 reference (`modeling_gemma4.py`) — the authoritative oracle.
- [MLX](https://github.com/ml-explore/mlx) — the Apple Silicon array framework.
- [ObjCTokenizer](https://github.com/apocryphx/ObjCTokenizer) — the tokenizer.

## License

[MIT](LICENSE) © 2026 Kolja Wawrowsky.
