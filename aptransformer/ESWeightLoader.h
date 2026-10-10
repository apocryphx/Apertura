#pragma once
//  ESWeightLoader — HF safetensors (sharded) -> mx::array, cast once to computeDtype.
//
//  Reads model.safetensors.index.json, loads each referenced shard via
//  mx::load_safetensors, strips the `model.language_model.` text-decoder prefix,
//  and casts every tensor to config.computeDtype. Vision/audio weights are ignored.
//  Tied embeddings: there is no lm_head weight; the LM head reuses embed_tokens.
#include "mlx/mlx.h"
#include "ESModelConfig.h"
#include "ESLinear.h"
#include "ESEmbedding.h"
#include "ESExperts.h"
#include <string>
#include <unordered_map>

namespace es {
namespace mx = mlx::core;

class ESWeightLoader {
public:
    // modelDir: the HF snapshot directory (contains config.json, the shards, index json).
    ESWeightLoader(const std::string & modelDir, const ESModelConfig & config);

    bool has(const std::string & name) const { return weights_.count(name) > 0; }

    // Throws if missing. Names are the text-decoder-relative keys, e.g.
    //   "embed_tokens.weight", "norm.weight",
    //   "layers.7.self_attn.q_proj.weight", "layers.7.layer_scalar", ...
    const mx::array & get(const std::string & name) const;

    // Convenience for per-layer weights.
    const mx::array & layer(int idx, const std::string & suffix) const;

    size_t count() const { return weights_.size(); }

    // Read-only view of every loaded (text-decoder-relative) tensor, for bulk
    // operations like quantized-bundle export.
    const std::unordered_map<std::string, mx::array> & all() const { return weights_; }

    // --- .apml bundle (reload) mode ---------------------------------------
    // True when this loader was built from an .apml package (pre-quantized variant)
    // rather than an HF snapshot. In bundle mode tensors are stored verbatim
    // (packed weights stay uint32 — never cast to computeDtype).
    bool isBundle() const { return isBundle_; }
    // A tensor is pre-quantized iff its companion `<name>.scales` is present.
    bool hasQuantized(const std::string & name) const { return weights_.count(name + ".scales") > 0; }
    struct QuantTriple { mx::array weight, scales, biases; };
    QuantTriple quantized(const std::string & name) const;
    int bundleBits() const { return bundleBits_; }
    int bundleGroupSize() const { return bundleGroupSize_; }
    int bundleEmbedBits() const { return bundleEmbedBits_; }
    int bundlePleBits() const { return bundlePleBits_; }      // 0 for pre-ple bundles (table bf16)

    std::string layerKey(int idx, const std::string & suffix) const {
        return "layers." + std::to_string(idx) + "." + suffix;
    }

private:
    void loadHF(const std::string & modelDir, const ESModelConfig & config);
    void loadBundle(const std::string & packageDir, const ESModelConfig & config);

    std::unordered_map<std::string, mx::array> weights_;
    bool isBundle_         = false;
    int  bundleBits_       = 0;
    int  bundleGroupSize_  = 64;
    int  bundleEmbedBits_  = 0;
    int  bundlePleBits_    = 0;
};

// --- Layer factories --------------------------------------------------------
// Build a layer for `name`: pre-quantized from the bundle when present, else the
// bf16 path (quantize-now iff the config bits > 0). These are the single point
// that routes the reload-vs-quantize-now decision, so construction sites stay simple.
ESLinear    esMakeLinear   (const ESWeightLoader & w, const std::string & name,
                            int quantBits, int groupSize);
ESEmbedding esMakeEmbedding (const ESWeightLoader & w, const std::string & name,
                            int quantEmbedBits, int groupSize);
// Elastic per-layer embedding table (embed_tokens_per_layer): same reload-vs-quantize-now
// routing as the token embedding, but keyed on the bundle's `ple_bits` / config.quantPleBits.
ESEmbedding esMakePleTable(const ESWeightLoader & w, const std::string & name,
                           int quantPleBits, int groupSize);
ESExperts   esMakeExperts  (const ESWeightLoader & w, const std::string & gateUpName,
                            const std::string & downName, int quantBits, int groupSize);

// --- Quantized .apml bundle export -----------------------------------------
//
// Quantize an HF model snapshot and write a self-describing `.apml` package (a
// macOS document package — see BUNDLE.md). Quantizes exactly the projections the
// runtime quantizes (q/k/v/o, gate/up/down, MoE experts) at `bits`, the token
// embedding at `embedBits`, the elastic per-layer embedding table at `pleBits`, and
// leaves norms/scalars/router weights bf16.
// The package is assembled in a temp directory and moved into place atomically.
struct ESBundleExportOptions {
    int bits       = 4;          // layer-projection quant bits (0 = keep bf16)
    int groupSize  = 64;         // affine group size
    int embedBits  = 8;          // token-embedding / tied-head bits (0 = keep bf16)
    int pleBits    = 8;          // elastic per-layer embedding table bits (0 = keep bf16; no-op on dense)
    // QAT lattice-exact mode (see quantizeQ4Lattice below). Forces bits=4, groupSize=32. Every
    // quantized tensor that sits on the QAT int4 lattice (Google's `*-qat-q4_0-unquantized`
    // checkpoints) is stored with the TRAINED step as the affine scale, so the bundle dequantizes
    // to the QAT weights exactly; a tensor off the lattice falls back to mx::quantize at the
    // requested bits and is reported. embed/ple tensors on the lattice are stored at 4 bits
    // (exact) regardless of embedBits/pleBits, and the manifest records the bits actually written.
    bool lattice   = false;
    std::string variantId      = "mlx-q4";
    std::string sourceModelId;   // provenance (optional)
    std::string sourceRevision;  // provenance (optional)
};

// Returns true on success. On failure returns false and, if `error` is non-null,
// sets it to a human-readable message.
bool exportQuantizedBundle(const std::string & modelDir,
                           const std::string & outPackagePath,
                           const ESBundleExportOptions & opts,
                           std::string * error = nullptr);

// --- QAT lattice-exact quantization (int4, group 32, learned step) ---------------
//
// Google's Gemma 4 QAT checkpoints (`*-qat-q4_0-unquantized`) are bf16 weights that already
// sit on an int4 lattice: per 32-element block along the input dim, w = bf16(k * d) with codes
// k in [-8, 7] and a per-block step d that is the QAT-LEARNED scale (not absmax/-8 as in ggml
// q4_0, and not absmax/7 — the extreme code present varies per block). Re-quantizing that with
// a min/max affine quantizer (mx::quantize) picks a different scale and zero point and lands
// most weights off their trained values (llama.cpp's naive Q4_0 conversion matches only ~25%
// of bytes — Unsloth, Gemma 4 QAT notes). quantizeQ4Lattice recovers d per block from the
// lattice structure and emits MLX's ordinary affine format with scale = d, bias = -8*d,
// code = k + 8 — group_size 32, 4 bits — so mx::dequantize / quantized_matmul reproduce k*d
// in float, with no loader or kernel change.
//
// ESLatticeFit reports how well the tensor fits: `exact` counts weights whose k*d rounds to
// the original bf16 bit pattern, `near` those within one bf16 ulp (|err| <= |w|/128),
// `maxAbsErr` the worst reconstruction error. Because the checkpoint stores bf16 roundings of
// k*d, a single bf16 step cannot hit every element: ~90.5% exact / 100% near is the ceiling on
// the 31B (i.e. the Q4 weights are as close to the trained lattice as the bf16 checkpoint is).
// A tensor NOT on the lattice (plain post-training checkpoint) fits poorly; exportQuantizedBundle
// falls back to mx::quantize when near/total < 0.999.
struct ESLatticeFit {
    uint64_t total = 0, exact = 0, near = 0;
    float    maxAbsErr = 0.f;
    double exactFrac() const { return total ? (double) exact / (double) total : 0.0; }
    double nearFrac()  const { return total ? (double) near  / (double) total : 0.0; }
};

// Returns {w_q (uint32, [..., in/8]), scales (w.dtype, [..., in/32]), biases (same)} —
// the triple ESLinear/ESEmbedding/ESExperts adopt verbatim with bits=4, groupSize=32.
// Throws if the last dimension is not a multiple of 32. The returned arrays are lazy; the
// fit (if requested) is evaluated eagerly.
std::vector<mx::array> quantizeQ4Lattice(const mx::array & w, ESLatticeFit * fit = nullptr);

// Report-only: loads an HF snapshot and prints, per quantizable tensor class, how much of it
// sits on the QAT int4 lattice. Run this on a checkpoint BEFORE exporting with `lattice`.
bool scanQ4Lattice(const std::string & modelDir, std::string * error = nullptr);

// Weight-level gate for an exported bundle: dequantizes every quantized tensor and compares it
// to the source checkpoint, printing the same per-class table. PASS iff >= 99.9% of weights are
// within one bf16 ulp of the source (any recipe; a lattice bundle should also show ~90% exact).
bool verifyLatticeBundle(const std::string & modelDir, const std::string & apml, std::string * error = nullptr);

}  // namespace es
