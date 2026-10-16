#pragma once
//  ESVisionConfig — Gemma-4 vision tower hyperparameters + the image placeholder ids.
//
//  Parsed from the HF `config.json` (`vision_config` plus the root-level token ids) by
//  ESModelConfig::fromConfigJSON and carried as ESModelConfig::vision. `present` is false
//  for a text-only checkpoint (no vision_config), and the engine never touches the tower
//  then. The image-processor constants (patch 16, 3x3 pooling, 280 soft tokens) are
//  mirrored in vision_config as patch_size / pooling_kernel_size / default_output_length.
namespace es {

// Two Gemma-4 vision front ends share the placeholder grammar and the soft-token scatter:
//   tower   — Gemma4ForConditionalGeneration (E2B/E4B/…): SigLIP-style encoder, 16-px patches,
//             3x3 pooling (ESGemma4VisionModel's tower path);
//   unified — Gemma4UnifiedForConditionalGeneration (12B QAT): encoder-free, the processor merges
//             3x3 patches into one 48x48x3 vector per soft token and a LayerNorm/Dense/pos-emb
//             stack embeds it (ESUnifiedVisionEmbedder).
enum class ESVisionKind { none, tower, unified };

struct ESVisionConfig {
    bool present = false;
    ESVisionKind kind = ESVisionKind::none;

    // Gemma4VisionConfig
    int   hiddenSize            = 768;
    int   numHiddenLayers       = 16;
    int   numAttentionHeads     = 12;
    int   numKeyValueHeads      = 12;
    int   headDim               = 64;
    int   intermediateSize      = 3072;
    int   patchSize             = 16;
    int   poolingKernelSize     = 3;
    int   positionEmbeddingSize = 10240;   // rows of the learned 2-D position table (per axis)
    int   maxSoftTokens         = 280;     // default_output_length: the per-image patch budget / k^2
    float rmsNormEps            = 1e-6f;
    float ropeTheta             = 100.0f;  // axial 2-D RoPE base
    bool  useClippedLinears     = true;    // QAT: every projection clamps input and output
    bool  standardize           = false;   // (std_bias, std_scale) after pooling — absent on E2B

    // Gemma4UnifiedVisionConfig (kind == unified)
    int mmEmbedDim     = 3840;   // patch_dense output / pos-emb width
    int mmPosembSize   = 1120;   // rows of the factorized position table (per axis)
    int outputProjDims = 3840;   // embedding_projection input width
    float layerNormEps = 1e-5f;  // nn.LayerNorm default (patch_ln1/ln2, pos_norm)
    int modelPatchSize() const { return patchSize * poolingKernelSize; }           // 48
    int modelPatchPixels() const { return 3 * modelPatchSize() * modelPatchSize(); } // 6912

    // Root-level placeholder ids (shared Gemma-4 vocab).
    int imageTokenId = 258880;   // <|image|>   one per soft token
    int boiTokenId   = 255999;   // <|image>    begin-of-image
    int eoiTokenId   = 258882;   // <image|>    end-of-image

    // The processor's patch budget per image: max_soft_tokens * pooling^2 (2520).
    int maxPatches() const { return maxSoftTokens * poolingKernelSize * poolingKernelSize; }
    // Pixels per patch row/col that one pooled token covers (48).
    int tokenSidePixels() const { return poolingKernelSize * patchSize; }
    int patchPixels() const { return 3 * patchSize * patchSize; }   // 768
};

}  // namespace es
