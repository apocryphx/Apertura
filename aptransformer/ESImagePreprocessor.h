#pragma once
//  ESImagePreprocessor — Gemma4ImageProcessor, bit-for-bit: CGImage -> patch grid + positions.
//
//  The reference pipeline (transformers Gemma4ImageProcessor, torchvision backend) is:
//    1. PIL.convert("RGB") -> uint8 CHW tensor.
//    2. Aspect-preserving resize to the largest (h, w) with h*w/16^2 <= 2520 patches and both
//       sides multiples of 48 (get_aspect_ratio_preserving_size). Bicubic, antialias=True:
//       torchvision casts uint8 to float32, runs aten's separable antialiased resampler
//       (PIL-compatible Keys cubic a=-0.5, support 2*scale, normalized weights, W pass then
//       H pass, float32 accumulation), clamps to [0,255], rounds half-to-even, casts to uint8.
//    3. Rescale x * (1/255) in float32. No normalization (the model scales 2(x-0.5) itself).
//    4. Patchify (C,H,W) -> (nh*nw, 16*16*3) with the per-patch vector ordered (py, px, c),
//       patches row-major (y outer, x inner); positions (x, y); pad to 2520 with (-1, -1).
//  This class reproduces 1-4 except the padding: the engine runs the valid patches only
//  (padding is masked out of attention and zeroed before pooling in the reference, so it
//  contributes nothing). The conformance gate checks pixel_values and positions exactly.
//
//  Decoding: ImageIO/CoreGraphics draw the image into an 8-bit RGB bitmap in the image's own
//  RGB colour space (no colour matching, like PIL's byte reinterpretation). Images with alpha
//  are composited over black (PIL drops the alpha channel instead) — a documented deviation
//  that cannot affect opaque photos.
#include "mlx/mlx.h"
#include "ESVisionConfig.h"
#include <CoreGraphics/CoreGraphics.h>
#include <string>
#include <cstdint>
#include <vector>

namespace es {
namespace mx = mlx::core;

struct ESImagePatches {
    mx::array pixelValues;   // float32 [numPatches, 3*p*p], values k/255
    mx::array positionIds;   // int32   [numPatches, 2]  (x, y)
    int patchWidth  = 0;     // nw
    int patchHeight = 0;     // nh
    int numSoftTokens() const;   // (nh/k) * (nw/k); == numPatches() when the patches are merged
    int numPatches() const { return patchWidth * patchHeight; }
    int poolingKernel = 3;
    // Unified family: each row is a merged 48x48x3 block (one soft token); positions are block coordinates.
    bool merged = false;
};

class ESImagePreprocessor {
public:
    explicit ESImagePreprocessor(const ESVisionConfig & config) : cfg_(config) {}

    // Tower family: 16-px patches + (x, y). Unified family (cfg.kind == unified): the processor's
    // patches_merge output — one (3*16)^2*3 row per soft token in plain row-major pixel order
    // (y, x, c) over the 48x48 block, positions (x/3, y/3), tokens row-major over the block grid.
    ESImagePatches process(CGImageRef image) const;
    // Decodes with ImageIO; throws std::runtime_error when the file is not a decodable image.
    ESImagePatches processFile(const std::string & path) const;

    // get_aspect_ratio_preserving_size -> {targetHeight, targetWidth} in pixels.
    std::pair<int, int> targetSize(int height, int width) const;
    // Soft tokens an image of this size will occupy in the prompt (the chat template needs
    // it before any pixels are read).
    int softTokensForSize(int height, int width) const;

    // aten upsample_bicubic2d_aa, uint8 path (PIL integer algorithm) on one plane: [h, w] -> [th, tw].
    static std::vector<uint8_t> resizeBicubicAA(const std::vector<uint8_t> & src, int h, int w, int th, int tw);

private:
    ESVisionConfig cfg_;
};

}  // namespace es
