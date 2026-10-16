#include "ESImagePreprocessor.h"

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace es {

int ESImagePatches::numSoftTokens() const {
    if (merged) return numPatches();
    return (patchHeight / poolingKernel) * (patchWidth / poolingKernel);
}

bool ESImagePreprocessor::isSupportedBudget(int budget) {
    for (int b : kSupportedBudgets) if (b == budget) return true;
    return false;
}

int ESImagePreprocessor::resolveBudget(int budget) const {
    if (budget == 0) budget = cfg_.maxSoftTokens;
    if (!isSupportedBudget(budget))
        throw std::invalid_argument("ESImagePreprocessor: unsupported visual token budget " + std::to_string(budget)
                                    + " (supported: 70, 140, 280, 560, 1120)");
    return budget;
}

// ── get_aspect_ratio_preserving_size ────────────────────────────────────────────────────────
std::pair<int, int> ESImagePreprocessor::targetSize(int height, int width, int budget) const {
    const int patch = cfg_.patchSize, pool = cfg_.poolingKernelSize;
    const int maxPatches = resolveBudget(budget) * pool * pool;
    const double totalPx  = (double) height * (double) width;
    const double targetPx = (double) maxPatches * (double) (patch * patch);
    const double factor   = std::sqrt(targetPx / totalPx);
    const double idealH = factor * height, idealW = factor * width;
    const int sideMult = pool * patch;
    int th = (int) std::floor(idealH / sideMult) * sideMult;
    int tw = (int) std::floor(idealW / sideMult) * sideMult;
    if (th == 0 && tw == 0)
        throw std::runtime_error("ESImagePreprocessor: image degenerates to 0x0 patches");
    const int maxSide = (maxPatches / (pool * pool)) * sideMult;
    if (th == 0) {
        th = sideMult;
        tw = std::min((int) std::floor((double) width / height) * sideMult, maxSide);
    } else if (tw == 0) {
        tw = sideMult;
        th = std::min((int) std::floor((double) height / width) * sideMult, maxSide);
    }
    if ((double) th * tw > targetPx)
        throw std::runtime_error("ESImagePreprocessor: resized image exceeds the patch budget");
    return {th, tw};
}

int ESImagePreprocessor::softTokensForSize(int height, int width, int budget) const {
    auto [th, tw] = targetSize(height, width, budget);
    const int k = cfg_.poolingKernelSize;
    return (th / cfg_.patchSize * (tw / cfg_.patchSize)) / (k * k);
}

// ── aten separable antialiased bicubic, uint8 path ──────────────────────────────────────────
// torchvision hands the uint8 CHW image straight to aten on this platform, and aten's uint8
// resampler is the PIL(-SIMD) integer algorithm: weights are computed in double
// (HelperInterpBase::_compute_indices_min_size_weights_aa<double> with the Keys cubic a=-0.5),
// scaled to int16 at a per-pass precision chosen so the largest weight fits 15 bits, and each
// output byte is (2^(prec-1) + sum(src * w)) >> prec, clamped to [0, 255]. Horizontal pass
// first, then vertical, with a uint8 intermediate. The float32 separable path (what a cast
// would have produced) rounds differently on ~1% of pixels, so it is not used.
namespace {

inline double aaFilterKeys(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1;        // cubic_convolution1
    if (x < 2.0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;  // cubic_convolution2
    return 0.0;
}

struct AAWeights {
    std::vector<int64_t> xmin, xsize;
    std::vector<int16_t> w;      // [outputSize * maxInterp]
    int maxInterp = 0;
    unsigned precision = 0;
};

AAWeights computeAAWeightsInt16(int64_t inputSize, int64_t outputSize) {
    const double scale = (double) inputSize / outputSize;   // area_pixel_compute_scale<double>
    const int interpSize = 4;
    const double support = (scale >= 1.0) ? (interpSize * 0.5) * scale : interpSize * 0.5;
    const int maxInterp = (int) std::ceil(support) * 2 + 1;
    std::vector<double> wd((size_t) outputSize * maxInterp, 0.0);
    AAWeights r;
    r.maxInterp = maxInterp;
    r.xmin.resize(outputSize); r.xsize.resize(outputSize);
    double wtMax = 0.0;
    for (int64_t i = 0; i < outputSize; ++i) {
        double center   = scale * (i + 0.5);
        double totalW   = 0.0;
        double invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
        int64_t xmin  = std::max(static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
        int64_t xsize = std::min(static_cast<int64_t>(center + support + 0.5), inputSize) - xmin;
        xsize = std::clamp(xsize, static_cast<int64_t>(0), (int64_t) maxInterp);
        double * wt = &wd[(size_t) i * maxInterp];
        int64_t j = 0;
        for (; j < xsize; ++j) {
            double w = aaFilterKeys((j + xmin - center + 0.5) * invscale);
            wt[j] = w;
            totalW += w;
        }
        if (totalW != 0.0) {
            for (j = 0; j < xsize; ++j) { wt[j] /= totalW; wtMax = std::max(wtMax, wt[j]); }
        }
        r.xmin[i] = xmin; r.xsize[i] = xsize;
    }
    // _compute_index_ranges_int16_weights: precision so that the largest weight fits int16.
    unsigned precision = 0;
    for (precision = 0; precision < 22; ++precision) {
        int next = (int) (0.5 + wtMax * (1 << (precision + 1)));
        if (next >= (1 << 15)) break;
    }
    r.precision = precision;
    r.w.resize(wd.size());
    for (size_t k = 0; k < wd.size(); ++k) {
        double v = wd[k] * (1 << precision);
        r.w[k] = (int16_t) ((v < 0) ? (int) (-0.5 + v) : (int) (0.5 + v));
    }
    return r;
}

inline uint8_t aaAccumulate(const uint8_t * s, int64_t stride, const int16_t * wt, int64_t xsize, unsigned prec) {
    int out = 1 << (prec - 1);   // the +0.5 of the rounding
    for (int64_t j = 0; j < xsize; ++j) out += (int) s[(size_t) j * stride] * (int) wt[j];
    return (uint8_t) std::clamp(out >> prec, 0, 255);
}

}  // namespace

std::vector<uint8_t> ESImagePreprocessor::resizeBicubicAA(const std::vector<uint8_t> & src, int h, int w,
                                                          int th, int tw) {
    // Horizontal pass (W), then vertical (H) — upsample_separable_Nd_kernel_impl's order.
    std::vector<uint8_t> tmp((size_t) h * tw);
    {
        AAWeights ww = computeAAWeightsInt16(w, tw);
        for (int y = 0; y < h; ++y) {
            const uint8_t * row = &src[(size_t) y * w];
            for (int i = 0; i < tw; ++i)
                tmp[(size_t) y * tw + i] = aaAccumulate(row + ww.xmin[i], 1, &ww.w[(size_t) i * ww.maxInterp],
                                                        ww.xsize[i], ww.precision);
        }
    }
    std::vector<uint8_t> dst((size_t) th * tw);
    {
        AAWeights wh = computeAAWeightsInt16(h, th);
        for (int i = 0; i < th; ++i)
            for (int x = 0; x < tw; ++x)
                dst[(size_t) i * tw + x] = aaAccumulate(&tmp[(size_t) wh.xmin[i] * tw + x], tw,
                                                        &wh.w[(size_t) i * wh.maxInterp], wh.xsize[i], wh.precision);
    }
    return dst;
}

// ── CGImage -> RGB8 ────────────────────────────────────────────────────────────────────────────
static std::vector<uint8_t> rgb8FromCGImage(CGImageRef image, int & width, int & height) {
    width  = (int) CGImageGetWidth(image);
    height = (int) CGImageGetHeight(image);
    if (width <= 0 || height <= 0) throw std::runtime_error("ESImagePreprocessor: empty image");

    // Draw in the image's own RGB space so CoreGraphics performs no colour matching; non-RGB
    // sources (gray, indexed, CMYK) are converted to sRGB.
    CGColorSpaceRef srcSpace = CGImageGetColorSpace(image);
    CGColorSpaceRef space = (srcSpace && CGColorSpaceGetModel(srcSpace) == kCGColorSpaceModelRGB)
                                ? CGColorSpaceRetain(srcSpace)
                                : CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    const size_t rowBytes = (size_t) width * 4;
    std::vector<uint8_t> rgba(rowBytes * height, 0);
    CGContextRef ctx = CGBitmapContextCreate(rgba.data(), width, height, 8, rowBytes, space,
                                             kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(space);
    if (!ctx) throw std::runtime_error("ESImagePreprocessor: cannot create bitmap context");
    CGContextSetInterpolationQuality(ctx, kCGInterpolationNone);
    CGContextSetBlendMode(ctx, kCGBlendModeCopy);
    CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), image);
    CGContextRelease(ctx);

    std::vector<uint8_t> rgb((size_t) width * height * 3);
    for (size_t p = 0, n = (size_t) width * height; p < n; ++p) {
        rgb[p * 3 + 0] = rgba[p * 4 + 0];
        rgb[p * 3 + 1] = rgba[p * 4 + 1];
        rgb[p * 3 + 2] = rgba[p * 4 + 2];
    }
    return rgb;
}

ESImagePatches ESImagePreprocessor::process(CGImageRef image, int budget) const {
    budget = resolveBudget(budget);
    int w = 0, h = 0;
    std::vector<uint8_t> rgb = rgb8FromCGImage(image, w, h);
    auto [th, tw] = targetSize(h, w, budget);
    const int k = cfg_.poolingKernelSize;
    // Unified: the 3x3-merged patch IS the unit (48 px); tower: 16-px patches, pooled in the model.
    const bool merged = cfg_.kind == ESVisionKind::unified;
    const int p = merged ? cfg_.modelPatchSize() : cfg_.patchSize;
    const int nh = th / p, nw = tw / p;

    // Per-channel uint8 planes, resized with aten's integer resampler (identity when no resize).
    std::vector<std::vector<uint8_t>> planes(3);
    for (int c = 0; c < 3; ++c) {
        std::vector<uint8_t> plane((size_t) w * h);
        for (size_t i = 0, n = (size_t) w * h; i < n; ++i) plane[i] = rgb[i * 3 + c];
        planes[c] = (th == h && tw == w) ? std::move(plane) : resizeBicubicAA(plane, h, w, th, tw);
    }

    // rescale: image * (1/255) in float32, then patchify to (nh*nw, p*p*3) ordered (py, px, c).
    const float rescale = (float) (1.0 / 255.0);
    const int pp = merged ? cfg_.modelPatchPixels() : cfg_.patchPixels();
    std::vector<float> patches((size_t) nh * nw * pp);
    std::vector<int32_t> pos((size_t) nh * nw * 2);
    for (int py = 0; py < nh; ++py) {
        for (int px = 0; px < nw; ++px) {
            const size_t patchIdx = (size_t) py * nw + px;
            float * dst = &patches[patchIdx * pp];
            for (int yy = 0; yy < p; ++yy) {
                for (int xx = 0; xx < p; ++xx) {
                    const size_t src = (size_t) (py * p + yy) * tw + (px * p + xx);
                    for (int c = 0; c < 3; ++c)
                        dst[(yy * p + xx) * 3 + c] = (float) planes[c][src] * rescale;
                }
            }
            pos[patchIdx * 2 + 0] = px;   // x
            pos[patchIdx * 2 + 1] = py;   // y
        }
    }

    ESImagePatches out{mx::array(patches.data(), {nh * nw, pp}, mx::float32),
                       mx::array(pos.data(), {nh * nw, 2}, mx::int32)};
    out.patchWidth = nw; out.patchHeight = nh; out.poolingKernel = k; out.merged = merged; out.budget = budget;
    return out;
}

ESImagePatches ESImagePreprocessor::processFile(const std::string & path, int budget) const {
    NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef) url, nullptr);
    if (!src) throw std::runtime_error("ESImagePreprocessor: cannot open " + path);
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!img) throw std::runtime_error("ESImagePreprocessor: not a decodable image: " + path);
    ESImagePatches out = process(img, budget);
    CGImageRelease(img);
    return out;
}

}  // namespace es
