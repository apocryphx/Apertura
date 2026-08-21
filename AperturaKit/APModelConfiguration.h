//  APModelConfiguration — the vetted engine defaults, exposed narrowly.
//
//  Everything here defaults to the measured, conformance-gated configuration
//  (PERFORMANCE_ROADMAP.md); an app that never touches this class runs exactly the
//  llama.cpp-parity setup. Research switches (cache modes, eviction toggles, bench
//  paths) are deliberately NOT exposed — they live in the AperturaResearch CLI.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Global-layer KV cache representation — see APModelConfiguration.globalKVCacheMode.
typedef NS_ENUM(NSInteger, APGlobalKVCacheMode) {
    APGlobalKVCacheModeStandard = 0,
    APGlobalKVCacheModeRaw      = 1,
    APGlobalKVCacheModeRawQ8    = 2,
};

@interface APModelConfiguration : NSObject <NSCopying>

/// The measured defaults. Equivalent to +new.
+ (instancetype)defaultConfiguration;

/// LM-head precision. 8 (default): the model's shipped head (Q8 in .apml bundles) —
/// byte-stable, quality-first. 4: re-quantized Q4 head at load — +3.3-3.6% decode at
/// 99.40% top-1 agreement vs the Q8 head (roadmap P4).
@property (nonatomic) NSInteger headBits;

/// Prefill chunk length in tokens. Default 512 (roadmap P5). 0 disables chunking.
@property (nonatomic) NSInteger prefillChunkLength;

/// Upper bound on cached context per session, in tokens. Default 0 = the model's
/// maximum. Sessions fail with APErrorContextOverflow rather than silently evicting
/// conversation history.
@property (nonatomic) NSInteger maximumContextLength;

/// Reserved for the research/probe surface. Sealed (NO, default): sessions run only the
/// vetted graph. This flag has no effect yet.
@property (nonatomic) BOOL instrumented;

/// Global-layer KV cache representation (the 10 unwindowed layers whose cache grows with
/// context; sliding layers are unaffected). Measured at 60K context, 2026-08-20
/// (DECODE_AT_DEPTH.md):
///   Standard — post-norm bf16 K/V, the long-vetted default.
///   Raw      — pre-norm bf16 raw stream: HALF the growing bytes (cache + checkpoints)
///              at ≤1 ms/token decode cost; bit-exact prefill, decode drift in class.
///   RawQ8    — q8-packed raw stream: QUARTER of Standard's bytes (~60K global KV in
///              ~0.6 GB) at ~2.6 ms/token decode cost today; lossy within the same drift
///              class as the shipping fused path. The capacity mode.
/// Snapshots and checkpoints persisted under Raw and RawQ8 convert automatically when the
/// mode changes (q8 -> bf16 "rehydrates" by dequantizing — the bf16 cache then holds
/// exactly the values the q8 path saw, which cleanly separates quantization loss from
/// q8-path faults when diagnosing; bf16 -> q8 quantizes). Switching between Standard and
/// the raw modes re-primes from the transcript instead (the representations are not
/// interconvertible); nothing is ever silently mixed.
@property (nonatomic) APGlobalKVCacheMode globalKVCacheMode;

@end

NS_ASSUME_NONNULL_END
