//  APModelConfiguration — the vetted engine defaults, exposed narrowly.
//
//  Everything here defaults to the measured, conformance-gated configuration
//  (PERFORMANCE_ROADMAP.md); an app that never touches this class runs exactly the
//  llama.cpp-parity setup. Research switches (cache modes, eviction toggles, bench
//  paths) are deliberately NOT exposed — they live in the AperturaResearch CLI.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

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

@end

NS_ASSUME_NONNULL_END
