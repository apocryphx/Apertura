//  APGenerationOptions — sampling controls for one response.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface APGenerationOptions : NSObject <NSCopying>

/// Deterministic greedy decoding — byte-stable across runs on the same model and
/// configuration (the conformance-gated mode; cross-run stability requires headBits 8).
+ (instancetype)deterministicOptions;

/// Chat sampling defaults (temperature 0.7, topK 64, topP 0.95).
+ (instancetype)defaultOptions;

@property (nonatomic) float temperature;                 // 0 == greedy
@property (nonatomic) NSInteger topK;                    // 0 == off
@property (nonatomic) float topP;                        // 1.0 == off
@property (nonatomic) NSInteger maximumResponseTokens;   // 0 == until end-of-turn/context

@end

NS_ASSUME_NONNULL_END
