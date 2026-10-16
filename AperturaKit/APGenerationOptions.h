//  APGenerationOptions — sampling controls for one response.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// The default visual-token budget (soft tokens per image), 280.
FOUNDATION_EXPORT const NSInteger APVisualTokenBudgetDefault;

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

/// LCG seed for sampled decoding. 0 is the engine default — the value every build
/// before this property used — so existing callers are unaffected. Distinct non-zero
/// seeds give distinct reproducible sample streams; greedy decoding ignores it.
@property (nonatomic) unsigned long long seed;

/// Soft tokens per image on the LOCAL backend — the Gemma-4 model card's "Variable Image
/// Resolution". One of 70, 140, 280 (default), 560, 1120: the picture is resized so it lands on
/// at most this many soft tokens, so the prompt grows with it. Lower budgets suit captioning
/// and classification and prefill faster; 560/1120 keep the detail OCR and small text need.
/// Any other value fails the turn with APErrorInvalidMessage. The Google backend ignores it:
/// Gemini has no soft-token budget (its per-image accounting is fixed), and the Gemma models
/// there expose no resolution control this SDK could map it to.
@property (nonatomic) NSInteger visualTokenBudget;

/// The five budgets the engine accepts, ascending.
+ (NSArray<NSNumber *> *)supportedVisualTokenBudgets;
+ (BOOL)isSupportedVisualTokenBudget:(NSInteger)budget;

@end

NS_ASSUME_NONNULL_END
