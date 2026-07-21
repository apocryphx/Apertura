//  APModel — a loaded Apertura model (weights + tokenizer + chat grammar).
//
//  One concrete class covers all gated families (dense 31B, MoE 26B, elastic E2B): the
//  engine dispatches on the model's own config.json. Weights ACQUISITION is out of
//  scope by design — this class takes a URL to an .apml bundle or an HF snapshot
//  directory; downloading and license flows belong to the application.
#import <Foundation/Foundation.h>
#import <AperturaKit/APModelConfiguration.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, APModelAvailability) {
    APModelAvailable            = 0,
    APModelNotFound             = 1,
    APModelIncompatible         = 2,
    APModelInsufficientMemory   = 3,   // on disk, but would not fit in RAM with headroom
};

@interface APModel : NSObject

/// Cheap pre-flight: reads config + sums weight file sizes + checks RAM headroom.
/// Does NOT load weights. Call before offering a model in UI.
+ (APModelAvailability)availabilityOfModelAtURL:(NSURL *)url
                                  configuration:(nullable APModelConfiguration *)configuration;

/// Loads weights. BLOCKING (seconds to tens of seconds for large models) — call on a
/// background queue, or use the async variant below.
+ (nullable instancetype)modelWithContentsOfURL:(NSURL *)url
                                  configuration:(nullable APModelConfiguration *)configuration
                                          error:(NSError **)error;

/// Async load on a background queue; completion on the main queue.
+ (void)loadModelAtURL:(NSURL *)url
         configuration:(nullable APModelConfiguration *)configuration
            completion:(void (^)(APModel *_Nullable model, NSError *_Nullable error))completion;

/// Runs the one-time Metal JIT warmup (~2 s in a fresh process) off the critical path.
/// Idempotent; completion on the main queue. Sessions created after prewarm reach
/// first-token fastest.
- (void)prewarmWithCompletion:(nullable void (^)(void))completion;

/// Releases reclaimable engine memory (cached Metal buffers; never weights).
- (void)reclaimMemory;

@property (readonly) NSURL *modelURL;
@property (readonly) NSString *modelIdentifier;       // directory/bundle name
@property (readonly) NSInteger maximumContextLength;

/// Capability flags (from the model config). NO for all current text bundles; the
/// gate for future APContent kinds.
@property (readonly) BOOL supportsImageInput;
@property (readonly) BOOL supportsAudioInput;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
