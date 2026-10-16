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

/// Builds the engine's Metal pipeline states (a 16-token prefill + one decode step into a
/// throwaway cache) so the first real prefill runs at full speed. Queued on the engine
/// thread, so sessions created right after it simply run after it.
/// Idempotent; completion on the main queue. Sessions created after prewarm reach
/// first-token fastest.
- (void)prewarmWithCompletion:(nullable void (^)(void))completion;

/// Releases reclaimable engine memory (cached Metal buffers; never weights).
- (void)reclaimMemory;

/// Tokenize `text` with the model's tokenizer (no special tokens). Blocking: the encode
/// runs on the engine thread and this call waits — cheap for prose-sized inputs, but a
/// generation in flight serializes ahead of it.
- (NSArray<NSNumber *> *)tokenizeText:(NSString *)text;

/// The token count of `text` — tokenizeText: without materializing the id array.
- (NSInteger)tokenCountForText:(NSString *)text;

@property (readonly) NSURL *modelURL;
@property (readonly) NSString *modelIdentifier;       // directory/bundle name
@property (readonly) NSInteger maximumContextLength;

/// Whether the loaded weights include the Gemma-4 vision tower (an HF snapshot with a
/// vision_config, or an .apml exported from one). Only then can a local session take
/// APImage parts; a bundle exported before vision support, or a text-only checkpoint, says NO.
@property (readonly) BOOL hasVisionTower;

/// Capability flags. supportsImageInput == hasVisionTower; audio is not implemented.
@property (readonly) BOOL supportsImageInput;
@property (readonly) BOOL supportsAudioInput;

/// Prompt tokens one image of `pixelSize` occupies on the local backend: the soft tokens
/// of the processor's resized patch grid plus the two image delimiters. 0 without a tower.
- (NSInteger)imageTokenCountForPixelSize:(CGSize)pixelSize;

/// Cheap pre-flight (no weights loaded): does the model at `url` carry a vision tower?
/// HF snapshot: a vision_config whose tensors are in the safetensors index; .apml: the
/// manifest's `modalities` lists "vision".
+ (BOOL)modelAtURLHasVisionTower:(NSURL *)url;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
