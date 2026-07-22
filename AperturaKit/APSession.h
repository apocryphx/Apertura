//  APSession — one conversation over one persistent KV cache.
//
//  The engine's prefix cache (ESSession semantics underneath) is the DEFAULT behavior:
//  the persona/context is prefilled once at prime and every turn appends only its delta
//  (the measured 33.7x multi-turn win). The streaming loop mirrors the gated CLI
//  session path token-for-token; byte-identity is enforced by the --facade-verify gate.
//
//  Concurrency: a session serializes its own work on an internal queue; concurrent
//  respond calls queue in order. Multiple sessions may share one APModel (weights are
//  shared; generation interleaves at token granularity). Callbacks arrive on
//  `callbackQueue` (default: main).
#import <Foundation/Foundation.h>
#import <AperturaKit/APModel.h>
#import <AperturaKit/APMessage.h>
#import <AperturaKit/APGenerationOptions.h>
#import <AperturaKit/APResponse.h>
#import <AperturaKit/APTool.h>

NS_ASSUME_NONNULL_BEGIN

@class APSession;

@protocol APSessionDelegate <NSObject>
@optional
/// Fired once when the cached context passes 80% of the maximum — the app's cue to
/// summarize, truncate, or start a new session. The framework never drops history.
- (void)sessionContextIsNearlyFull:(APSession *)session;
@end

@interface APSession : NSObject

- (instancetype)initWithModel:(APModel *)model;

@property (weak, nullable) id<APSessionDelegate> delegate;
@property (nonatomic) dispatch_queue_t callbackQueue;   // default: main queue

/// Ingest the standing prefix (persona / instructions) once. Chunked prefill applies;
/// combine with -[APModel prewarmWithCompletion:] to hide the whole cold path at app
/// launch. v1 accepts system/user/assistant messages (no tool role).
- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                           completion:(void (^)(NSError *_Nullable error))completion;

/// Same, with a persistent KV snapshot: when `cacheURL` holds a snapshot matching this
/// model + configuration + the EXACT prime content, the prefilled cache restores in
/// roughly file-read time instead of re-prefilling (a ~13.5K-token persona: ~a minute of
/// prefill vs ~a second of restore); otherwise primes normally and writes the snapshot.
/// Continuation from a restored cache is byte-identical to a fresh prime (gated by
/// --persist-verify). Any change to the persona text, model, head precision, or tokenizer
/// changes the fingerprint and invalidates the snapshot automatically. Only honored on a
/// fresh session (no prior context). Snapshot files are large (roughly the cached K/V:
/// ~1 GB for a 13.5K-token persona on the 31B). Use a ".safetensors" path.
- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                             cacheURL:(nullable NSURL *)cacheURL
                           completion:(void (^)(NSError *_Nullable error))completion;

/// YES when the most recent prime restored from its cacheURL snapshot instead of
/// prefilling — for status UI ("restored in 1.2 s" vs "primed and cached").
@property (readonly) BOOL lastPrimeRestoredFromSnapshot;

/// One turn. `message` must be role user (v1). Deltas stream as generated; completion
/// delivers the parsed response. Pass nil options for chat sampling defaults; use
/// +[APGenerationOptions deterministicOptions] for byte-stable greedy decoding.
- (APResponseTask *)respondToMessage:(APMessage *)message
                             options:(nullable APGenerationOptions *)options
                        deltaHandler:(nullable void (^)(APResponseDelta *delta))deltaHandler
                          completion:(void (^)(APResponse *_Nullable response,
                                               NSError *_Nullable error))completion;

/// Conversation state. The transcript is the app-persistable source of truth; replay it
/// into a fresh session (prime + turns) to restore after relaunch.
@property (readonly) NSArray<APMessage *> *transcript;
@property (readonly) NSInteger contextTokenCount;       // tokens currently cached
- (void)reset;                                          // drop cache + transcript

/// Tool registration (advertisement/dispatch lands in a later phase; see APTool.h).
- (void)registerTool:(id<APTool>)tool;
- (void)unregisterToolNamed:(NSString *)name;

@end

NS_ASSUME_NONNULL_END
