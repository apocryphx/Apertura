//  APSession — one conversation, any backend (ABSTRACT).
//
//  The base class owns everything a conversation needs regardless of where tokens come
//  from: the transcript, the tool registry, delegate + callback delivery, and the
//  reasoning flag. Concrete backends implement prime/respond/reset:
//    APLocalSession  — the on-device Apertura engine (persistent KV cache, snapshots,
//                      token-level tool dispatch; the conformance-gated path).
//    APGoogleSession — Gemma on Google's Gemini API (streaming REST, server-side
//                      thinking, native function calling).
//  Instantiate a concrete backend, never APSession itself. Applications hold APSession*
//  and switch backends without touching chat, tool, or rendering code.
//
//  Concurrency: a session serializes its own work; concurrent respond calls queue in
//  order. Callbacks arrive on `callbackQueue` (default: main).
#import <Foundation/Foundation.h>
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

/// Tool mediation: return NO to veto this invocation (the model receives an
/// "unavailable" tool response and continues). Default when unimplemented: YES.
- (BOOL)session:(APSession *)session shouldInvokeTool:(id<APTool>)tool
      arguments:(NSDictionary<NSString *, id> *)arguments;

/// Informational: a tool ran (or was vetoed) mid-response — for activity UI.
- (void)session:(APSession *)session didInvokeTool:(NSString *)toolName
      arguments:(NSDictionary<NSString *, id> *)arguments result:(NSString *)result;
@end

@interface APSession : NSObject

@property (weak, nullable) id<APSessionDelegate> delegate;
@property (nonatomic) dispatch_queue_t callbackQueue;   // default: main queue

/// Enable the model's reasoning channel (Gemma-4 "thinking"). Default NO. SET BEFORE
/// PRIME: the flag shapes the primed system turn (local) or the request's thinking
/// configuration (remote), so it is fixed for the session's lifetime (toggle = new
/// session). With it on, responses stream reasoning first (deltas with isThought YES)
/// and then the visible answer; APResponse.reasoning carries the thought text.
@property (nonatomic) BOOL reasoningEnabled;

/// Ingest the standing prefix (persona / instructions) once. v1 accepts
/// system/user/assistant messages (no tool role).
- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                           completion:(void (^)(NSError *_Nullable error))completion;

/// Same, with a persistent KV snapshot at `cacheURL` — an ON-DEVICE fast path: when the
/// file holds a snapshot matching this model + configuration + the EXACT prime content,
/// the prefilled cache restores in roughly file-read time instead of re-prefilling
/// (~13.5K-token persona: ~a minute of prefill vs ~a second of restore); otherwise primes
/// normally and writes the snapshot. Continuation from a restored cache is byte-identical
/// to a fresh prime (gated by --persist-verify); any change to the persona text, model,
/// head precision, or tokenizer invalidates the snapshot automatically. Only honored on a
/// fresh session; use a ".safetensors" path (files are large — roughly the cached K/V).
/// Remote backends have no local cache and IGNORE cacheURL (their prime is instant; the
/// standing prefix rides every request instead).
- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                             cacheURL:(nullable NSURL *)cacheURL
                           completion:(void (^)(NSError *_Nullable error))completion;

/// YES when the most recent prime restored from its cacheURL snapshot instead of
/// prefilling — for status UI ("restored in 1.2 s" vs "primed and cached"). Always NO on
/// backends that ignore cacheURL.
@property (readonly) BOOL lastPrimeRestoredFromSnapshot;

/// One turn. `message` must be role user (v1). Deltas stream as generated; completion
/// delivers the parsed response. Pass nil options for chat sampling defaults; use
/// +[APGenerationOptions deterministicOptions] for byte-stable greedy decoding (local
/// backend; remote backends apply the sampling values but cannot promise byte identity).
- (APResponseTask *)respondToMessage:(APMessage *)message
                             options:(nullable APGenerationOptions *)options
                        deltaHandler:(nullable void (^)(APResponseDelta *delta))deltaHandler
                          completion:(void (^)(APResponse *_Nullable response,
                                               NSError *_Nullable error))completion;

/// Conversation state. The transcript is the app-persistable source of truth; replay it
/// into a fresh session (prime + turns) to restore after relaunch.
@property (readonly) NSArray<APMessage *> *transcript;
@property (readonly) NSInteger contextTokenCount;       // tokens currently in context
- (void)reset;                                          // drop context + transcript

/// Tool registration. REGISTER BEFORE PRIME: declarations are advertised to the model
/// (sorted by name, so local prime ids — and persona snapshots — stay stable). During a
/// response, a completed tool call pauses generation, dispatches to the matching tool
/// (delegate may veto), returns the result to the model, and resumes — all within one
/// respond lifetime. Local dispatch is gated by --tools-verify.
- (void)registerTool:(id<APTool>)tool;
- (void)unregisterToolNamed:(NSString *)name;

@end

NS_ASSUME_NONNULL_END
