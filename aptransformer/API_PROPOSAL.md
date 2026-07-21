# AperturaKit — public API proposal (draft 1)

**Status: proposal for discussion. Nothing here is implemented.**

The product surface for embedding the Apertura engine in applications: **pure Objective-C
headers** over **Objective-C++ facade implementations** that call the existing `es::`
engine unchanged. The engine — with its conformance gates, measured llama.cpp parity, and
the P0/P5 defaults — does not move; the facade is a new *caller* of the same entry points
the gated CLI uses.

```
┌─────────────────────────────────────────────┐
│ App (Swift or Objective-C)                  │
├─────────────────────────────────────────────┤
│ AperturaKit public headers   (pure ObjC)    │  APModel, APSession, APMessage, APTool …
├─────────────────────────────────────────────┤
│ Facade implementations       (ObjC++ .mm)   │  C++ ivars: es::ESSession, tokenizer …
├─────────────────────────────────────────────┤
│ es:: engine                  (C++ in .mm)   │  unchanged, conformance-gated
├─────────────────────────────────────────────┤
│ MLX (static)                                │  hidden symbol visibility
└─────────────────────────────────────────────┘
```

Design rules:

1. Public headers compile as plain Objective-C (importable from `.m`): no `mlx/`, no
   `es::`, no `std::`. Foundation + CoreGraphics only.
2. Facade call sequences mirror the gated CLI paths exactly (`lastLogits` + cache +
   sampler), so the existing verify gates transfer; a facade-level byte-identity gate
   (`APSession` vs the CLI session path) is part of the deliverable.
3. Everything asynchronous is block-based with an optional delegate for mediation;
   callbacks arrive on a configurable queue (default: main).
4. The surface is shaped for things it does not yet do — multimodal content, stateful
   tools (memory), structured output — so those arrive as additive API, not breaking
   changes. See §Future.

Open questions at the end. Header-by-header:

---

## AperturaKit.h (umbrella)

```objc
#import <AperturaKit/APModel.h>
#import <AperturaKit/APModelConfiguration.h>
#import <AperturaKit/APSession.h>
#import <AperturaKit/APGenerationOptions.h>
#import <AperturaKit/APMessage.h>
#import <AperturaKit/APContent.h>
#import <AperturaKit/APResponse.h>
#import <AperturaKit/APTool.h>
#import <AperturaKit/APError.h>
```

All headers are wrapped in `NS_ASSUME_NONNULL_BEGIN/END`; nullability below is only
annotated where nullable.

---

## APError.h

```objc
FOUNDATION_EXPORT NSErrorDomain const APErrorDomain;

typedef NS_ERROR_ENUM(APErrorDomain, APErrorCode) {
    APErrorModelNotFound        = 1,   // no bundle/snapshot at the URL
    APErrorIncompatibleModel    = 2,   // unknown format / config family
    APErrorInsufficientMemory   = 3,   // model would not fit with headroom
    APErrorContextOverflow      = 4,   // transcript exceeds the model context
    APErrorCancelled            = 5,
    APErrorToolFailed           = 6,   // userInfo: APErrorToolNameKey, underlying error
    APErrorUnsupportedContent   = 7,   // e.g. image content on a text-only model
};
```

---

## APModelConfiguration.h

The vetted engine defaults, exposed narrowly. Every property ships with the measured
default from the performance work; apps that never touch this get the gated
configuration.

```objc
@interface APModelConfiguration : NSObject <NSCopying>

/// Measured defaults (llama.cpp-parity configuration). This is what +new returns.
+ (instancetype)defaultConfiguration;

/// LM-head precision. 8 (default): byte-stable, quality-first. 4: +3.3-3.6% decode at
/// 99.40% top-1 agreement vs the Q8 head (re-quantized at load; see roadmap P4).
@property (nonatomic) NSInteger headBits;

/// Prefill chunk length. Default 512 (roadmap P5). 0 disables chunking.
@property (nonatomic) NSInteger prefillChunkLength;

/// Upper bound on cached context per session, in tokens. Default: model maximum.
/// Sessions fail with APErrorContextOverflow rather than silently evict history.
@property (nonatomic) NSInteger maximumContextLength;

/// Reserved for the research/probe surface (§Future). Sealed (NO, default): the session
/// runs only the vetted graph. Instrumented: probes may attach; not for production.
@property (nonatomic) BOOL instrumented;

@end
```

Deliberately absent: cache-mode and eviction toggles (`--no-prealloc-cache` etc. stay
research CLI flags; the framework runs the gated defaults, full stop).

---

## APModel.h

One concrete class for all three gated families (dense 31B, MoE 26B, elastic E2B) —
`ESModelConfig` already dispatches on the model's own `config.json`, so the facade does
not need a class cluster yet (revisit if family-specific API ever appears).

```objc
typedef NS_ENUM(NSInteger, APModelAvailability) {
    APModelAvailable            = 0,
    APModelNotFound             = 1,
    APModelIncompatible         = 2,
    APModelInsufficientMemory   = 3,   // fits on disk, would not fit in RAM with headroom
};

@interface APModel : NSObject

/// Cheap pre-flight (reads config + manifest, checks RAM headroom; does NOT load
/// weights). Apps call this before offering a model in UI.
+ (APModelAvailability)availabilityOfModelAtURL:(NSURL *)url
                                  configuration:(nullable APModelConfiguration *)config;

/// Loads weights (blocking; dispatch to a background queue or use the async variant).
/// `url` is an .apml bundle or an HF snapshot directory — same as the engine.
+ (nullable instancetype)modelWithContentsOfURL:(NSURL *)url
                                  configuration:(nullable APModelConfiguration *)config
                                          error:(NSError **)error;
+ (void)loadModelAtURL:(NSURL *)url
         configuration:(nullable APModelConfiguration *)config
            completion:(void (^)(APModel *_Nullable model, NSError *_Nullable error))completion;

/// Runs the one-time Metal JIT warmup (~2 s on first use of a fresh process) off the
/// critical path. Idempotent. Sessions created after prewarm reach first token fastest.
- (void)prewarmWithCompletion:(nullable void (^)(void))completion;

/// Releases reclaimable engine memory (cached buffers; not weights). Called
/// automatically on memory-pressure notifications; exposed for explicit control.
- (void)reclaimMemory;

@property (readonly) NSString *modelIdentifier;      // e.g. "gemma-4-31b-it-qat-q4"
@property (readonly) NSInteger maximumContextLength;
@property (readonly) NSInteger parameterCount;

/// Capability flags read from the model config — the gate for §Future content kinds.
@property (readonly) BOOL supportsImageInput;        // NO for all current bundles
@property (readonly) BOOL supportsAudioInput;        // NO for all current bundles

@end
```

Weights acquisition is **out of scope** by design: the framework takes a URL and verifies
integrity against the bundle manifest; downloading (Background Assets, custom CDN, user
license flows for Gemma weights) belongs to the app.

---

## APContent.h + APMessage.h — roles and multimodal-ready content

Messages carry an **array of typed content parts**, not a string. Today only text exists;
the part model is what lets images/audio arrive later as new factories + a capability
check, with no change to `APMessage`, `APSession`, or the transcript model.

```objc
typedef NS_ENUM(NSInteger, APRole) {
    APRoleSystem    = 0,
    APRoleUser      = 1,
    APRoleAssistant = 2,
    APRoleTool      = 3,   // tool results re-entering the conversation
};

typedef NS_ENUM(NSInteger, APContentKind) {
    APContentKindText  = 0,
    // Reserved (see §Future; constructing these on an unsupporting model fails with
    // APErrorUnsupportedContent at send time):
    APContentKindImage = 1,
    APContentKindAudio = 2,
};

/// Immutable content part (class cluster).
@interface APContent : NSObject <NSCopying>
+ (instancetype)textContent:(NSString *)text;
@property (readonly) APContentKind kind;
@property (nullable, readonly) NSString *text;   // kind == text
@end

@interface APMessage : NSObject <NSCopying>
+ (instancetype)messageWithRole:(APRole)role content:(NSArray<APContent *> *)content;
+ (instancetype)systemMessageWithText:(NSString *)text;     // conveniences
+ (instancetype)userMessageWithText:(NSString *)text;
+ (instancetype)assistantMessageWithText:(NSString *)text;
@property (readonly) APRole role;
@property (readonly) NSArray<APContent *> *content;
@property (readonly) NSString *textRepresentation;  // concatenated text parts
@end
```

The facade maps roles onto `ESChatTemplate`'s token-level grammar (system block, turn
markers, tool-call channel) — apps never see template tokens.

---

## APGenerationOptions.h

```objc
@interface APGenerationOptions : NSObject <NSCopying>

/// Deterministic greedy decoding — byte-stable across runs on the same model + config
/// (the conformance-gated mode; requires headBits 8 for cross-run stability).
+ (instancetype)deterministicOptions;

/// Sampling defaults for chat (temperature 0.7, topK 64, topP 0.95 — generation_config).
+ (instancetype)defaultOptions;

@property (nonatomic) float temperature;      // 0 == greedy
@property (nonatomic) NSInteger topK;         // 0 == off
@property (nonatomic) float topP;             // 1.0 == off
@property (nonatomic) NSInteger maximumResponseTokens;   // 0 == until EOS/context
@property (nonatomic, copy, nullable) NSArray<NSString *> *stopSequences;

@end
```

Determinism as a first-class, documented option is a product differentiator this engine
can actually back (it is what the gates certify); it is the default only for tests, not
for chat.

---

## APResponse.h — streaming output

Block-first streaming with a cancellable task object; a delegate (on `APSession`) layers
mediation on top without complicating the common path.

```objc
typedef NS_ENUM(NSInteger, APFinishReason) {
    APFinishReasonEndOfTurn   = 0,   // model emitted end-of-turn/EOS
    APFinishReasonMaxTokens   = 1,
    APFinishReasonStopString  = 2,
    APFinishReasonCancelled   = 3,
    APFinishReasonContextFull = 4,
};

/// One streamed increment. Text deltas are UTF-8-safe: a delta never splits a character
/// (the detokenizer buffers incomplete byte sequences across token boundaries).
@interface APResponseDelta : NSObject
@property (readonly) NSString *text;
@property (readonly) NSInteger tokenCount;        // tokens represented by this delta
@end

/// Engine-measured statistics for one response (same definitions as the CLI benches).
@interface APResponseStats : NSObject
@property (readonly) NSInteger promptTokenCount;
@property (readonly) NSInteger responseTokenCount;
@property (readonly) NSTimeInterval timeToFirstToken;
@property (readonly) double prefillTokensPerSecond;
@property (readonly) double decodeTokensPerSecond;
@end

@interface APResponse : NSObject
@property (readonly) APMessage *message;              // role == assistant
@property (readonly) APFinishReason finishReason;
@property (readonly) APResponseStats *stats;
@end

/// Handle for an in-flight response.
@interface APResponseTask : NSObject
@property (readonly) NSProgress *progress;    // totalUnitCount == max tokens when known
- (void)cancel;                               // checked per token; finishes with
@end                                          //   APFinishReasonCancelled (not an error)
```

---

## APSession.h

Owns one conversation over one persistent KV cache (`es::ESSession` underneath — the
33.7× multi-turn win is the default behavior, not an optimization apps opt into).

```objc
@class APSession;

@protocol APSessionDelegate <NSObject>
@optional
/// Tool mediation: return NO to veto (the model sees a refusal result). Default YES.
- (BOOL)session:(APSession *)session shouldInvokeTool:(id<APTool>)tool
      arguments:(NSDictionary<NSString *, id> *)arguments;
/// Fired once when context passes 80% of the maximum — the app's cue to summarize,
/// truncate, or start a new session. The framework never silently drops history.
- (void)sessionContextIsNearlyFull:(APSession *)session;
@end

@interface APSession : NSObject

- (instancetype)initWithModel:(APModel *)model;

@property (weak, nullable) id<APSessionDelegate> delegate;
@property (nonatomic) dispatch_queue_t callbackQueue;      // default: main queue

/// Ingest the standing prefix (persona / instructions) once. Chunked-prefill applies;
/// combine with -[APModel prewarm] to hide the whole cold path at app launch.
- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                           completion:(void (^)(NSError *_Nullable error))completion;

/// One turn. Deltas stream as generated; completion delivers the full response.
/// Tool calls emitted by the model are dispatched to registered tools automatically
/// (subject to delegate veto), their results appended as APRoleTool messages, and
/// generation continues — all within this one call's streaming lifetime.
- (APResponseTask *)respondToMessage:(APMessage *)message
                             options:(nullable APGenerationOptions *)options
                        deltaHandler:(nullable void (^)(APResponseDelta *delta))deltaHandler
                          completion:(void (^)(APResponse *_Nullable response,
                                               NSError *_Nullable error))completion;

/// Conversation state. The transcript is the source of truth an app can persist and
/// replay into a fresh session (prime + turns) after relaunch.
@property (readonly) NSArray<APMessage *> *transcript;
@property (readonly) NSInteger contextTokenCount;      // tokens currently cached
- (void)reset;                                         // drop cache + transcript

/// Tools available to the model in this session (see APTool.h).
- (void)registerTool:(id<APTool>)tool;
- (void)unregisterToolNamed:(NSString *)name;

@end
```

Concurrency contract: a session serializes its own work on an internal queue; concurrent
`respondToMessage:` calls on one session queue up in order. Multiple sessions may share
one `APModel` (weights are shared; generation across sessions is serialized by the
engine — documented, not hidden).

---

## APTool.h — tools, shaped for stateful additions

The protocol is object-based (not function-based) precisely so tools can carry state and
lifecycle — which is what a future memory tool needs (§Future). The runtime-dispatch
adapter (`APSelectorTool`) is the ObjC showpiece: register a method, get a tool.

```objc
@protocol APTool <NSObject>

@property (readonly) NSString *name;                       // e.g. "search_notes"
@property (readonly) NSString *toolDescription;            // shown to the model
/// JSON-Schema fragment for the arguments object (what the chat template advertises).
@property (readonly) NSDictionary<NSString *, id> *parameterSchema;

/// Asynchronous by design — tools do IO. Result becomes an APRoleTool message.
- (void)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments
                 completion:(void (^)(APContent *_Nullable result,
                                      NSError *_Nullable error))completion;

@optional
/// Lifecycle for stateful tools (memory stores, connections). Attach/detach bracket the
/// tool's registration on a session.
- (void)willAttachToSession:(APSession *)session;
- (void)didDetachFromSession:(APSession *)session;

@end

/// Adapter: wraps a target/selector as a tool. The parameter schema is derived from the
/// method signature via the ObjC runtime (type encodings + a keyword-to-parameter map),
/// and invocation goes through NSInvocation — one @selector, zero boilerplate.
@interface APSelectorTool : NSObject <APTool>
+ (instancetype)toolWithName:(NSString *)name
             toolDescription:(NSString *)description
                      target:(id)target
                      action:(SEL)action;      // (NSDictionary *args, completion block)
@end
```

---

## §Future — what this surface is shaped for (not in v1)

**Images and audio.** Additive: two factories on `APContent`
(`+imageContentWithCGImage:` / `+imageContentWithContentsOfURL:`,
`+audioContentWithData:sampleRate:channels:`), gated at send time by
`model.supportsImageInput/AudioInput` (read from the model config — the Gemma-4
snapshots already carry `processor_config.json`). Message, transcript, session, and
streaming APIs are untouched; the engine grows a vision/audio tower behind the same
`es::` boundary when that work happens. Reserved `APContentKind` cases exist from day
one so persisted transcripts stay forward-compatible.

**Memory.** A stateful tool, not new session API:

```objc
@protocol APMemoryStore <NSObject>            // app-provided persistence
- (void)storeMemory:(NSString *)text completion:(void (^)(NSError *_Nullable))completion;
- (void)recallMatching:(NSString *)query limit:(NSInteger)limit
            completion:(void (^)(NSArray<NSString *> *_Nullable, NSError *_Nullable))completion;
@end

@interface APMemoryTool : NSObject <APTool>   // framework-provided adapter: exposes
+ (instancetype)toolWithStore:(id<APMemoryStore>)store;   // store/recall to the model
@end
```

The lifecycle hooks (`willAttachToSession:`) let it inject a recall pass at prime time.
Storage stays the app's business (Core Data, files, a vector index) — the framework
defines the conversation-facing contract only.

**Structured output.** A future `APGenerationOptions.responseSchema` (JSON Schema) backed
by constrained decoding in the sampler we own. Deliberately out of v1.

**Probes / instrumented mode.** The research surface (`configuration.instrumented`):
block-based observation hooks on module outputs, timings, and logits — the productized
form of `forwardTrace`/the profiling work. Sealed mode (default) refuses attachment, so
production sessions run only the vetted graph.

**Swift refinements.** Additive polish once the ObjC surface settles: `NS_SWIFT_NAME`
audit, `AsyncThrowingStream` wrappers over the delta blocks, a small overlay — no
architectural change.

---

## Open questions

1. **Framework naming/packaging**: new `AperturaKit` product wrapping the existing
   `aptransformer` target, vs. renaming `aptransformer`'s public face (app target is
   already named "Apertura" — avoid a three-way name collision).
2. **Minimum OS**: macOS 14 (current deployment target) — confirm; iOS later via the E2B
   family (the only member with a phone-plausible footprint).
3. **Tool auto-invocation policy**: default-automatic with delegate veto (proposed) vs.
   default-manual. Automatic matches FoundationModels ergonomics; veto keeps control.
4. **`primeWithMessages:` vs. system-message-in-first-respond**: explicit prime (proposed)
   makes the 33.7× prefix-cache behavior legible; the convenience path can be added.
5. **Stats surface**: is exposing `APResponseStats` (tok/s, TTFT) product API or debug
   noise? Proposed: keep it — this project's users care, and llama-server ships the same.

## Validation plan

- Facade byte-identity gate: `APSession` transcript-for-transcript vs. the gated CLI
  session path, greedy, on all three families (the `session-verify` sibling).
- Port the CLI verify gates into `aptransformerTests` as XCTests (⌘U runs the whole
  conformance discipline).
- Streaming-specific tests: UTF-8 delta integrity (multibyte pieces split across tokens),
  cancellation mid-decode, tool round-trip, context-overflow behavior.
