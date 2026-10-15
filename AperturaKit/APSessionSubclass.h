//  APSessionSubclass — PROJECT header (never public): the protected base-class surface
//  concrete backends build on. Everything here is implemented once in APSession.m so
//  every backend shares one transcript, tool registry, and delivery discipline.
#pragma once
#import "APSession.h"

NS_ASSUME_NONNULL_BEGIN

/// The tool-round cap every backend shares: a response pauses for at most this many
/// tool dispatches before the model must answer (local: token-level, remote: per request).
FOUNDATION_EXPORT const int APSessionMaxToolRounds;

/// What one mediated tool dispatch came back with. Exactly one of the three shapes:
/// vetoed (unknown tool, or the delegate said no — nothing ran), a result, or an error.
/// Backends render it into their own response grammar.
@interface APToolOutcome : NSObject
@property (readonly) BOOL vetoed;
@property (readonly, nullable) NSString *resultText;   // the tool's text, when it ran
@property (readonly, nullable) NSError *error;         // the tool's failure, when it ran
/// The same outcome as a transcript record (name, arguments, status, result, duration).
@property (readonly) APToolCall *record;
@end

@interface APSession (Subclass)

/// Deliver a callback block on `callbackQueue` (falls back to the main queue).
- (void)deliver:(dispatch_block_t)block;

/// Transcript mutation (thread-safe; the readonly `transcript` property snapshots).
- (void)appendTranscriptMessages:(NSArray<APMessage *> *)messages;
- (void)clearTranscript;

/// Registered tools sorted by name — the stable order every backend advertises
/// (local: stable prime ids and snapshot fingerprints; remote: stable request bodies).
- (NSArray<id<APTool>> *)toolsSortedByName;

/// Record whether the most recent prime restored from a KV snapshot (local backend;
/// backends without snapshots leave the default NO).
- (void)noteLastPrimeRestoredFromSnapshot:(BOOL)restored;

/// The mediated tool dispatch both backends share: look `name` up among `activeTools`,
/// ask the delegate (session:shouldInvokeTool:arguments:), invoke and BLOCK the calling
/// thread until the tool's completion fires (tools do IO — the documented serialization
/// on the engine thread / SSE queue), notify session:didInvokeTool:… on the callback
/// queue, and append the outcome's record to `calls` (vetoed ones included — the
/// transcript shows what the model asked for). The caller renders the outcome into its
/// grammar (local: tool_response body; remote: functionResponse part).
- (APToolOutcome *)dispatchToolNamed:(NSString *)name
                           arguments:(NSDictionary<NSString *, id> *)arguments
                               among:(NSArray<id<APTool>> *)activeTools
                               calls:(NSMutableArray<APToolCall *> *)calls;

/// The assistant turn a backend records for one response: reasoning first (when any),
/// then the tool calls in order, then the answer text; with the engine's stats and the
/// producing model. The shared shape, so both backends' transcripts read alike.
+ (APMessage *)assistantMessageWithAnswer:(NSString *)answer
                                reasoning:(nullable NSString *)reasoning
                                toolCalls:(NSArray<APToolCall *> *)toolCalls
                                    stats:(nullable APResponseStats *)stats
                          modelIdentifier:(nullable NSString *)modelIdentifier;

/// The message contract, checked once per backend entry point: prime accepts
/// system/user/assistant messages; respond requires a user message. Content may be text,
/// attachments, and — on replayed assistant turns — reasoning and tool records (which
/// never reach the model, see APContent.promptText); image and audio are refused. nil
/// when the input is acceptable, else the APError to fail with (APErrorInvalidMessage /
/// APErrorUnsupportedContent).
- (nullable NSError *)validationErrorForPrimeMessages:(NSArray<APMessage *> *)messages;
- (nullable NSError *)validationErrorForRespondMessage:(APMessage *)message;

@end

NS_ASSUME_NONNULL_END
