//  APResponse — streamed deltas, the completed response, engine-measured stats, and the
//  in-flight task handle.
#import <Foundation/Foundation.h>
#import <AperturaKit/APMessage.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, APFinishReason) {
    APFinishReasonEndOfTurn   = 0,   // model closed its turn
    APFinishReasonMaxTokens   = 1,
    APFinishReasonCancelled   = 2,
    APFinishReasonContextFull = 3,
};

typedef NS_ENUM(NSInteger, APResponseDeltaKind) {
    APResponseDeltaKindText      = 0,   // a piece of the visible answer
    APResponseDeltaKindReasoning = 1,   // a piece of the reasoning channel
    APResponseDeltaKindToolCall  = 2,   // a tool call completed (dispatched or vetoed)
};

/// One streamed increment, in the order the turn unfolds: reasoning, tool calls and
/// answer text interleave as the model produces them (think, call, think again, call,
/// answer). The deltas of a turn, folded in order, ARE the assistant message's content
/// parts (APMessage.content) — a view renders a live turn and an archived one the same
/// way. Text deltas are UTF-8-safe: a delta never splits a character (the detokenizer
/// holds back incomplete byte sequences across token boundaries).
@interface APResponseDelta : NSObject

/// Replay: a recorded turn as the deltas that would have produced it, for a renderer
/// that has one path for live and archived content.
+ (instancetype)textDelta:(NSString *)text;
+ (instancetype)reasoningDelta:(NSString *)text;
+ (instancetype)toolCallDelta:(APToolCall *)toolCall;

@property (readonly) APResponseDeltaKind kind;
/// The text of a Text or Reasoning delta; empty for a tool call.
@property (readonly) NSString *text;
@property (readonly) NSInteger tokenCount;   // tokens represented by this delta (0 when unknown)
/// The call record of a ToolCall delta — name, arguments, status, result, duration.
@property (nullable, readonly) APToolCall *toolCall;
/// kind == APResponseDeltaKindReasoning: render these distinctly (or hide them).
@property (readonly) BOOL isThought;

- (instancetype)init NS_UNAVAILABLE;

@end

/// Engine-measured statistics for one response (same definitions as the CLI benches).
@interface APResponseStats : NSObject
@property (readonly) NSInteger promptTokenCount;      // this turn's prefilled tokens
@property (readonly) NSInteger responseTokenCount;
@property (readonly) NSTimeInterval timeToFirstToken;
@property (readonly) double prefillTokensPerSecond;
@property (readonly) double decodeTokensPerSecond;
@end

@interface APResponse : NSObject
@property (readonly) APMessage *message;              // role == APRoleAssistant (the ANSWER)
@property (readonly) APFinishReason finishReason;
@property (readonly) APResponseStats *stats;
/// The reasoning-channel text, when the session ran with reasoningEnabled and the model
/// produced one; nil otherwise. Parsed from the same grammar the reference engine gates.
/// Convenience for `message.reasoning` — the message carries the full generation record.
@property (readonly, nullable) NSString *reasoning;
/// Names of tools that RAN while producing this response, in call order (empty if none;
/// vetoed calls are not listed). Convenience over `message.toolCalls`.
@property (readonly) NSArray<NSString *> *executedToolNames;
@end

/// Handle for an in-flight prime or response.
@interface APResponseTask : NSObject
@property (readonly) NSProgress *progress;
/// Checked per token; the response finishes with APFinishReasonCancelled (not an error).
- (void)cancel;
@end

NS_ASSUME_NONNULL_END
