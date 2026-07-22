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

/// One streamed increment. Text deltas are UTF-8-safe: a delta never splits a character
/// (the detokenizer holds back incomplete byte sequences across token boundaries).
@interface APResponseDelta : NSObject
@property (readonly) NSString *text;
@property (readonly) NSInteger tokenCount;   // tokens represented by this delta
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
@property (readonly) APMessage *message;              // role == APRoleAssistant
@property (readonly) APFinishReason finishReason;
@property (readonly) APResponseStats *stats;
@end

/// Handle for an in-flight prime or response.
@interface APResponseTask : NSObject
@property (readonly) NSProgress *progress;
/// Checked per token; the response finishes with APFinishReasonCancelled (not an error).
- (void)cancel;
@end

NS_ASSUME_NONNULL_END
