//  APLiveMessage — an assistant turn as it streams.
//
//  Folds APResponseDelta in arrival order exactly as the kit's turn recorder does
//  (consecutive text of one kind coalesces into one part, a tool call is a part of its own),
//  and keeps the blocks current after every delta: closed parts keep their blocks, the open
//  reasoning part is re-wrapped, and the open text part goes through an APMarkdownStream so
//  only its unsettled tail is re-parsed. When the response finishes, -messageWithStats:
//  modelIdentifier: yields the same APMessage the session records, so the view can swap the
//  live turn for the archived one without a visible change.
#import <Foundation/Foundation.h>
#import <AperturaKit/APResponse.h>
#import <AperturaChatModel/APChatBlock.h>

NS_ASSUME_NONNULL_BEGIN

/// Not thread-safe: feed and read it on one queue (the session's callback queue).
@interface APLiveMessage : NSObject

- (instancetype)init NS_DESIGNATED_INITIALIZER;

/// Appends one delta, in the order the session delivered it.
- (void)appendDelta:(APResponseDelta *)delta;

/// The blocks so far: one per closed part, plus the open part's.
@property (readonly) NSArray<APChatBlock *> *blocks;
/// The parts so far, as the turn recorder would hold them.
@property (readonly) NSArray<APContent *> *content;

/// The assistant message for the deltas so far: the parts, the engine's stats and the
/// producing model — field for field what APSession records for the turn.
- (APMessage *)messageWithStats:(nullable APResponseStats *)stats
                modelIdentifier:(nullable NSString *)modelIdentifier;

@end

NS_ASSUME_NONNULL_END
