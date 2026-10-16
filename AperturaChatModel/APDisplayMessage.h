//  APDisplayMessage — a recorded message in its renderable form.
//
//  The message's parts, in order, as blocks (APChatBlock.h): built once, off the main
//  thread if a transcript is long, then handed to the view. The message itself stays
//  attached, so a view can reach the role, the generation record (stats, model) and the
//  original parts without a second lookup.
#import <Foundation/Foundation.h>
#import <AperturaKit/APMessage.h>
#import <AperturaChatModel/APChatBlock.h>

NS_ASSUME_NONNULL_BEGIN

@interface APDisplayMessage : NSObject

+ (instancetype)displayMessageWithMessage:(APMessage *)message;

@property (readonly) APMessage *message;
/// One block per part, in part order: an APTextBlock per text part (its Markdown parsed),
/// an APReasoningBlock per reasoning part, an APToolCallBlock per call, an APAttachmentBlock
/// per attachment. Reserved kinds (image, audio) produce no block.
@property (readonly) NSArray<APChatBlock *> *blocks;

/// The block mapping alone, for content that is not (yet) a message.
+ (NSArray<APChatBlock *> *)blocksForContent:(NSArray<APContent *> *)content;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
