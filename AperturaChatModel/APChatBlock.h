//  APChatBlock — the renderable pieces of one message.
//
//  A message's content parts (APContent) map onto blocks one to one: a text part is one
//  APTextBlock holding its Markdown blocks, a reasoning part one APReasoningBlock, a tool
//  call one APToolCallBlock with its arguments and result laid out for reading, an attached
//  document one APAttachmentBlock. Blocks are immutable and compared by IDENTITY, so a view
//  can key per-block state (expanded or collapsed) on the block itself.
#import <Foundation/Foundation.h>
#import <AperturaKit/APContent.h>

@class APMarkdownDocument, APCodeSnippet;

NS_ASSUME_NONNULL_BEGIN

/// Abstract: every block is one of the subclasses below.
@interface APChatBlock : NSObject
@end

/// The visible answer (or a user's message): Markdown, parsed into blocks.
@interface APTextBlock : APChatBlock
@property (readonly) APMarkdownDocument *document;
- (instancetype)initWithDocument:(APMarkdownDocument *)document NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// The model's reasoning channel, shown collapsed or dimmed. Plain text: reasoning is not
/// written for display and is not parsed as Markdown.
@interface APReasoningBlock : APChatBlock
@property (readonly) NSString *text;
- (instancetype)initWithText:(NSString *)text NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// A tool the model called, with the call and its outcome laid out for reading.
@interface APToolCallBlock : APChatBlock

/// Results can be huge; only this many characters are laid out and highlighted.
@property (class, readonly) NSUInteger resultPreviewLimit;

@property (readonly) APToolCall *toolCall;
/// The arguments as readable JSON (keys sorted, multi-line strings as text blocks).
@property (readonly) APCodeSnippet *argumentsSnippet;
/// The start of the result (up to resultPreviewLimit characters): JSON wrapping multi-line
/// text is laid out readably, other JSON is highlighted, anything else is plain. nil when
/// the call has no result (vetoed).
@property (nullable, readonly) APCodeSnippet *resultPreview;
/// Whether `resultPreview` stops short of the full `toolCall.result`.
@property (readonly, getter=isResultTruncated) BOOL resultTruncated;

- (instancetype)initWithToolCall:(APToolCall *)toolCall NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@end

/// A document the user attached, shown as a chip (name, size, type).
@interface APAttachmentBlock : APChatBlock
@property (readonly) APAttachment *attachment;
- (instancetype)initWithAttachment:(APAttachment *)attachment NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

NS_ASSUME_NONNULL_END
