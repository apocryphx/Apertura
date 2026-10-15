//  APContent — one typed content part of a message.
//
//  Messages carry ARRAYS of parts, not strings. A part is either something the model
//  receives (text, an attached document) or a record of what the model did while
//  answering (its reasoning, a tool call with its result). Both live in the same
//  transcript so one document serves replay AND display: `promptText` is what reaches the
//  model, and the display layer reads the parts themselves. Image and audio stay RESERVED
//  (constructing them is not yet possible; the enum cases exist so serialized values stay
//  forward-compatible).
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, APContentKind) {
    APContentKindText       = 0,
    APContentKindImage      = 1,   // reserved — arrives with a vision-capable model
    APContentKindAudio      = 2,   // reserved — arrives with an audio-capable model
    APContentKindReasoning  = 3,   // the model's thought channel (assistant turns)
    APContentKindToolCall   = 4,   // a tool the model called, with its outcome (assistant turns)
    APContentKindAttachment = 5,   // a document the user attached (user turns)
};

#pragma mark - Attachment

/// A document attached to a user turn: the file's text as it was read at attach time,
/// plus what is needed to show it as a chip (name, size, type). The model receives it
/// framed as a block of text (see APContent.promptText).
@interface APAttachment : NSObject <NSCopying>

+ (instancetype)attachmentWithName:(NSString *)name
                              text:(NSString *)text
                         sizeBytes:(unsigned long long)sizeBytes
                    contentTypeIdentifier:(nullable NSString *)contentTypeIdentifier;

@property (readonly) NSString *name;
@property (readonly) NSString *text;
@property (readonly) unsigned long long sizeBytes;
/// A Uniform Type identifier (e.g. "net.daringfireball.markdown"), when known.
@property (nullable, readonly) NSString *contentTypeIdentifier;

/// The text the model receives: the content between named begin/end lines. Dashes rather
/// than angle brackets — nothing here can be mistaken for a chat-grammar marker.
@property (readonly) NSString *promptText;

- (instancetype)init NS_UNAVAILABLE;

@end

#pragma mark - Tool call

typedef NS_ENUM(NSInteger, APToolCallStatus) {
    APToolCallStatusSucceeded = 0,   // the tool ran and returned a result
    APToolCallStatusFailed    = 1,   // the tool ran and returned an error
    APToolCallStatusVetoed    = 2,   // unknown tool, or the delegate declined — nothing ran
};

/// One tool invocation the model made mid-response, as the session mediated it: the call
/// (name + arguments), the outcome the model was given back, and how long the tool took.
@interface APToolCall : NSObject <NSCopying>

+ (instancetype)toolCallWithName:(NSString *)name
                       arguments:(NSDictionary<NSString *, id> *)arguments
                          status:(APToolCallStatus)status
                          result:(nullable NSString *)result
                        duration:(NSTimeInterval)duration;

@property (readonly) NSString *name;
/// The parsed arguments object (JSON-compatible values only).
@property (readonly) NSDictionary<NSString *, id> *arguments;
@property (readonly) APToolCallStatus status;
/// The tool's text on success, its error description on failure, nil when vetoed.
@property (nullable, readonly) NSString *result;
/// Wall-clock seconds the invocation took; 0 when it did not run.
@property (readonly) NSTimeInterval duration;

- (instancetype)init NS_UNAVAILABLE;

@end

#pragma mark - Content

@interface APContent : NSObject <NSCopying>

+ (instancetype)textContent:(NSString *)text;
+ (instancetype)reasoningContent:(NSString *)text;
+ (instancetype)toolCallContent:(APToolCall *)toolCall;
+ (instancetype)attachmentContent:(APAttachment *)attachment;

@property (readonly) APContentKind kind;
@property (nullable, readonly) NSString *text;             // kind == Text or Reasoning
@property (nullable, readonly) APToolCall *toolCall;       // kind == ToolCall
@property (nullable, readonly) APAttachment *attachment;   // kind == Attachment

/// What the model receives for this part: the text of a text part, the framed block of an
/// attachment, nil for parts that never reach the model (reasoning, tool records — the
/// reference template strips prior-turn thinking, and tool rounds live in the model's own
/// turn grammar, not in replayed text).
@property (nullable, readonly) NSString *promptText;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
