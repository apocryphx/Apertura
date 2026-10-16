//  APMessage — one turn of conversation: a role plus typed content parts, and for an
//  assistant turn the generation record (which model, what it measured).
//  The facade maps roles onto the Gemma-4 chat grammar (ESChatTemplate) at the token-id
//  level; applications never see template tokens.
#import <Foundation/Foundation.h>
#import <AperturaKit/APContent.h>

NS_ASSUME_NONNULL_BEGIN

@class APResponseStats;

typedef NS_ENUM(NSInteger, APRole) {
    APRoleSystem    = 0,
    APRoleUser      = 1,
    APRoleAssistant = 2,
    APRoleTool      = 3,   // tool results re-entering the conversation (dispatch: later phase)
};

/// The role's wire name — "system" / "user" / "assistant" / "tool": the APTranscript
/// document format and the chat template's role labels. These strings are persisted;
/// never rename. nil for a value outside the enum (refuse to guess).
FOUNDATION_EXPORT NSString *_Nullable APRoleName(APRole role);

/// The inverse: YES and `*outRole` set for one of the four names, NO otherwise.
FOUNDATION_EXPORT BOOL APRoleFromName(NSString *name, APRole *outRole);

@interface APMessage : NSObject <NSCopying>

+ (instancetype)messageWithRole:(APRole)role content:(NSArray<APContent *> *)content;
/// An assistant turn with its generation record: the parts in the order they happened
/// (reasoning, tool calls, answer text), the engine's measurements, and the model that
/// produced it. Backends build these; applications read them.
+ (instancetype)messageWithRole:(APRole)role
                        content:(NSArray<APContent *> *)content
                          stats:(nullable APResponseStats *)stats
                modelIdentifier:(nullable NSString *)modelIdentifier;
+ (instancetype)systemMessageWithText:(NSString *)text;
+ (instancetype)userMessageWithText:(NSString *)text;
+ (instancetype)assistantMessageWithText:(NSString *)text;

@property (readonly) APRole role;
@property (readonly) NSArray<APContent *> *content;

/// What the model receives for this turn as text: every part's promptText, concatenated
/// (empty string if none). Reasoning, tool records and images are not part of it
/// (APContent.promptText).
@property (readonly) NSString *textRepresentation;

/// The visible text alone — text parts concatenated, attachments and records left out.
/// What a transcript view shows as the message body.
@property (readonly) NSString *displayText;

/// The reasoning-channel text (reasoning parts concatenated); nil when there is none.
@property (nullable, readonly) NSString *reasoning;
/// The tool calls recorded in this turn, in call order (empty if none).
@property (readonly) NSArray<APToolCall *> *toolCalls;
/// The documents attached to this turn, in order (empty if none).
@property (readonly) NSArray<APAttachment *> *attachments;
/// The images attached to this turn, in order (empty if none).
@property (readonly) NSArray<APImage *> *images;

/// Generation record, assistant turns only; nil on turns that were not generated here
/// (user, system, or a v1 transcript's assistant turns).
@property (nullable, readonly) APResponseStats *stats;
/// The model that produced an assistant turn ("google/gemma-4-12b-it-qat", a Gemini model
/// name); nil when not recorded.
@property (nullable, readonly) NSString *modelIdentifier;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
