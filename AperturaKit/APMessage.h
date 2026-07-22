//  APMessage — one turn of conversation: a role plus typed content parts.
//  The facade maps roles onto the Gemma-4 chat grammar (ESChatTemplate) at the token-id
//  level; applications never see template tokens.
#import <Foundation/Foundation.h>
#import <AperturaKit/APContent.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, APRole) {
    APRoleSystem    = 0,
    APRoleUser      = 1,
    APRoleAssistant = 2,
    APRoleTool      = 3,   // tool results re-entering the conversation (dispatch: later phase)
};

@interface APMessage : NSObject <NSCopying>

+ (instancetype)messageWithRole:(APRole)role content:(NSArray<APContent *> *)content;
+ (instancetype)systemMessageWithText:(NSString *)text;
+ (instancetype)userMessageWithText:(NSString *)text;
+ (instancetype)assistantMessageWithText:(NSString *)text;

@property (readonly) APRole role;
@property (readonly) NSArray<APContent *> *content;

/// All text parts concatenated (empty string if none).
@property (readonly) NSString *textRepresentation;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
