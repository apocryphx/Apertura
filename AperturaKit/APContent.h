//  APContent — one typed content part of a message.
//
//  Messages carry ARRAYS of parts, not strings, so richer inputs (images, audio) arrive
//  later as new factories plus a model-capability check — with no change to APMessage,
//  APSession, or persisted transcripts. v1 ships text; the other kinds are RESERVED
//  (constructing them is not yet possible; the enum cases exist so serialized values
//  stay forward-compatible).
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, APContentKind) {
    APContentKindText  = 0,
    APContentKindImage = 1,   // reserved — arrives with a vision-capable model
    APContentKindAudio = 2,   // reserved — arrives with an audio-capable model
};

@interface APContent : NSObject <NSCopying>

+ (instancetype)textContent:(NSString *)text;

@property (readonly) APContentKind kind;
@property (nullable, readonly) NSString *text;   // kind == APContentKindText

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
