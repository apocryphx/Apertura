//  APError — AperturaKit error domain and codes.
//  Pure Objective-C: importable from .m files; no C++/MLX types.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const APErrorDomain;

/// userInfo key: the tool name for APErrorToolFailed.
FOUNDATION_EXPORT NSErrorUserInfoKey const APErrorToolNameKey;

typedef NS_ERROR_ENUM(APErrorDomain, APErrorCode) {
    APErrorModelNotFound        = 1,   // no bundle/snapshot at the URL
    APErrorIncompatibleModel    = 2,   // unknown format / config family
    APErrorInsufficientMemory   = 3,   // model would not fit with headroom
    APErrorContextOverflow      = 4,   // transcript exceeds the model context
    APErrorCancelled            = 5,
    APErrorToolFailed           = 6,
    APErrorUnsupportedContent   = 7,   // e.g. image content on a text-only model
    APErrorInvalidMessage       = 8,   // e.g. respond with a non-user message (v1)
    APErrorSessionBusy          = 9,   // reserved (session serializes internally)
    APErrorEngineFailure        = 100, // wrapped engine exception; see localizedDescription
};

NS_ASSUME_NONNULL_END
