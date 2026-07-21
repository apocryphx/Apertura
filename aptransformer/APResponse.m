#import "APResponse.h"
#import <stdatomic.h>

// Private initializers are declared in APInternal.h; this file is plain ObjC, so the
// (Internal) categories on these value types are redeclared locally where needed.

@interface APResponseDelta ()
- (instancetype)initWithText:(NSString *)text tokenCount:(NSInteger)tokenCount;
@end
@implementation APResponseDelta
- (instancetype)initWithText:(NSString *)text tokenCount:(NSInteger)tokenCount {
    if ((self = [super init])) { _text = [text copy]; _tokenCount = tokenCount; }
    return self;
}
@end

@interface APResponseStats ()
- (instancetype)initWithPromptTokens:(NSInteger)prompt responseTokens:(NSInteger)response
                    timeToFirstToken:(NSTimeInterval)ttft
                           prefillTPS:(double)prefillTPS decodeTPS:(double)decodeTPS;
@end
@implementation APResponseStats
- (instancetype)initWithPromptTokens:(NSInteger)prompt responseTokens:(NSInteger)response
                    timeToFirstToken:(NSTimeInterval)ttft
                           prefillTPS:(double)prefillTPS decodeTPS:(double)decodeTPS {
    if ((self = [super init])) {
        _promptTokenCount = prompt; _responseTokenCount = response;
        _timeToFirstToken = ttft;
        _prefillTokensPerSecond = prefillTPS; _decodeTokensPerSecond = decodeTPS;
    }
    return self;
}
@end

@interface APResponse ()
- (instancetype)initWithMessage:(APMessage *)message
                   finishReason:(APFinishReason)reason
                          stats:(APResponseStats *)stats;
@end
@implementation APResponse
- (instancetype)initWithMessage:(APMessage *)message
                   finishReason:(APFinishReason)reason
                          stats:(APResponseStats *)stats {
    if ((self = [super init])) { _message = message; _finishReason = reason; _stats = stats; }
    return self;
}
@end

@implementation APResponseTask {
    atomic_bool _cancelled;
}
- (instancetype)initWithProgress:(NSProgress *)progress {
    if ((self = [super init])) {
        _progress = progress;
        atomic_store(&_cancelled, false);
    }
    return self;
}
- (void)cancel { atomic_store(&_cancelled, true); }
- (BOOL)isCancelled { return atomic_load(&_cancelled); }
@end
