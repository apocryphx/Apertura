#import "APResponseInternal.h"
#import <stdatomic.h>

@implementation APResponseDelta
- (instancetype)initWithText:(NSString *)text tokenCount:(NSInteger)tokenCount
                   isThought:(BOOL)isThought {
    if ((self = [super init])) { _text = [text copy]; _tokenCount = tokenCount; _isThought = isThought; }
    return self;
}
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

@implementation APResponse
- (instancetype)initWithMessage:(APMessage *)message
                   finishReason:(APFinishReason)reason
                          stats:(APResponseStats *)stats
                      reasoning:(NSString *)reasoning
              executedToolNames:(NSArray<NSString *> *)executedToolNames {
    if ((self = [super init])) {
        _message = message; _finishReason = reason; _stats = stats; _reasoning = [reasoning copy];
        _executedToolNames = [executedToolNames copy] ?: @[];
    }
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
