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
                          stats:(APResponseStats *)stats {
    if ((self = [super init])) {
        _message = message; _finishReason = reason; _stats = stats;
    }
    return self;
}

- (NSString *)reasoning { return _message.reasoning; }

- (NSArray<NSString *> *)executedToolNames {
    NSMutableArray * names = [NSMutableArray array];
    for (APToolCall * call in _message.toolCalls)
        if (call.status != APToolCallStatusVetoed) [names addObject:call.name];
    return names;
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
