#import "APResponseInternal.h"
#import <stdatomic.h>

@implementation APResponseDelta
- (instancetype)initWithKind:(APResponseDeltaKind)kind text:(NSString *)text
                  tokenCount:(NSInteger)tokenCount toolCall:(APToolCall *)toolCall {
    if ((self = [super init])) {
        _kind = kind; _text = [text copy] ?: @""; _tokenCount = tokenCount; _toolCall = toolCall;
    }
    return self;
}
- (instancetype)initWithText:(NSString *)text tokenCount:(NSInteger)tokenCount
                   isThought:(BOOL)isThought {
    return [self initWithKind:(isThought ? APResponseDeltaKindReasoning : APResponseDeltaKindText)
                         text:text tokenCount:tokenCount toolCall:nil];
}
+ (instancetype)textDelta:(NSString *)text {
    return [[self alloc] initWithKind:APResponseDeltaKindText text:text tokenCount:0 toolCall:nil];
}
+ (instancetype)reasoningDelta:(NSString *)text {
    return [[self alloc] initWithKind:APResponseDeltaKindReasoning text:text tokenCount:0 toolCall:nil];
}
+ (instancetype)toolCallDelta:(APToolCall *)toolCall {
    return [[self alloc] initWithKind:APResponseDeltaKindToolCall text:@"" tokenCount:0 toolCall:toolCall];
}
- (BOOL)isThought { return _kind == APResponseDeltaKindReasoning; }
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
