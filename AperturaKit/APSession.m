//  APSession — abstract base implementation: the state every backend shares.
//  Subclasses (APLocalSession, APGoogleSession) implement prime/respond/reset and reach
//  the shared state ONLY through APSessionSubclass.h, so locking stays in one place.
#import "APSession.h"
#import "APSessionSubclass.h"
#import "APError.h"

const int APSessionMaxToolRounds = 4;

@interface APToolOutcome ()
- (instancetype)initWithVetoed:(BOOL)vetoed resultText:(NSString *)resultText error:(NSError *)error
                        record:(APToolCall *)record;
@end
@implementation APToolOutcome
- (instancetype)initWithVetoed:(BOOL)vetoed resultText:(NSString *)resultText error:(NSError *)error
                        record:(APToolCall *)record {
    if ((self = [super init])) {
        _vetoed = vetoed; _resultText = [resultText copy]; _error = error; _record = record;
    }
    return self;
}
@end

@implementation APTurnRecorder {
    __weak APSession * _session;
    void (^_deltaHandler)(APResponseDelta *);
    NSMutableArray<APContent *> * _parts;
    NSMutableString * _open;        // text of the trailing text/reasoning part, still growing
    APContentKind _openKind;
}

- (instancetype)initWithSession:(APSession *)session
                   deltaHandler:(void (^)(APResponseDelta *))deltaHandler {
    if ((self = [super init])) {
        _session = session;
        _deltaHandler = [deltaHandler copy];
        _parts = [NSMutableArray array];
    }
    return self;
}

- (void)closeOpenPart {
    if (!_open) return;
    [_parts addObject:(_openKind == APContentKindReasoning
                       ? [APContent reasoningContent:_open] : [APContent textContent:_open])];
    _open = nil;
}

- (void)emit:(APResponseDelta *)delta {
    @synchronized(self) {
        switch (delta.kind) {
            case APResponseDeltaKindToolCall:
                [self closeOpenPart];
                if (delta.toolCall) [_parts addObject:[APContent toolCallContent:delta.toolCall]];
                break;
            case APResponseDeltaKindText:
            case APResponseDeltaKindReasoning: {
                APContentKind kind = delta.kind == APResponseDeltaKindReasoning
                    ? APContentKindReasoning : APContentKindText;
                if (_open && _openKind != kind) [self closeOpenPart];
                if (!_open) { _open = [NSMutableString string]; _openKind = kind; }
                [_open appendString:delta.text];
                break;
            }
        }
    }
    void (^handler)(APResponseDelta *) = _deltaHandler;
    if (handler) [_session deliver:^{ handler(delta); }];
}

- (NSArray<APContent *> *)parts {
    @synchronized(self) {
        NSMutableArray * parts = [_parts mutableCopy];
        if (_open) [parts addObject:(_openKind == APContentKindReasoning
                                     ? [APContent reasoningContent:_open] : [APContent textContent:_open])];
        return parts;
    }
}

- (APMessage *)assistantMessageWithStats:(APResponseStats *)stats modelIdentifier:(NSString *)modelIdentifier {
    return [APMessage messageWithRole:APRoleAssistant content:self.parts
                                stats:stats modelIdentifier:modelIdentifier];
}

@end

@implementation APSession {
    NSMutableArray<APMessage *> * _transcript;
    NSMutableDictionary<NSString *, id<APTool>> * _tools;
    BOOL _lastPrimeRestoredFromSnapshot;
}

- (instancetype)init {
    if ((self = [super init])) {
        _callbackQueue = dispatch_get_main_queue();
        _excludesReasoningFromContext = YES;   // reference-template convention (see APSession.h)
        _transcript = [NSMutableArray array];
        _tools = [NSMutableDictionary dictionary];
    }
    return self;
}

#pragma mark - Shared conversation state

- (NSArray<APMessage *> *)transcript {
    @synchronized(self) { return [_transcript copy]; }
}

- (BOOL)lastPrimeRestoredFromSnapshot {
    @synchronized(self) { return _lastPrimeRestoredFromSnapshot; }
}

- (void)registerTool:(id<APTool>)tool {
    @synchronized(self) { _tools[tool.name] = tool; }
    if ([tool respondsToSelector:@selector(willAttachToSession:)])
        [tool willAttachToSession:self];
}

- (void)unregisterToolNamed:(NSString *)name {
    id<APTool> tool;
    @synchronized(self) { tool = _tools[name]; [_tools removeObjectForKey:name]; }
    if (tool && [tool respondsToSelector:@selector(didDetachFromSession:)])
        [tool didDetachFromSession:self];
}

#pragma mark - Abstract (backend responsibility)

- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                           completion:(void (^)(NSError *_Nullable))completion {
    return [self primeWithMessages:messages cacheURL:nil completion:completion];
}

- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                             cacheURL:(NSURL *)cacheURL
                           completion:(void (^)(NSError *_Nullable))completion {
    [self doesNotRecognizeSelector:_cmd];
    return nil;
}

- (APResponseTask *)respondToMessage:(APMessage *)message
                             options:(APGenerationOptions *)options
                        deltaHandler:(void (^)(APResponseDelta *))deltaHandler
                          completion:(void (^)(APResponse *, NSError *))completion {
    [self doesNotRecognizeSelector:_cmd];
    return nil;
}

- (NSInteger)contextTokenCount {
    [self doesNotRecognizeSelector:_cmd];
    return 0;
}

- (void)reset {
    [self doesNotRecognizeSelector:_cmd];
}

#pragma mark - Protected (APSessionSubclass.h)

- (void)deliver:(dispatch_block_t)block {
    dispatch_async(self.callbackQueue ?: dispatch_get_main_queue(), block);
}

- (void)appendTranscriptMessages:(NSArray<APMessage *> *)messages {
    @synchronized(self) { [_transcript addObjectsFromArray:messages]; }
}

- (void)clearTranscript {
    @synchronized(self) { [_transcript removeAllObjects]; }
}

- (NSArray<id<APTool>> *)toolsSortedByName {
    @synchronized(self) {
        NSMutableArray * ts = [NSMutableArray arrayWithCapacity:_tools.count];
        for (NSString * name in [_tools.allKeys sortedArrayUsingSelector:@selector(compare:)])
            [ts addObject:_tools[name]];
        return ts;
    }
}

- (void)noteLastPrimeRestoredFromSnapshot:(BOOL)restored {
    @synchronized(self) { _lastPrimeRestoredFromSnapshot = restored; }
}

#pragma mark - Tool dispatch (shared mediation semantics)

- (APToolOutcome *)dispatchToolNamed:(NSString *)name
                           arguments:(NSDictionary<NSString *, id> *)args
                               among:(NSArray<id<APTool>> *)activeTools
                            recorder:(APTurnRecorder *)recorder {
    id<APTool> tool = nil;
    for (id<APTool> t in activeTools)
        if ([t.name isEqualToString:name]) { tool = t; break; }

    id<APSessionDelegate> delegate = self.delegate;
    BOOL allowed = tool != nil;
    if (allowed && [delegate respondsToSelector:@selector(session:shouldInvokeTool:arguments:)])
        allowed = [delegate session:self shouldInvokeTool:tool arguments:args];

    __block NSString * resultText = nil;
    __block NSError * toolError = nil;
    NSTimeInterval duration = 0;
    if (allowed) {
        NSDate * t0 = [NSDate date];
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [tool invokeWithArguments:args completion:^(APContent * result, NSError * error) {
            resultText = result.text;
            toolError = error;
            dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        duration = -[t0 timeIntervalSinceNow];
    }
    APToolCall * record = [APToolCall
        toolCallWithName:name arguments:args
                  status:(!allowed ? APToolCallStatusVetoed
                          : toolError ? APToolCallStatusFailed : APToolCallStatusSucceeded)
                  result:(!allowed ? nil : toolError ? (toolError.localizedDescription ?: @"") : (resultText ?: @""))
                duration:duration];
    if ([delegate respondsToSelector:@selector(session:didInvokeTool:arguments:result:)]) {
        NSString * info = allowed ? (resultText ?: toolError.localizedDescription ?: @"")
                                  : @"(vetoed)";
        [self deliver:^{ [delegate session:self didInvokeTool:name arguments:args result:info]; }];
    }
    [recorder emit:[APResponseDelta toolCallDelta:record]];
    return [[APToolOutcome alloc] initWithVetoed:!allowed resultText:resultText error:toolError
                                          record:record];
}

#pragma mark - Message validation

/// Text and attachments reach the model as text; reasoning and tool records are carried
/// (replayed transcripts) and ignored. Images need a backend that says it takes them
/// (acceptsImageContent — a local session needs a model with a vision tower); audio needs a model that
/// can take it. Returns the offending kind's error, or nil.
- (NSError *)contentErrorForMessage:(APMessage *)m {
    for (APContent * c in m.content) {
        switch (c.kind) {
            case APContentKindText:
            case APContentKindAttachment:
            case APContentKindReasoning:
            case APContentKindToolCall:
                continue;
            case APContentKindImage:
                if (self.acceptsImageContent) continue;
                return APMakeError(APErrorUnsupportedContent, [self imageContentRefusalReason]);
            case APContentKindAudio:
                return APMakeError(APErrorUnsupportedContent, [NSString stringWithFormat:
                    @"%@ does not accept audio content", [self backendDisplayName]]);
        }
        return APMakeError(APErrorUnsupportedContent, [NSString stringWithFormat:
            @"%@ does not know content kind %ld", [self backendDisplayName], (long)c.kind]);
    }
    return nil;
}

- (NSString *)backendDisplayName {
    return [NSString stringWithFormat:@"the %@ backend", NSStringFromClass(self.class)];
}

- (BOOL)acceptsImageContent {
    return NO;
}

- (NSString *)imageContentRefusalReason {
    return [NSString stringWithFormat:@"%@ does not accept images", [self backendDisplayName]];
}

- (NSError *)validationErrorForPrimeMessages:(NSArray<APMessage *> *)messages {
    for (APMessage * m in messages) {
        if (m.role == APRoleTool)
            return APMakeError(APErrorInvalidMessage, @"tool messages are not supported in prime");
        NSError * content = [self contentErrorForMessage:m];
        if (content) return content;
    }
    return nil;
}

- (NSError *)validationErrorForRespondMessage:(APMessage *)message {
    if (message.role != APRoleUser)
        return APMakeError(APErrorInvalidMessage, @"respond requires a user message");
    return [self contentErrorForMessage:message];
}

@end
