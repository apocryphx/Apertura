//  APSession — abstract base implementation: the state every backend shares.
//  Subclasses (APLocalSession, APGoogleSession) implement prime/respond/reset and reach
//  the shared state ONLY through APSessionSubclass.h, so locking stays in one place.
#import "APSession.h"
#import "APSessionSubclass.h"

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

@end
