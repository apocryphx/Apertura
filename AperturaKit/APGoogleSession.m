//  APGoogleSession — streaming conversation backend over the Gemini API.
//
//  One respond = up to kMaxToolRounds+1 streamGenerateContent requests (SSE): text and
//  thought parts stream as deltas; functionCall parts pause the turn, dispatch through
//  the shared tool path (delegate veto included), and the functionResponse parts ride
//  the next request — mirroring the local backend's pause→dispatch→splice→resume shape
//  at the request level instead of the token level.
//
//  Threading: `_workQueue` serializes turns (concurrent respond calls queue in order,
//  matching the base-class contract); each HTTP round runs an APGoogleStreamingRound
//  whose NSURLSession delegate callbacks arrive on a private serial queue. Callbacks to
//  the app are delivered on callbackQueue via the base `deliver:`.
#import "APGoogleSession.h"
#import "APSessionSubclass.h"
#import "APInternal.h"
#import "APError.h"

static NSString * const kAPGoogleBaseURL =
    @"https://generativelanguage.googleapis.com/v1beta/models";
static const int kMaxToolRounds = 4;   // same cap as the local backend

// NSError userInfo keys carried on APErrorRemoteService (private to this backend).
static NSString * const kAPGoogleHTTPStatusKey = @"APGoogleHTTPStatus";
static NSString * const kAPGoogleRetryAfterKey = @"APGoogleRetryAfterSeconds";
static const int    kMaxQuotaRetries   = 2;     // per respond, across all rounds
static const double kDefaultRetryDelay = 20.0;  // when 429 gives no RetryInfo
static const double kMaxRetryDelay     = 60.0;  // longer than this → surface the error

static NSError * apGoogleError(APErrorCode code, NSString * message) {
    return [NSError errorWithDomain:APErrorDomain code:code
                           userInfo:@{ NSLocalizedDescriptionKey : message }];
}

#pragma mark - One SSE request/response round

/// Owns a single streamGenerateContent exchange: sends the request, parses the SSE
/// stream incrementally, forwards parts as they arrive, and reports the round's end.
@interface APGoogleStreamingRound : NSObject <NSURLSessionDataDelegate>
@property (copy) void (^onPart)(NSDictionary * part);          // SSE-queue context
@property (copy) void (^onDone)(NSString *_Nullable finishReason,
                                NSDictionary *_Nullable usage,
                                NSError *_Nullable error);
@property (copy) BOOL (^isCancelled)(void);
- (void)startWithRequest:(NSURLRequest *)request;
@end

@implementation APGoogleStreamingRound {
    NSURLSession * _session;
    NSURLSessionDataTask * _task;
    NSMutableData * _buffer;
    NSInteger _httpStatus;
    NSString * _finishReason;
    NSDictionary * _usage;
    BOOL _done;
}

- (void)startWithRequest:(NSURLRequest *)request {
    _buffer = [NSMutableData data];
    NSOperationQueue * q = [[NSOperationQueue alloc] init];
    q.maxConcurrentOperationCount = 1;
    _session = [NSURLSession sessionWithConfiguration:
                    NSURLSessionConfiguration.ephemeralSessionConfiguration
                                             delegate:self delegateQueue:q];
    _task = [_session dataTaskWithRequest:request];
    [_task resume];
}

- (void)finishWithError:(NSError *)error {
    if (_done) return;
    _done = YES;
    self.onDone(_finishReason, _usage, error);
    [_session finishTasksAndInvalidate];
}

- (void)URLSession:(NSURLSession *)s dataTask:(NSURLSessionDataTask *)t
    didReceiveResponse:(NSURLResponse *)response
     completionHandler:(void (^)(NSURLSessionResponseDisposition))handler {
    _httpStatus = [(NSHTTPURLResponse *)response statusCode];
    handler(NSURLSessionResponseAllow);
}

- (void)URLSession:(NSURLSession *)s dataTask:(NSURLSessionDataTask *)t
    didReceiveData:(NSData *)data {
    if (self.isCancelled && self.isCancelled()) { [t cancel]; return; }
    [_buffer appendData:data];
    if (_httpStatus == 200) [self drainEventLines];
    // non-200: keep buffering; the whole body is the error message, read at completion
}

/// SSE framing: complete "data: {json}" lines, one JSON chunk per line.
- (void)drainEventLines {
    while (true) {
        NSRange nl = [_buffer rangeOfData:[NSData dataWithBytes:"\n" length:1]
                                  options:0 range:NSMakeRange(0, _buffer.length)];
        if (nl.location == NSNotFound) return;
        NSData * lineData = [_buffer subdataWithRange:NSMakeRange(0, nl.location)];
        [_buffer replaceBytesInRange:NSMakeRange(0, NSMaxRange(nl)) withBytes:NULL length:0];
        NSString * line = [[NSString alloc] initWithData:lineData encoding:NSUTF8StringEncoding];
        line = [line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
        if (![line hasPrefix:@"data:"]) continue;
        NSString * json = [[line substringFromIndex:5] stringByTrimmingCharactersInSet:
                           NSCharacterSet.whitespaceCharacterSet];
        NSDictionary * chunk = [NSJSONSerialization JSONObjectWithData:
                                [json dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
        if (![chunk isKindOfClass:NSDictionary.class]) continue;
        NSDictionary * cand = [chunk[@"candidates"] isKindOfClass:NSArray.class]
            ? [(NSArray *)chunk[@"candidates"] firstObject] : nil;
        for (NSDictionary * part in cand[@"content"][@"parts"]) {
            if ([part isKindOfClass:NSDictionary.class]) self.onPart(part);
        }
        if ([cand[@"finishReason"] isKindOfClass:NSString.class]) _finishReason = cand[@"finishReason"];
        if ([chunk[@"usageMetadata"] isKindOfClass:NSDictionary.class]) _usage = chunk[@"usageMetadata"];
    }
}

- (void)URLSession:(NSURLSession *)s task:(NSURLSessionTask *)t
    didCompleteWithError:(NSError *)error {
    if (error) {
        BOOL cancelled = (error.code == NSURLErrorCancelled &&
                          [error.domain isEqualToString:NSURLErrorDomain]);
        [self finishWithError:cancelled ? nil : apGoogleError(APErrorNetworkFailure,
            error.localizedDescription ?: @"network failure")];
        return;
    }
    if (_httpStatus != 200) {
        id body = [NSJSONSerialization JSONObjectWithData:_buffer options:0 error:nil];
        id err = [body isKindOfClass:NSDictionary.class] ? body[@"error"] : nil;
        id msgV = [err isKindOfClass:NSDictionary.class] ? err[@"message"] : nil;
        NSString * msg = [msgV isKindOfClass:NSString.class] ? msgV
            : [NSString stringWithFormat:@"HTTP %ld from the Gemini API", (long)_httpStatus];
        NSMutableDictionary * info = [NSMutableDictionary dictionary];
        info[NSLocalizedDescriptionKey] = msg;
        info[kAPGoogleHTTPStatusKey] = @(_httpStatus);
        // RetryInfo detail: "retryDelay":"17.462137515s" — the API's own back-off hint.
        NSString * raw = [[NSString alloc] initWithData:_buffer encoding:NSUTF8StringEncoding];
        NSTextCheckingResult * m = [[NSRegularExpression
            regularExpressionWithPattern:@"\"retryDelay\"\\s*:\\s*\"([0-9.]+)s\""
                                 options:0 error:nil]
            firstMatchInString:(raw ?: @"") options:0 range:NSMakeRange(0, raw.length)];
        if (m) info[kAPGoogleRetryAfterKey] =
            @([[raw substringWithRange:[m rangeAtIndex:1]] doubleValue]);
        [self finishWithError:[NSError errorWithDomain:APErrorDomain
                                                  code:APErrorRemoteService userInfo:info]];
        return;
    }
    [self drainEventLines];
    [self finishWithError:nil];
}

@end

#pragma mark - Session backend

@implementation APGoogleSession {
    NSString * _modelName;
    NSString * (^_keyProvider)(void);
    dispatch_queue_t _workQueue;
    NSString * _systemInstruction;                 // from prime; resent every request
    NSMutableArray<NSDictionary *> * _contents;    // API-side history (incl. tool turns)
    NSInteger _lastTotalTokens;
}

- (instancetype)initWithModelName:(NSString *)modelName
                   apiKeyProvider:(NSString * (^)(void))apiKeyProvider {
    if ((self = [super init])) {
        _modelName = [modelName copy];
        _keyProvider = [apiKeyProvider copy];
        _workQueue = dispatch_queue_create("com.elarity.aperturakit.google", DISPATCH_QUEUE_SERIAL);
        _contents = [NSMutableArray array];
    }
    return self;
}

- (NSString *)modelName { return _modelName; }

- (NSInteger)contextTokenCount {
    @synchronized(self) { return _lastTotalTokens; }
}

- (void)reset {
    dispatch_sync(_workQueue, ^{
        [self->_contents removeAllObjects];
        self->_systemInstruction = nil;
        @synchronized(self) { self->_lastTotalTokens = 0; }
        [self clearTranscript];
    });
}

#pragma mark - Prime (instant: the prefix rides every request)

- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                             cacheURL:(NSURL *)cacheURL
                           completion:(void (^)(NSError *_Nullable))completion {
    // cacheURL is a local-KV hint — meaningless here, deliberately ignored (see header).
    APResponseTask * task = [[APResponseTask alloc]
        initWithProgress:[NSProgress progressWithTotalUnitCount:-1]];
    dispatch_async(_workQueue, ^{
        NSMutableArray<NSString *> * system = [NSMutableArray array];
        NSMutableArray<NSDictionary *> * seed = [NSMutableArray array];
        for (APMessage * m in messages) {
            if (m.role == APRoleTool) {
                [self deliver:^{ completion(apGoogleError(APErrorInvalidMessage,
                    @"tool messages are not supported in prime (v1)")); }];
                return;
            }
            for (APContent * c in m.content) {
                if (c.kind != APContentKindText) {
                    [self deliver:^{ completion(apGoogleError(APErrorUnsupportedContent,
                        @"this backend accepts text content only (v1)")); }];
                    return;
                }
            }
            if (m.role == APRoleSystem) [system addObject:m.textRepresentation];
            else [seed addObject:@{ @"role"  : m.role == APRoleUser ? @"user" : @"model",
                                    @"parts" : @[ @{ @"text" : m.textRepresentation } ] }];
        }
        self->_systemInstruction = system.count ? [system componentsJoinedByString:@"\n\n"] : nil;
        [self->_contents addObjectsFromArray:seed];
        [self noteLastPrimeRestoredFromSnapshot:NO];
        [self appendTranscriptMessages:messages];
        [self deliver:^{ completion(nil); }];
    });
    return task;
}

#pragma mark - Respond

- (APResponseTask *)respondToMessage:(APMessage *)message
                             options:(APGenerationOptions *)options
                        deltaHandler:(void (^)(APResponseDelta *))deltaHandler
                          completion:(void (^)(APResponse *, NSError *))completion {
    APGenerationOptions * opts = [options copy] ?: [APGenerationOptions defaultOptions];
    APResponseTask * task = [[APResponseTask alloc]
        initWithProgress:[NSProgress progressWithTotalUnitCount:-1]];

    dispatch_async(_workQueue, ^{
        if (message.role != APRoleUser) {
            [self deliver:^{ completion(nil, apGoogleError(APErrorInvalidMessage,
                @"respond requires a user message (v1)")); }];
            return;
        }
        for (APContent * c in message.content) {
            if (c.kind != APContentKindText) {
                [self deliver:^{ completion(nil, apGoogleError(APErrorUnsupportedContent,
                    @"this backend accepts text content only (v1)")); }];
                return;
            }
        }
        // The work queue stays blocked until the turn (all rounds) finishes — that is
        // the serialization contract, same shape as the local backend's engine thread.
        dispatch_semaphore_t turnDone = dispatch_semaphore_create(0);
        [self runTurnWithMessage:message options:opts task:task
                    deltaHandler:deltaHandler completion:completion
                        finished:^{ dispatch_semaphore_signal(turnDone); }];
        dispatch_semaphore_wait(turnDone, DISPATCH_TIME_FOREVER);
    });
    return task;
}

- (void)runTurnWithMessage:(APMessage *)message
                   options:(APGenerationOptions *)opts
                      task:(APResponseTask *)task
              deltaHandler:(void (^)(APResponseDelta *))deltaHandler
                completion:(void (^)(APResponse *, NSError *))completion
                  finished:(dispatch_block_t)finished {
    NSArray<id<APTool>> * activeTools = [self toolsSortedByName];
    NSMutableArray<NSDictionary *> * contents = [_contents mutableCopy];
    [contents addObject:@{ @"role" : @"user",
                           @"parts" : @[ @{ @"text" : message.textRepresentation } ] }];

    NSMutableString * answer = [NSMutableString string];
    NSMutableString * reasoning = [NSMutableString string];
    NSMutableArray<NSString *> * executedTools = [NSMutableArray array];
    NSDate * t0 = [NSDate date];
    __block NSDate * tFirst = nil;
    __block BOOL sawThought = NO, answerStarted = NO;

    __weak APGoogleSession * weakSelf = self;
    __block int quotaRetries = 0;
    // Declared __block so the block can recurse for tool rounds.
    __block void (^sendRound)(NSMutableArray<NSDictionary *> *, int) = nil;
    sendRound = ^(NSMutableArray<NSDictionary *> * roundContents, int round) {
        // A self-recursive __block block must NOT nil its own reference mid-execution
        // (ARC would free the executing block); this strong local pins it to frame end.
        void (^pinnedSelf)(NSMutableArray<NSDictionary *> *, int) = sendRound;
        (void)pinnedSelf;
        APGoogleSession * self = weakSelf;
        if (!self) { sendRound = nil; return; }

        NSError * reqError = nil;
        NSURLRequest * request = [self requestWithContents:roundContents
                                                     tools:activeTools options:opts error:&reqError];
        if (!request) {
            [self deliver:^{ completion(nil, reqError); }];
            sendRound = nil; finished();
            return;
        }

        NSMutableArray<NSDictionary *> * callParts = [NSMutableArray array];
        __block BOOL roundStreamedAnything = NO;
        APGoogleStreamingRound * sse = [[APGoogleStreamingRound alloc] init];
        sse.isCancelled = ^{ return [task isCancelled]; };
        sse.onPart = ^(NSDictionary * part) {
            roundStreamedAnything = YES;
            if (!tFirst) tFirst = [NSDate date];
            if ([part[@"functionCall"] isKindOfClass:NSDictionary.class]) {
                [callParts addObject:part];   // replayed VERBATIM (keeps thoughtSignature)
                return;
            }
            NSString * text = [part[@"text"] isKindOfClass:NSString.class] ? part[@"text"] : nil;
            if (text.length == 0) return;
            BOOL thought = [part[@"thought"] boolValue];
            if (thought) { sawThought = YES; [reasoning appendString:text]; }
            else {
                // Trim the answer's leading newlines, like the local stream shape.
                if (!answerStarted) {
                    NSUInteger i = 0;
                    while (i < text.length && [text characterAtIndex:i] == '\n') i++;
                    if (i == text.length) return;
                    text = [text substringFromIndex:i];
                    answerStarted = YES;
                }
                [answer appendString:text];
            }
            if (!deltaHandler) return;
            APResponseDelta * delta = [[APResponseDelta alloc]
                initWithText:text tokenCount:0 isThought:thought];
            [self deliver:^{ deltaHandler(delta); }];
        };
        sse.onDone = ^(NSString * apiFinish, NSDictionary * usage, NSError * error) {
            APGoogleSession * self = weakSelf;
            if (!self) { sendRound = nil; return; }
            if (error) {
                // 429 quota trips get one honest wait-and-resend: the API says when to
                // come back, and a round that streamed NOTHING is safe to repeat. This
                // is what rescues tool continuations — round 2 rides the same minute's
                // budget as round 1 and trips first (persona rides every request).
                NSInteger status = [error.userInfo[kAPGoogleHTTPStatusKey] integerValue];
                double delay = [error.userInfo[kAPGoogleRetryAfterKey] doubleValue];
                if (delay <= 0) delay = kDefaultRetryDelay;
                if (status == 429 && !roundStreamedAnything &&
                    quotaRetries < kMaxQuotaRetries && delay <= kMaxRetryDelay &&
                    ![task isCancelled]) {
                    quotaRetries += 1;
                    dispatch_after(dispatch_time(DISPATCH_TIME_NOW,
                                                 (int64_t)((delay + 0.5) * NSEC_PER_SEC)),
                                   dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                        sendRound(roundContents, round);
                    });
                    return;
                }
                [self deliver:^{ completion(nil, error); }];
                sendRound = nil; finished();
                return;
            }

            // ---- tool round: dispatch every completed call, splice, request again ----
            if (callParts.count && activeTools.count && round < kMaxToolRounds &&
                ![task isCancelled]) {
                [roundContents addObject:@{ @"role" : @"model", @"parts" : [callParts copy] }];
                NSMutableArray<NSDictionary *> * responseParts = [NSMutableArray array];
                for (NSDictionary * part in callParts) {
                    NSDictionary * call = part[@"functionCall"];
                    NSString * name = [call[@"name"] isKindOfClass:NSString.class] ? call[@"name"] : @"";
                    NSDictionary * args = [call[@"args"] isKindOfClass:NSDictionary.class]
                        ? call[@"args"] : @{};
                    NSDictionary * value = [self dispatchToolNamed:name arguments:args
                                                             among:activeTools executed:executedTools];
                    [responseParts addObject:@{ @"functionResponse" :
                                                    @{ @"name" : name, @"response" : value } }];
                }
                [roundContents addObject:@{ @"role" : @"user", @"parts" : responseParts }];
                sendRound(roundContents, round + 1);
                return;
            }

            // ---- finalize ----
            APFinishReason reason = [task isCancelled] ? APFinishReasonCancelled
                : [apiFinish isEqualToString:@"MAX_TOKENS"] ? APFinishReasonMaxTokens
                                                            : APFinishReasonEndOfTurn;
            APMessage * reply = [APMessage assistantMessageWithText:[answer copy]];
            if (answer.length || reason == APFinishReasonCancelled) {
                [roundContents addObject:@{ @"role" : @"model",
                                            @"parts" : @[ @{ @"text" : [answer copy] } ] }];
            }
            [self->_contents setArray:roundContents];
            [self appendTranscriptMessages:@[ message, reply ]];

            NSInteger promptTok = [usage[@"promptTokenCount"] integerValue];
            NSInteger respTok = [usage[@"candidatesTokenCount"] integerValue]
                              + [usage[@"thoughtsTokenCount"] integerValue];
            @synchronized(self) {
                self->_lastTotalTokens = [usage[@"totalTokenCount"] integerValue];
            }
            NSTimeInterval ttft = tFirst ? [tFirst timeIntervalSinceDate:t0] : 0;
            NSTimeInterval decodeS = tFirst ? -[tFirst timeIntervalSinceNow] : 0;
            APResponseStats * stats = [[APResponseStats alloc]
                initWithPromptTokens:promptTok responseTokens:respTok
                    timeToFirstToken:ttft
                          prefillTPS:(ttft > 0 ? promptTok / ttft : 0)
                           decodeTPS:(decodeS > 0 ? respTok / decodeS : 0)];
            APResponse * response = [[APResponse alloc]
                initWithMessage:reply finishReason:reason stats:stats
                      reasoning:(sawThought && reasoning.length ? [reasoning copy] : nil)
              executedToolNames:executedTools];
            [self deliver:^{ completion(response, nil); }];
            sendRound = nil; finished();
        };
        [sse startWithRequest:request];
    };
    sendRound(contents, 0);
}

/// Veto → invoke → notify, byte-for-byte the local backend's mediation semantics.
/// Runs on the SSE delegate queue; blocks it until the tool's completion fires (tools
/// do IO — the documented serialization, same as the local engine thread).
- (NSDictionary *)dispatchToolNamed:(NSString *)name
                          arguments:(NSDictionary *)args
                              among:(NSArray<id<APTool>> *)activeTools
                           executed:(NSMutableArray<NSString *> *)executedTools {
    id<APTool> tool = nil;
    for (id<APTool> t in activeTools)
        if ([t.name isEqualToString:name]) { tool = t; break; }

    id<APSessionDelegate> delegate = self.delegate;
    BOOL allowed = tool != nil;
    if (allowed && [delegate respondsToSelector:@selector(session:shouldInvokeTool:arguments:)])
        allowed = [delegate session:self shouldInvokeTool:tool arguments:args];

    __block NSString * resultText = nil;
    __block NSError * toolError = nil;
    if (allowed) {
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [tool invokeWithArguments:args completion:^(APContent * result, NSError * error) {
            resultText = result.text;
            toolError = error;
            dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        [executedTools addObject:name];
    }
    if ([delegate respondsToSelector:@selector(session:didInvokeTool:arguments:result:)]) {
        NSString * info = allowed ? (resultText ?: toolError.localizedDescription ?: @"")
                                  : @"(vetoed)";
        [self deliver:^{ [delegate session:self didInvokeTool:name arguments:args result:info]; }];
    }
    if (toolError) return @{ @"error" : toolError.localizedDescription ?: @"tool failed" };
    if (!allowed)  return @{ @"error" : @"tool unavailable" };
    return @{ @"result" : resultText ?: @"" };
}

#pragma mark - Request construction

- (NSURLRequest *)requestWithContents:(NSArray<NSDictionary *> *)contents
                                tools:(NSArray<id<APTool>> *)activeTools
                              options:(APGenerationOptions *)opts
                                error:(NSError **)error {
    NSString * key = _keyProvider ? _keyProvider() : nil;
    if (key.length == 0) {
        if (error) *error = apGoogleError(APErrorMissingAPIKey,
            @"no Gemini API key — the key provider returned nothing");
        return nil;
    }

    NSMutableDictionary * body = [NSMutableDictionary dictionary];
    body[@"contents"] = contents;
    if (_systemInstruction.length)
        body[@"systemInstruction"] = @{ @"parts" : @[ @{ @"text" : _systemInstruction } ] };
    if (activeTools.count) {
        NSMutableArray * decls = [NSMutableArray array];
        for (id<APTool> tool in activeTools) {
            NSMutableDictionary * d = [NSMutableDictionary dictionary];
            d[@"name"] = tool.name;
            d[@"description"] = tool.toolDescription;
            // The API rejects an object schema with zero properties — omit it instead.
            NSDictionary * schema = tool.parameterSchema;
            if ([schema[@"properties"] isKindOfClass:NSDictionary.class] &&
                [(NSDictionary *)schema[@"properties"] count] > 0)
                d[@"parameters"] = schema;
            [decls addObject:d];
        }
        body[@"tools"] = @[ @{ @"functionDeclarations" : decls } ];
    }
    NSMutableDictionary * gen = [NSMutableDictionary dictionary];
    gen[@"temperature"] = @(opts.temperature);
    if (opts.topK > 0)  gen[@"topK"] = @(opts.topK);
    if (opts.topP < 1)  gen[@"topP"] = @(opts.topP);
    if (opts.maximumResponseTokens > 0) gen[@"maxOutputTokens"] = @(opts.maximumResponseTokens);
    gen[@"thinkingConfig"] = self.reasoningEnabled
        ? @{ @"thinkingLevel" : @"high",    @"includeThoughts" : @YES }
        : @{ @"thinkingLevel" : @"minimal" };
    body[@"generationConfig"] = gen;

    NSData * payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:nil];
    if (!payload) {
        if (error) *error = apGoogleError(APErrorRemoteService, @"could not encode the request");
        return nil;
    }
    NSURL * url = [NSURL URLWithString:[NSString stringWithFormat:
        @"%@/%@:streamGenerateContent?alt=sse", kAPGoogleBaseURL, _modelName]];
    NSMutableURLRequest * request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = @"POST";
    request.HTTPBody = payload;
    [request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
    [request setValue:key forHTTPHeaderField:@"x-goog-api-key"];
    request.timeoutInterval = 300;
    return request;
}

@end
