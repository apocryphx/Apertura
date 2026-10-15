//  APGoogleSessionLiveTests — the remote backend against the real Gemini API.
//
//  Gated by the environment, like the local model tests:
//    APERTURAKIT_GEMINI_API_KEY_FILE=/path/to/key.txt   (file holding the key, trimmed)
//    APERTURAKIT_GEMINI_API_KEY=<key>                   (alternative: the key itself)
//  Without either the tests skip. These are smoke tests of the live wire format
//  (streaming, function calling, thinking parts): a real model answers, so they assert
//  shape and the tool datum, never exact text. Exact-path coverage (429 retry, cancel,
//  chunk framing) is the job of a model-free fake; see the session notes.
#import <XCTest/XCTest.h>
#import <AperturaKit/AperturaKit.h>

static NSString * const kLiveModel = @"gemma-4-31b-it";

@interface APLiveCodewordHandler : NSObject
@property (nonatomic) int invocations;
@end
@implementation APLiveCodewordHandler
- (void)handleArgs:(NSDictionary<NSString *, id> *)args
        completion:(void (^)(APContent *, NSError *))completion {
    self.invocations += 1;
    completion([APContent textContent:@"AZURE-HERON-42"], nil);
}
@end

@interface APLiveDelegate : NSObject <APSessionDelegate>
@property (nonatomic) int didInvokeCount;
@property (nonatomic, copy) NSString * lastToolName;
@end
@implementation APLiveDelegate
- (void)session:(APSession *)session didInvokeTool:(NSString *)toolName
      arguments:(NSDictionary<NSString *, id> *)arguments result:(NSString *)result {
    self.didInvokeCount += 1;
    self.lastToolName = toolName;
}
@end

@interface APGoogleSessionLiveTests : XCTestCase
@end

@implementation APGoogleSessionLiveTests {
    NSString * _key;
    dispatch_queue_t _cbq;
}

- (void)setUp {
    NSDictionary * env = NSProcessInfo.processInfo.environment;
    NSString * key = env[@"APERTURAKIT_GEMINI_API_KEY"];
    NSString * file = env[@"APERTURAKIT_GEMINI_API_KEY_FILE"];
    if (key.length == 0 && file.length)
        key = [NSString stringWithContentsOfFile:file encoding:NSUTF8StringEncoding error:nil];
    key = [key stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    if (key.length == 0) XCTSkip(@"set APERTURAKIT_GEMINI_API_KEY(_FILE) to run the live Gemini tests");
    _key = key;
    _cbq = dispatch_queue_create("test.google.live", DISPATCH_QUEUE_SERIAL);
}

- (APGoogleSession *)sessionWithReasoning:(BOOL)reasoning {
    NSString * key = _key;
    APGoogleSession * s = [[APGoogleSession alloc] initWithModelName:kLiveModel
                                                      apiKeyProvider:^NSString * { return key; }];
    s.callbackQueue = _cbq;
    s.reasoningEnabled = reasoning;
    return s;
}

- (void)prime:(APSession *)s with:(NSString *)persona {
    XCTestExpectation * primed = [self expectationWithDescription:@"primed"];
    [s primeWithMessages:@[ [APMessage systemMessageWithText:persona] ]
              completion:^(NSError * e) { XCTAssertNil(e, @"%@", e); [primed fulfill]; }];
    [self waitForExpectations:@[ primed ] timeout:30];
}

- (APResponse *)respond:(APSession *)s to:(NSString *)text
                deltas:(void (^)(APResponseDelta *))deltas {
    return [self respond:s to:text budget:256 deltas:deltas];
}

- (APResponse *)respond:(APSession *)s to:(NSString *)text budget:(NSInteger)budget
                deltas:(void (^)(APResponseDelta *))deltas {
    XCTestExpectation * done = [self expectationWithDescription:@"responded"];
    __block APResponse * resp = nil;
    APGenerationOptions * opts = [APGenerationOptions deterministicOptions];
    opts.maximumResponseTokens = budget;
    [s respondToMessage:[APMessage userMessageWithText:text] options:opts
           deltaHandler:deltas
             completion:^(APResponse * r, NSError * e) {
                 XCTAssertNil(e, @"%@", e); resp = r; [done fulfill]; }];
    [self waitForExpectations:@[ done ] timeout:180];
    return resp;
}

- (void)testLivePlainTurnStreamsAndRecordsTranscript {
    APGoogleSession * s = [self sessionWithReasoning:NO];
    [self prime:s with:@"You are a terse assistant. Answer in one short sentence."];
    NSMutableString * streamed = [NSMutableString string];
    APResponse * r = [self respond:s to:@"What colour is a ripe banana?"
                            deltas:^(APResponseDelta * d) {
                                XCTAssertFalse(d.isThought);
                                [streamed appendString:d.text];
                            }];
    XCTAssertNotNil(r);
    XCTAssertEqual(r.message.role, APRoleAssistant);
    XCTAssertGreaterThan(r.message.textRepresentation.length, 0u);
    XCTAssertEqualObjects(streamed, r.message.textRepresentation);   // deltas reassemble to the answer
    XCTAssertEqual(r.finishReason, APFinishReasonEndOfTurn);
    XCTAssertGreaterThan(r.stats.promptTokenCount, 0);
    XCTAssertGreaterThan(s.contextTokenCount, 0);
    XCTAssertEqual(s.transcript.count, 3u);   // system + user + assistant
    XCTAssertEqual(r.executedToolNames.count, 0u);
}

- (void)testLiveToolRoundDispatchesAndAnswersWithDatum {
    APGoogleSession * s = [self sessionWithReasoning:NO];
    APLiveCodewordHandler * handler = [[APLiveCodewordHandler alloc] init];
    APLiveDelegate * delegate = [[APLiveDelegate alloc] init];
    s.delegate = delegate;
    [s registerTool:[APSelectorTool
        toolWithName:@"lookup_codeword"
     toolDescription:@"Returns the current secret codeword. Call this whenever the user asks for the codeword."
     parameterSchema:@{ @"type" : @"object", @"properties" : @{} }
              target:handler action:@selector(handleArgs:completion:)]];
    [self prime:s with:@"You are a terse assistant. Use your tools when they can answer the question."];
    NSMutableString * streamed = [NSMutableString string];
    APResponse * r = [self respond:s to:@"Use the lookup_codeword tool and tell me the codeword exactly."
                            deltas:^(APResponseDelta * d) { [streamed appendString:d.text]; }];
    XCTAssertNotNil(r);
    XCTAssertGreaterThanOrEqual(handler.invocations, 1);
    XCTAssertTrue([r.executedToolNames containsObject:@"lookup_codeword"]);
    XCTAssertGreaterThanOrEqual(delegate.didInvokeCount, 1);
    XCTAssertEqualObjects(delegate.lastToolName, @"lookup_codeword");
    XCTAssertTrue([r.message.textRepresentation containsString:@"AZURE-HERON-42"], @"%@", r.message);
    XCTAssertTrue([streamed containsString:@"AZURE-HERON-42"]);
}

- (void)testLiveReasoningTurnSeparatesThoughtFromAnswer {
    APGoogleSession * s = [self sessionWithReasoning:YES];
    [self prime:s with:@"You are a careful assistant."];
    NSMutableString * thought = [NSMutableString string], * answer = [NSMutableString string];
    // Thinking is billed against the response budget; give it room, and accept a
    // budget-capped turn (then only the thought channel is guaranteed).
    APResponse * r = [self respond:s to:@"Is 91 a prime number? Think it through, then answer in one sentence."
                            budget:4096
                            deltas:^(APResponseDelta * d) {
                                [d.isThought ? thought : answer appendString:d.text];
                            }];
    XCTAssertNotNil(r);
    XCTAssertEqualObjects(answer, r.message.textRepresentation);
    if (r.finishReason == APFinishReasonEndOfTurn) XCTAssertGreaterThan(answer.length, 0u);
    else XCTAssertEqual(r.finishReason, APFinishReasonMaxTokens);
    // The API returns thought summaries when it thinks; they must land in `reasoning`,
    // the same text the isThought deltas streamed.
    XCTAssertGreaterThan(thought.length, 0u, @"finish=%ld", (long) r.finishReason);
    XCTAssertEqualObjects(thought, r.reasoning);
}

@end
