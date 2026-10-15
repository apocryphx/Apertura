//  APSessionDispatchTests — the shared tool mediation and message validation in
//  APSession(Subclass), exercised through the remote backend (no network, no model):
//  veto → invoke (blocking) → didInvokeTool notify → tool-call record bookkeeping.
#import <XCTest/XCTest.h>
#import <AperturaKit/AperturaKit.h>
#import "APSessionSubclass.h"

@interface APDispatchHandler : NSObject
@property (nonatomic) int calls;
@property (nonatomic, copy) NSDictionary * lastArgs;
@property (nonatomic) BOOL fail;
@end
@implementation APDispatchHandler
- (void)handleArgs:(NSDictionary<NSString *, id> *)args
        completion:(void (^)(APContent *, NSError *))completion {
    self.calls += 1;
    self.lastArgs = args;
    // Complete asynchronously: the dispatcher must block until this fires.
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        if (self.fail)
            completion(nil, [NSError errorWithDomain:APErrorDomain code:APErrorToolFailed
                                            userInfo:@{ NSLocalizedDescriptionKey : @"boom" }]);
        else
            completion([APContent textContent:@"AZURE-HERON-42"], nil);
    });
}
@end

@interface APDispatchDelegate : NSObject <APSessionDelegate>
@property (nonatomic) BOOL allow;
@property (nonatomic) int shouldCount, didCount;
@property (nonatomic, copy) NSString * lastName, * lastResult;
@property (nonatomic, strong) XCTestExpectation * notified;
@end
@implementation APDispatchDelegate
- (BOOL)session:(APSession *)session shouldInvokeTool:(id<APTool>)tool
      arguments:(NSDictionary<NSString *, id> *)arguments {
    self.shouldCount += 1;
    return self.allow;
}
- (void)session:(APSession *)session didInvokeTool:(NSString *)toolName
      arguments:(NSDictionary<NSString *, id> *)arguments result:(NSString *)result {
    self.didCount += 1;
    self.lastName = toolName;
    self.lastResult = result;
    [self.notified fulfill];
}
@end

@interface APSessionDispatchTests : XCTestCase
@end

@implementation APSessionDispatchTests {
    APGoogleSession * _session;
    APDispatchHandler * _handler;
    APDispatchDelegate * _delegate;
    NSArray<id<APTool>> * _tools;
    APTurnRecorder * _recorder;
    NSMutableArray<APResponseDelta *> * _deltas;
}

- (void)setUp {
    _session = [[APGoogleSession alloc] initWithModelName:@"gemma-4-31b-it"
                                           apiKeyProvider:^NSString * { return nil; }];
    _session.callbackQueue = dispatch_queue_create("test.dispatch.cb", DISPATCH_QUEUE_SERIAL);
    _handler = [[APDispatchHandler alloc] init];
    _delegate = [[APDispatchDelegate alloc] init];
    _delegate.allow = YES;
    _session.delegate = _delegate;
    [_session registerTool:[APSelectorTool toolWithName:@"lookup" toolDescription:@"d"
                                        parameterSchema:@{} target:_handler
                                                 action:@selector(handleArgs:completion:)]];
    _tools = [_session toolsSortedByName];
    _deltas = [NSMutableArray array];
    NSMutableArray<APResponseDelta *> * deltas = _deltas;
    _recorder = [[APTurnRecorder alloc] initWithSession:_session
                                           deltaHandler:^(APResponseDelta * d) { [deltas addObject:d]; }];
}

- (void)testDispatchInvokesNotifiesAndRecords {
    _delegate.notified = [self expectationWithDescription:@"didInvoke"];
    APToolOutcome * o = [_session dispatchToolNamed:@"lookup" arguments:@{ @"k" : @1 }
                                              among:_tools recorder:_recorder];
    XCTAssertFalse(o.vetoed);
    XCTAssertEqualObjects(o.resultText, @"AZURE-HERON-42");   // completion already fired: blocking
    XCTAssertNil(o.error);
    XCTAssertEqual(_handler.calls, 1);
    XCTAssertEqualObjects(_handler.lastArgs, @{ @"k" : @1 });
    XCTAssertEqual(_recorder.parts.count, 1u);
    XCTAssertEqual(_recorder.parts.firstObject.kind, APContentKindToolCall);
    XCTAssertEqual(_recorder.parts.firstObject.toolCall, o.record);
    XCTAssertEqualObjects(o.record.name, @"lookup");
    XCTAssertEqualObjects(o.record.arguments, @{ @"k" : @1 });
    XCTAssertEqual(o.record.status, APToolCallStatusSucceeded);
    XCTAssertEqualObjects(o.record.result, @"AZURE-HERON-42");
    XCTAssertGreaterThan(o.record.duration, 0);
    // The call streamed as a typed delta, after the delegate notification.
    [self waitForExpectations:@[ _delegate.notified ] timeout:5];
    dispatch_sync(_session.callbackQueue, ^{});
    XCTAssertEqual(_deltas.count, 1u);
    XCTAssertEqual(_deltas.firstObject.kind, APResponseDeltaKindToolCall);
    XCTAssertEqual(_deltas.firstObject.toolCall, o.record);
    XCTAssertEqualObjects(_deltas.firstObject.text, @"");
    XCTAssertFalse(_deltas.firstObject.isThought);
    XCTAssertEqual(_delegate.shouldCount, 1);
    XCTAssertEqualObjects(_delegate.lastName, @"lookup");
    XCTAssertEqualObjects(_delegate.lastResult, @"AZURE-HERON-42");
}

- (void)testDispatchVetoedByDelegate {
    _delegate.allow = NO;
    _delegate.notified = [self expectationWithDescription:@"didInvoke"];
    APToolOutcome * o = [_session dispatchToolNamed:@"lookup" arguments:@{}
                                              among:_tools recorder:_recorder];
    XCTAssertTrue(o.vetoed);
    XCTAssertNil(o.resultText);
    XCTAssertNil(o.error);
    XCTAssertEqual(_handler.calls, 0);
    XCTAssertEqual(_recorder.parts.count, 1u);         // the record shows what was asked
    XCTAssertEqual(o.record.status, APToolCallStatusVetoed);
    XCTAssertNil(o.record.result);
    XCTAssertEqual(o.record.duration, 0);
    [self waitForExpectations:@[ _delegate.notified ] timeout:5];
    XCTAssertEqualObjects(_delegate.lastResult, @"(vetoed)");
}

- (void)testDispatchUnknownToolIsVetoedWithoutAskingDelegate {
    _delegate.notified = [self expectationWithDescription:@"didInvoke"];
    APToolOutcome * o = [_session dispatchToolNamed:@"nope" arguments:@{}
                                              among:_tools recorder:_recorder];
    XCTAssertTrue(o.vetoed);
    XCTAssertEqualObjects(o.record.name, @"nope");
    XCTAssertEqual(_handler.calls, 0);
    [self waitForExpectations:@[ _delegate.notified ] timeout:5];
    XCTAssertEqual(_delegate.shouldCount, 0);
    XCTAssertEqualObjects(_delegate.lastName, @"nope");
}

- (void)testDispatchSurfacesToolError {
    _handler.fail = YES;
    _delegate.notified = [self expectationWithDescription:@"didInvoke"];
    APToolOutcome * o = [_session dispatchToolNamed:@"lookup" arguments:@{}
                                              among:_tools recorder:_recorder];
    XCTAssertFalse(o.vetoed);
    XCTAssertEqual(o.error.code, APErrorToolFailed);
    XCTAssertEqual(o.record.status, APToolCallStatusFailed);   // it ran, even though it failed
    XCTAssertEqualObjects(o.record.result, @"boom");
    [self waitForExpectations:@[ _delegate.notified ] timeout:5];
    XCTAssertEqualObjects(_delegate.lastResult, @"boom");
}

- (void)testMessageValidation {
    APSession * s = _session;
    NSArray * fine = @[ [APMessage systemMessageWithText:@"p"],
                        [APMessage userMessageWithText:@"u"],
                        [APMessage assistantMessageWithText:@"a"] ];
    XCTAssertNil([s validationErrorForPrimeMessages:fine]);
    NSArray * toolTurn = @[ [APMessage messageWithRole:APRoleTool
                                               content:@[ [APContent textContent:@"t"] ]] ];
    NSError * e = [s validationErrorForPrimeMessages:toolTurn];
    XCTAssertEqualObjects(e.domain, APErrorDomain);
    XCTAssertEqual(e.code, APErrorInvalidMessage);
    XCTAssertNil([s validationErrorForRespondMessage:[APMessage userMessageWithText:@"u"]]);
    XCTAssertEqual([s validationErrorForRespondMessage:[APMessage assistantMessageWithText:@"a"]].code,
                   APErrorInvalidMessage);
    // A replayed v2 transcript primes: attachments ride as text, records are carried.
    APAttachment * file = [APAttachment attachmentWithName:@"n.md" text:@"body" sizeBytes:4
                                     contentTypeIdentifier:nil];
    APToolCall * call = [APToolCall toolCallWithName:@"lookup" arguments:@{} status:APToolCallStatusSucceeded
                                              result:@"r" duration:0.1];
    NSArray * replay = @[ [APMessage messageWithRole:APRoleUser
                                             content:@[ [APContent attachmentContent:file],
                                                        [APContent textContent:@"read it"] ]],
                          [APMessage messageWithRole:APRoleAssistant
                                             content:@[ [APContent reasoningContent:@"hm"],
                                                        [APContent toolCallContent:call],
                                                        [APContent textContent:@"done"] ]] ];
    XCTAssertNil([s validationErrorForPrimeMessages:replay]);
    XCTAssertNil([s validationErrorForRespondMessage:replay[0]]);
    XCTAssertEqual(APSessionMaxToolRounds, 4);
}

@end
