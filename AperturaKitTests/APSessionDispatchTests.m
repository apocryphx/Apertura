//  APSessionDispatchTests — the shared tool mediation and v1 message validation in
//  APSession(Subclass), exercised through the remote backend (no network, no model):
//  veto → invoke (blocking) → didInvokeTool notify → executed-name bookkeeping.
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
    NSMutableArray<NSString *> * _executed;
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
    _executed = [NSMutableArray array];
}

- (void)testDispatchInvokesNotifiesAndRecords {
    _delegate.notified = [self expectationWithDescription:@"didInvoke"];
    APToolOutcome * o = [_session dispatchToolNamed:@"lookup" arguments:@{ @"k" : @1 }
                                              among:_tools executed:_executed];
    XCTAssertFalse(o.vetoed);
    XCTAssertEqualObjects(o.resultText, @"AZURE-HERON-42");   // completion already fired: blocking
    XCTAssertNil(o.error);
    XCTAssertEqual(_handler.calls, 1);
    XCTAssertEqualObjects(_handler.lastArgs, @{ @"k" : @1 });
    XCTAssertEqualObjects(_executed, @[ @"lookup" ]);
    [self waitForExpectations:@[ _delegate.notified ] timeout:5];
    XCTAssertEqual(_delegate.shouldCount, 1);
    XCTAssertEqualObjects(_delegate.lastName, @"lookup");
    XCTAssertEqualObjects(_delegate.lastResult, @"AZURE-HERON-42");
}

- (void)testDispatchVetoedByDelegate {
    _delegate.allow = NO;
    _delegate.notified = [self expectationWithDescription:@"didInvoke"];
    APToolOutcome * o = [_session dispatchToolNamed:@"lookup" arguments:@{}
                                              among:_tools executed:_executed];
    XCTAssertTrue(o.vetoed);
    XCTAssertNil(o.resultText);
    XCTAssertNil(o.error);
    XCTAssertEqual(_handler.calls, 0);
    XCTAssertEqual(_executed.count, 0u);
    [self waitForExpectations:@[ _delegate.notified ] timeout:5];
    XCTAssertEqualObjects(_delegate.lastResult, @"(vetoed)");
}

- (void)testDispatchUnknownToolIsVetoedWithoutAskingDelegate {
    _delegate.notified = [self expectationWithDescription:@"didInvoke"];
    APToolOutcome * o = [_session dispatchToolNamed:@"nope" arguments:@{}
                                              among:_tools executed:_executed];
    XCTAssertTrue(o.vetoed);
    XCTAssertEqual(_handler.calls, 0);
    [self waitForExpectations:@[ _delegate.notified ] timeout:5];
    XCTAssertEqual(_delegate.shouldCount, 0);
    XCTAssertEqualObjects(_delegate.lastName, @"nope");
}

- (void)testDispatchSurfacesToolError {
    _handler.fail = YES;
    _delegate.notified = [self expectationWithDescription:@"didInvoke"];
    APToolOutcome * o = [_session dispatchToolNamed:@"lookup" arguments:@{}
                                              among:_tools executed:_executed];
    XCTAssertFalse(o.vetoed);
    XCTAssertEqual(o.error.code, APErrorToolFailed);
    XCTAssertEqualObjects(_executed, @[ @"lookup" ]);   // it ran, even though it failed
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
    XCTAssertEqual(APSessionMaxToolRounds, 4);
}

@end
