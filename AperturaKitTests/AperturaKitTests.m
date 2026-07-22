//
//  AperturaKitTests.m
//  AperturaKitTests
//
//  Created by Kolja Wawrowsky on 7/21/26.
//
//  Fast tests run everywhere (value types, availability pre-flight, options).
//  The end-to-end session test needs a real model and is gated by the environment:
//    APERTURAKIT_TEST_MODEL=/path/to/model.apml   (or an HF snapshot directory)
//  Byte-identity vs the engine reference path is the CLI's job (--facade-verify);
//  these tests cover the public contract.

#import <XCTest/XCTest.h>
#import <AperturaKit/AperturaKit.h>

@interface AperturaKitTests : XCTestCase
@end

@implementation AperturaKitTests

#pragma mark - Fast: availability pre-flight

- (void)testAvailabilityOfMissingModel {
    NSURL * bogus = [NSURL fileURLWithPath:@"/nonexistent/model.apml"];
    XCTAssertEqual([APModel availabilityOfModelAtURL:bogus configuration:nil], APModelNotFound);
}

- (void)testLoadOfMissingModelFailsWithError {
    NSError * err = nil;
    APModel * m = [APModel modelWithContentsOfURL:[NSURL fileURLWithPath:@"/nonexistent"]
                                    configuration:nil error:&err];
    XCTAssertNil(m);
    XCTAssertEqualObjects(err.domain, APErrorDomain);
    XCTAssertEqual(err.code, APErrorModelNotFound);
}

#pragma mark - Fast: value types

- (void)testMessageRolesAndTextRepresentation {
    APMessage * m = [APMessage userMessageWithText:@"hello"];
    XCTAssertEqual(m.role, APRoleUser);
    XCTAssertEqual(m.content.count, 1u);
    XCTAssertEqual(m.content.firstObject.kind, APContentKindText);
    XCTAssertEqualObjects(m.textRepresentation, @"hello");

    APMessage * multi = [APMessage messageWithRole:APRoleSystem
                                           content:@[ [APContent textContent:@"a"],
                                                      [APContent textContent:@"b"] ]];
    XCTAssertEqualObjects(multi.textRepresentation, @"ab");
}

- (void)testGenerationOptionsDefaultsAndDeterminism {
    APGenerationOptions * chat = [APGenerationOptions defaultOptions];
    XCTAssertGreaterThan(chat.temperature, 0);
    APGenerationOptions * det = [APGenerationOptions deterministicOptions];
    XCTAssertEqual(det.temperature, 0);
    APGenerationOptions * copy = [det copy];
    copy.maximumResponseTokens = 7;
    XCTAssertEqual(det.maximumResponseTokens, 0);   // copy is independent
}

- (void)testModelConfigurationDefaults {
    APModelConfiguration * c = [APModelConfiguration defaultConfiguration];
    XCTAssertEqual(c.headBits, 8);                  // quality-first default
    XCTAssertEqual(c.prefillChunkLength, 512);      // roadmap P5 default
    XCTAssertFalse(c.instrumented);
}

#pragma mark - Gated: end-to-end session (needs APERTURAKIT_TEST_MODEL)

- (void)testSessionEndToEnd {
    NSString * modelPath = NSProcessInfo.processInfo.environment[@"APERTURAKIT_TEST_MODEL"];
    if (modelPath.length == 0) {
        XCTSkip(@"set APERTURAKIT_TEST_MODEL to run the end-to-end session test");
    }
    NSError * err = nil;
    APModel * model = [APModel modelWithContentsOfURL:[NSURL fileURLWithPath:modelPath]
                                        configuration:nil error:&err];
    XCTAssertNotNil(model, @"%@", err);

    APSession * session = [[APSession alloc] initWithModel:model];
    dispatch_queue_t cbq = dispatch_queue_create("test.cb", DISPATCH_QUEUE_SERIAL);
    session.callbackQueue = cbq;

    XCTestExpectation * primed = [self expectationWithDescription:@"primed"];
    [session primeWithMessages:@[ [APMessage systemMessageWithText:
        @"You are a terse assistant. Answer in one short sentence."] ]
                    completion:^(NSError * e) { XCTAssertNil(e); [primed fulfill]; }];
    [self waitForExpectations:@[ primed ] timeout:600];

    APGenerationOptions * opts = [APGenerationOptions deterministicOptions];
    opts.maximumResponseTokens = 48;

    NSMutableString * streamed = [NSMutableString string];
    __block APResponse * response = nil;
    XCTestExpectation * done = [self expectationWithDescription:@"responded"];
    [session respondToMessage:[APMessage userMessageWithText:@"What is the capital of France?"]
                      options:opts
                 deltaHandler:^(APResponseDelta * d) { [streamed appendString:d.text]; }
                   completion:^(APResponse * r, NSError * e) {
                       XCTAssertNil(e);
                       response = r;
                       [done fulfill];
                   }];
    [self waitForExpectations:@[ done ] timeout:600];

    XCTAssertNotNil(response);
    XCTAssertEqual(response.message.role, APRoleAssistant);
    XCTAssertGreaterThan(response.message.textRepresentation.length, 0u);
    XCTAssertGreaterThan(streamed.length, 0u);
    XCTAssertTrue(response.finishReason == APFinishReasonEndOfTurn ||
                  response.finishReason == APFinishReasonMaxTokens);
    XCTAssertGreaterThan(response.stats.decodeTokensPerSecond, 0);
    XCTAssertEqual(session.transcript.count, 3u);   // system + user + assistant
    XCTAssertGreaterThan(session.contextTokenCount, 0);
}

@end
