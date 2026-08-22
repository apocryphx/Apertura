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
#import "APInternal.h"   // lastResponseTokenIDsForTesting (checkpoint round-trip gate)

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

#pragma mark - Fast: remote backend contract (no network, no key)

- (void)testGoogleSessionPrimeIsInstantAndIgnoresCacheURL {
    APGoogleSession * s = [[APGoogleSession alloc] initWithModelName:@"gemma-4-31b-it"
                                                      apiKeyProvider:^NSString * { return nil; }];
    dispatch_queue_t cbq = dispatch_queue_create("test.google.cb", DISPATCH_QUEUE_SERIAL);
    s.callbackQueue = cbq;
    XCTestExpectation * primed = [self expectationWithDescription:@"primed"];
    NSURL * bogusCache = [NSURL fileURLWithPath:@"/nonexistent/snapshot.safetensors"];
    [s primeWithMessages:@[ [APMessage systemMessageWithText:@"persona"],
                            [APMessage userMessageWithText:@"seed"] ]
                cacheURL:bogusCache
              completion:^(NSError * e) { XCTAssertNil(e); [primed fulfill]; }];
    [self waitForExpectations:@[ primed ] timeout:10];
    XCTAssertEqual(s.transcript.count, 2u);            // prime messages recorded
    XCTAssertFalse(s.lastPrimeRestoredFromSnapshot);   // cacheURL is a local-only hint
    XCTAssertEqual(s.contextTokenCount, 0);            // no request made yet
}

- (void)testGoogleSessionRespondWithoutKeyFails {
    APGoogleSession * s = [[APGoogleSession alloc] initWithModelName:@"gemma-4-31b-it"
                                                      apiKeyProvider:^NSString * { return nil; }];
    dispatch_queue_t cbq = dispatch_queue_create("test.google.cb2", DISPATCH_QUEUE_SERIAL);
    s.callbackQueue = cbq;
    XCTestExpectation * done = [self expectationWithDescription:@"responded"];
    [s respondToMessage:[APMessage userMessageWithText:@"hello"] options:nil
           deltaHandler:nil
             completion:^(APResponse * r, NSError * e) {
                 XCTAssertNil(r);
                 XCTAssertEqualObjects(e.domain, APErrorDomain);
                 XCTAssertEqual(e.code, APErrorMissingAPIKey);
                 [done fulfill];
             }];
    [self waitForExpectations:@[ done ] timeout:10];
    XCTAssertEqual(s.transcript.count, 0u);            // failed turn leaves no transcript
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

    APLocalSession * session = [[APLocalSession alloc] initWithModel:model];
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

#pragma mark - Gated: session checkpoint round-trip (needs APERTURAKIT_TEST_MODEL)

// The device-checkpoint license to exist: a session restored from checkpoint must
// continue TOKEN-IDENTICALLY to the same session had it never quit. Session A primes,
// takes one turn, checkpoints; A then continues live (the control). Session B restores
// the checkpoint and takes the same second turn — the two second-turn token streams
// must be equal (restore is byte-identical, greedy is deterministic).
- (void)testSessionCheckpointRoundTrip {
    NSString * modelPath = NSProcessInfo.processInfo.environment[@"APERTURAKIT_TEST_MODEL"];
    if (modelPath.length == 0) {
        XCTSkip(@"set APERTURAKIT_TEST_MODEL to run the checkpoint round-trip test");
    }
    NSError * err = nil;
    APModel * model = [APModel modelWithContentsOfURL:[NSURL fileURLWithPath:modelPath]
                                        configuration:nil error:&err];
    XCTAssertNotNil(model, @"%@", err);

    APGenerationOptions * opts = [APGenerationOptions deterministicOptions];
    opts.maximumResponseTokens = 32;
    APMessage * sys   = [APMessage systemMessageWithText:
        @"You are a terse assistant. Answer in one short sentence."];
    APMessage * turn1 = [APMessage userMessageWithText:@"What is the capital of France?"];
    APMessage * turn2 = [APMessage userMessageWithText:@"And of Italy?"];
    NSUUID * sid = [NSUUID UUID];
    dispatch_queue_t cbq = dispatch_queue_create("test.cb", DISPATCH_QUEUE_SERIAL);

    id (^respond)(APLocalSession *, APMessage *) = ^id (APLocalSession * s, APMessage * m) {
        XCTestExpectation * done = [self expectationWithDescription:@"responded"];
        __block APResponse * response = nil;
        [s respondToMessage:m options:opts deltaHandler:^(APResponseDelta * d) {}
                 completion:^(APResponse * r, NSError * e) {
                     XCTAssertNil(e); response = r; [done fulfill];
                 }];
        [self waitForExpectations:@[ done ] timeout:600];
        return response;
    };

    // Session A: prime, one turn, checkpoint, then continue live (control).
    APLocalSession * a = [[APLocalSession alloc] initWithModel:model];
    a.callbackQueue = cbq;
    XCTestExpectation * primed = [self expectationWithDescription:@"primed"];
    [a primeWithMessages:@[ sys ] completion:^(NSError * e) { XCTAssertNil(e); [primed fulfill]; }];
    [self waitForExpectations:@[ primed ] timeout:600];
    (void) respond(a, turn1);

    NSArray<APMessage *> * transcriptAtCheckpoint = a.transcript;   // system + user + assistant
    NSInteger posAtCheckpoint = a.contextTokenCount;
    XCTAssertTrue([a saveCheckpointForSessionID:sid]);
    XCTAssertEqualObjects([APLocalSession checkpointedSessionIDForModel:model], sid);

    APResponse * control = respond(a, turn2);
    NSArray<NSNumber *> * controlIds = [a lastResponseTokenIDsForTesting];

    // Session B: fresh, restore, same second turn.
    APLocalSession * b = [[APLocalSession alloc] initWithModel:model];
    b.callbackQueue = cbq;
    XCTestExpectation * restored = [self expectationWithDescription:@"restored"];
    [b restoreCheckpointForSessionID:sid messages:transcriptAtCheckpoint
                          completion:^(NSError * e) { XCTAssertNil(e); [restored fulfill]; }];
    [self waitForExpectations:@[ restored ] timeout:600];
    XCTAssertEqual(b.contextTokenCount, posAtCheckpoint);
    XCTAssertEqual(b.transcript.count, transcriptAtCheckpoint.count);

    APResponse * replay = respond(b, turn2);
    NSArray<NSNumber *> * replayIds = [b lastResponseTokenIDsForTesting];

    XCTAssertEqualObjects(replayIds, controlIds,
        @"restored continuation diverged from the live session");
    XCTAssertEqualObjects(replay.message.textRepresentation, control.message.textRepresentation);

    // A restore under the wrong session id must refuse.
    APLocalSession * c = [[APLocalSession alloc] initWithModel:model];
    c.callbackQueue = cbq;
    XCTestExpectation * refused = [self expectationWithDescription:@"refused"];
    [c restoreCheckpointForSessionID:[NSUUID UUID] messages:transcriptAtCheckpoint
                          completion:^(NSError * e) { XCTAssertNotNil(e); [refused fulfill]; }];
    [self waitForExpectations:@[ refused ] timeout:600];

    [APLocalSession removeDeviceCheckpoint];
    XCTAssertNil([APLocalSession checkpointedSessionIDForModel:model]);
}

#pragma mark - M1: transcript codec, seed, checkpoint URLs

- (void)testTranscriptRoundTrip {
    NSArray<APMessage *> * messages = @[
        [APMessage systemMessageWithText:@"persona"],
        [APMessage userMessageWithText:@"hello — with ümläuts and 🜂"],
        [APMessage assistantMessageWithText:@"reply\nwith lines"],
    ];
    NSData * data = [APTranscript dataFromMessages:messages];
    XCTAssertNotNil(data);
    int16_t version = 0;
    NSArray<APMessage *> * back = [APTranscript messagesFromData:data version:&version];
    XCTAssertEqual(version, APTranscriptCurrentSchemaVersion);
    XCTAssertEqual(back.count, messages.count);
    for (NSUInteger i = 0; i < messages.count; ++i) {
        XCTAssertEqual(back[i].role, messages[i].role);
        XCTAssertEqualObjects(back[i].textRepresentation, messages[i].textRepresentation);
    }
    // Deterministic bytes: identical transcripts encode identically.
    XCTAssertEqualObjects(data, [APTranscript dataFromMessages:back]);
}

- (void)testTranscriptToleratesUnknownRolesAndKinds {
    NSString * doc = @"{\"version\":9,\"messages\":["
        "{\"role\":\"user\",\"content\":[{\"kind\":\"text\",\"text\":\"kept\"}]},"
        "{\"role\":\"oracle\",\"content\":[{\"kind\":\"text\",\"text\":\"dropped\"}]},"
        "{\"role\":\"assistant\",\"content\":[{\"kind\":\"hologram\",\"text\":\"skipped\"},"
                                            "{\"kind\":\"text\",\"text\":\"kept too\"}]}]}";
    int16_t version = 0;
    NSArray<APMessage *> * back =
        [APTranscript messagesFromData:[doc dataUsingEncoding:NSUTF8StringEncoding]
                               version:&version];
    XCTAssertEqual(version, 9);              // the WRITER's version, preserved
    XCTAssertEqual(back.count, 2);           // unknown role dropped
    XCTAssertEqualObjects(back[1].textRepresentation, @"kept too");  // unknown kind skipped
}

- (void)testGenerationOptionsCopyCarriesSeed {
    APGenerationOptions * o = [APGenerationOptions defaultOptions];
    o.seed = 0xC0FFEEULL;
    APGenerationOptions * c = [o copy];
    XCTAssertEqual(c.seed, 0xC0FFEEULL);
    XCTAssertEqual([APGenerationOptions defaultOptions].seed, 0ULL);  // default unchanged
}

- (void)testCheckpointURLDerivation {
    // Device wrappers and URL forms agree; sidecar swaps the extension.
    NSURL * dev = [APLocalSession deviceCheckpointURL];
    XCTAssertEqualObjects(dev.pathExtension, @"safetensors");
    NSURL * url = [NSURL fileURLWithPath:@"/tmp/checkpoints/ABC-123.safetensors"];
    // No public sidecar accessor for arbitrary URLs on the session; removal of a
    // never-written checkpoint must be a no-op rather than an error.
    XCTAssertNoThrow([APLocalSession removeCheckpointAtURL:url]);
}

@end
