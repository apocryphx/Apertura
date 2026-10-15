//  APTranscriptTests — the v2 wire format: every content kind and the generation record
//  round-trip, v1 documents still decode, unknown parts are skipped, bytes are stable,
//  and the model-facing text excludes what the model must not see again.
#import <XCTest/XCTest.h>
#import <AperturaKit/AperturaKit.h>
#import "APSessionSubclass.h"   // +assistantMessageWithAnswer:… (the backends' shared shape)

@interface APTranscriptTests : XCTestCase
@end

@implementation APTranscriptTests

static NSDictionary * apDocument(NSData * data) {
    return [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
}

- (NSArray<APMessage *> *)richConversation {
    APAttachment * file = [APAttachment attachmentWithName:@"notes.md"
                                                      text:@"# Notes\nline two"
                                                 sizeBytes:17
                                     contentTypeIdentifier:@"net.daringfireball.markdown"];
    APToolCall * ran = [APToolCall toolCallWithName:@"search_notes"
                                          arguments:@{ @"query" : @"heron", @"limit" : @3 }
                                             status:APToolCallStatusSucceeded
                                             result:@"AZURE-HERON-42"
                                           duration:0.25];
    APToolCall * vetoed = [APToolCall toolCallWithName:@"erase_everything" arguments:@{}
                                                status:APToolCallStatusVetoed result:nil duration:0];
    APToolCall * failed = [APToolCall toolCallWithName:@"fetch" arguments:@{ @"url" : @"x" }
                                                status:APToolCallStatusFailed result:@"boom"
                                              duration:1.5];
    return @[
        [APMessage systemMessageWithText:@"persona"],
        [APMessage messageWithRole:APRoleUser
                           content:@[ [APContent attachmentContent:file],
                                      [APContent textContent:@"what does it say — ümläuts 🜂"] ]],
        [APSession assistantMessageWithAnswer:@"It says notes.\nTwo lines."
                                    reasoning:@"The user attached a file; read it first."
                                    toolCalls:@[ ran, vetoed, failed ]
                                        stats:[self sampleStats]
                              modelIdentifier:@"google/gemma-4-12b-it-qat"],
    ];
}

/// Decoding builds APResponseStats through the kit's project initializer; the test
/// target has no access to it, so a transcript round-trip is how it gets one here.
- (APResponseStats *)sampleStats {
    NSString * doc = @"{\"version\":2,\"messages\":[{\"role\":\"assistant\",\"content\":[],"
        "\"stats\":{\"prompt_tokens\":1345,\"response_tokens\":1104,\"ttft\":0.665,"
        "\"prefill_tps\":2021.5,\"decode_tps\":15.72}}]}";
    int16_t v = 0;
    return [APTranscript messagesFromData:[doc dataUsingEncoding:NSUTF8StringEncoding]
                                  version:&v].firstObject.stats;
}

#pragma mark - Round trip

- (void)testV2RoundTripKeepsEveryKindAndTheRecord {
    NSArray<APMessage *> * messages = [self richConversation];
    NSData * data = [APTranscript dataFromMessages:messages];
    XCTAssertNotNil(data);
    int16_t version = 0;
    NSArray<APMessage *> * back = [APTranscript messagesFromData:data version:&version];
    XCTAssertEqual(version, 2);
    XCTAssertEqual(version, APTranscriptCurrentSchemaVersion);
    XCTAssertEqual(back.count, 3u);

    // User turn: attachment then text, both intact.
    APMessage * user = back[1];
    XCTAssertEqual(user.content.count, 2u);
    XCTAssertEqual(user.content[0].kind, APContentKindAttachment);
    APAttachment * file = user.content[0].attachment;
    XCTAssertEqualObjects(file.name, @"notes.md");
    XCTAssertEqualObjects(file.text, @"# Notes\nline two");
    XCTAssertEqual(file.sizeBytes, 17ull);
    XCTAssertEqualObjects(file.contentTypeIdentifier, @"net.daringfireball.markdown");
    XCTAssertEqualObjects(user.displayText, @"what does it say — ümläuts 🜂");
    XCTAssertEqual(user.attachments.count, 1u);
    XCTAssertNil(user.stats);
    XCTAssertNil(user.modelIdentifier);

    // Assistant turn: reasoning, three tool records in order, answer; stats and model.
    APMessage * reply = back[2];
    XCTAssertEqual(reply.content.count, 5u);
    XCTAssertEqual(reply.content[0].kind, APContentKindReasoning);
    XCTAssertEqualObjects(reply.reasoning, @"The user attached a file; read it first.");
    NSArray<APToolCall *> * calls = reply.toolCalls;
    XCTAssertEqual(calls.count, 3u);
    XCTAssertEqualObjects(calls[0].name, @"search_notes");
    XCTAssertEqualObjects(calls[0].arguments, (@{ @"query" : @"heron", @"limit" : @3 }));
    XCTAssertEqual(calls[0].status, APToolCallStatusSucceeded);
    XCTAssertEqualObjects(calls[0].result, @"AZURE-HERON-42");
    XCTAssertEqualWithAccuracy(calls[0].duration, 0.25, 1e-9);
    XCTAssertEqual(calls[1].status, APToolCallStatusVetoed);
    XCTAssertNil(calls[1].result);
    XCTAssertEqual(calls[1].duration, 0);
    XCTAssertEqual(calls[2].status, APToolCallStatusFailed);
    XCTAssertEqualObjects(calls[2].result, @"boom");
    XCTAssertEqualObjects(reply.displayText, @"It says notes.\nTwo lines.");
    XCTAssertEqualObjects(reply.modelIdentifier, @"google/gemma-4-12b-it-qat");
    XCTAssertEqual(reply.stats.promptTokenCount, 1345);
    XCTAssertEqual(reply.stats.responseTokenCount, 1104);
    XCTAssertEqualWithAccuracy(reply.stats.timeToFirstToken, 0.665, 1e-9);
    XCTAssertEqualWithAccuracy(reply.stats.prefillTokensPerSecond, 2021.5, 1e-9);
    XCTAssertEqualWithAccuracy(reply.stats.decodeTokensPerSecond, 15.72, 1e-9);

    // Deterministic bytes: identical transcripts encode identically.
    XCTAssertEqualObjects(data, [APTranscript dataFromMessages:back]);
}

- (void)testWireNamesAreTheDocumentedStrings {
    NSDictionary * doc = apDocument([APTranscript dataFromMessages:[self richConversation]]);
    XCTAssertEqualObjects(doc[@"version"], @2);
    NSArray * turns = doc[@"messages"];
    NSArray * userParts = turns[1][@"content"];
    XCTAssertEqualObjects(userParts[0][@"kind"], @"attachment");
    XCTAssertEqualObjects(userParts[0][@"type"], @"net.daringfireball.markdown");
    XCTAssertEqualObjects(userParts[0][@"size"], @17);
    NSDictionary * reply = turns[2];
    XCTAssertEqualObjects(reply[@"model"], @"google/gemma-4-12b-it-qat");
    XCTAssertEqualObjects(reply[@"stats"][@"prompt_tokens"], @1345);
    NSArray * parts = reply[@"content"];
    XCTAssertEqualObjects(parts[0][@"kind"], @"reasoning");
    XCTAssertEqualObjects(parts[1][@"kind"], @"tool_call");
    XCTAssertEqualObjects(parts[1][@"status"], @"ok");
    XCTAssertEqualObjects(parts[2][@"status"], @"vetoed");
    XCTAssertNil(parts[2][@"result"]);
    XCTAssertNil(parts[2][@"duration"]);
    XCTAssertEqualObjects(parts[3][@"status"], @"error");
    XCTAssertEqualObjects(parts[4][@"kind"], @"text");
    // Plain turns carry no record keys at all (a v1 reader sees exactly a v1 turn).
    XCTAssertNil(turns[0][@"stats"]);
    XCTAssertNil(turns[0][@"model"]);
}

#pragma mark - Compatibility

- (void)testV1DocumentDecodesUnchanged {
    NSString * doc = @"{\"version\":1,\"messages\":["
        "{\"role\":\"user\",\"content\":[{\"kind\":\"text\",\"text\":\"hello\"}]},"
        "{\"role\":\"assistant\",\"content\":[{\"kind\":\"text\",\"text\":\"hi\"}]}]}";
    int16_t version = 0;
    NSArray<APMessage *> * back =
        [APTranscript messagesFromData:[doc dataUsingEncoding:NSUTF8StringEncoding] version:&version];
    XCTAssertEqual(version, 1);
    XCTAssertEqual(back.count, 2u);
    XCTAssertEqualObjects(back[1].displayText, @"hi");
    XCTAssertNil(back[1].reasoning);
    XCTAssertEqual(back[1].toolCalls.count, 0u);
    XCTAssertNil(back[1].stats);
}

- (void)testMalformedV2PartsAreSkippedNotGuessed {
    NSString * doc = @"{\"version\":2,\"messages\":[{\"role\":\"assistant\",\"content\":["
        "{\"kind\":\"tool_call\",\"name\":\"x\",\"status\":\"teleported\"},"   // unknown status
        "{\"kind\":\"tool_call\",\"status\":\"ok\"},"                           // no name
        "{\"kind\":\"attachment\",\"name\":\"a\"},"                             // no text
        "{\"kind\":\"reasoning\"},"                                             // no text
        "{\"kind\":\"image\",\"text\":\"reserved\"},"                           // reserved kind
        "{\"kind\":\"tool_call\",\"name\":\"ok\",\"status\":\"ok\",\"arguments\":[1,2]},"  // args not an object
        "{\"kind\":\"text\",\"text\":\"kept\"}],"
        "\"stats\":\"not a dict\",\"model\":7}]}";
    int16_t version = 0;
    NSArray<APMessage *> * back =
        [APTranscript messagesFromData:[doc dataUsingEncoding:NSUTF8StringEncoding] version:&version];
    XCTAssertEqual(back.count, 1u);
    XCTAssertEqual(back[0].content.count, 2u);
    XCTAssertEqualObjects(back[0].toolCalls.firstObject.name, @"ok");
    XCTAssertEqualObjects(back[0].toolCalls.firstObject.arguments, @{});
    XCTAssertEqualObjects(back[0].displayText, @"kept");
    XCTAssertNil(back[0].stats);
    XCTAssertNil(back[0].modelIdentifier);
}

#pragma mark - What the model sees

- (void)testPromptTextFramesAttachmentsAndHidesTheRecord {
    NSArray<APMessage *> * messages = [self richConversation];
    APMessage * user = messages[1];
    XCTAssertEqualObjects(user.textRepresentation,
                          @"--- attached file: notes.md ---\n# Notes\nline two\n--- end of notes.md ---\n\n"
                          @"what does it say — ümläuts 🜂");
    APMessage * reply = messages[2];
    XCTAssertEqualObjects(reply.textRepresentation, @"It says notes.\nTwo lines.");   // no reasoning, no tools
    XCTAssertNil([APContent reasoningContent:@"t"].promptText);
    XCTAssertNil([APContent toolCallContent:reply.toolCalls[0]].promptText);
}

- (void)testAssistantMessageShapeOmitsEmptyReasoning {
    APMessage * m = [APSession assistantMessageWithAnswer:@"a" reasoning:nil toolCalls:@[]
                                                    stats:nil modelIdentifier:nil];
    XCTAssertEqual(m.content.count, 1u);
    XCTAssertEqual(m.content[0].kind, APContentKindText);
    XCTAssertNil(m.reasoning);
    APMessage * e = [APSession assistantMessageWithAnswer:@"a" reasoning:@"" toolCalls:@[]
                                                    stats:nil modelIdentifier:nil];
    XCTAssertEqual(e.content.count, 1u);
}

@end
