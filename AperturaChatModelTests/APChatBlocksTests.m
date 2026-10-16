//  APChatBlocksTests — messages to blocks (one per part, in order), the tool-call layout
//  and its preview limit, and the live message: fed the deltas of a turn it shows the same
//  blocks as the archived message, and it finalizes to the message APTurnRecorder records.
#import "APChatModelTestSupport.h"
#import "APSessionSubclass.h"   // APTurnRecorder — the backends' shared record

@interface APChatBlocksTests : XCTestCase
@end

@implementation APChatBlocksTests

#pragma mark - Fixtures

- (APToolCall *)searchCall {
    return [APToolCall toolCallWithName:@"memory_search"
                              arguments:@{ @"query" : @"coffee preferences", @"limit" : @3 }
                                 status:APToolCallStatusSucceeded
                                 result:@"Prefers light roasts, uses a V60."
                               duration:0.124];
}

- (APToolCall *)vetoedCall {
    return [APToolCall toolCallWithName:@"erase_everything" arguments:@{}
                                 status:APToolCallStatusVetoed result:nil duration:0];
}

/// A turn as the recorder stores it: think, call, think again, call, answer.
- (NSArray<APResponseDelta *> *)interleavedDeltas {
    return @[
        [APResponseDelta reasoningDelta:@"The user asked about "],
        [APResponseDelta reasoningDelta:@"coffee; check memory."],
        [APResponseDelta toolCallDelta:[self searchCall]],
        [APResponseDelta reasoningDelta:@"Found it. Nothing to erase."],
        [APResponseDelta toolCallDelta:[self vetoedCall]],
        [APResponseDelta textDelta:@"## Pour-Over\n\nUse a **1:16** ratio."],
        [APResponseDelta textDelta:@"\n\n- bloom\n- pour"],
    ];
}

- (APMessage *)recordedMessageFromDeltas:(NSArray<APResponseDelta *> *)deltas {
    APTurnRecorder * recorder = [[APTurnRecorder alloc] initWithSession:nil deltaHandler:nil];
    for (APResponseDelta * delta in deltas) [recorder emit:delta];
    return [recorder assistantMessageWithStats:nil modelIdentifier:@"google/gemma-4-12b-it-qat"];
}

#pragma mark - Display

- (void)testInterleavedMessageBecomesBlocksInOrder {
    APMessage * message = [self recordedMessageFromDeltas:[self interleavedDeltas]];
    APDisplayMessage * display = [APDisplayMessage displayMessageWithMessage:message];
    XCTAssertEqual(display.message, message);

    NSArray<APChatBlock *> * blocks = display.blocks;
    XCTAssertEqual(blocks.count, 5u);
    XCTAssertEqualObjects([APAs(blocks[0], APReasoningBlock) text], @"The user asked about coffee; check memory.");
    XCTAssertEqualObjects([APAs(blocks[1], APToolCallBlock) toolCall].name, @"memory_search");
    XCTAssertEqualObjects([APAs(blocks[2], APReasoningBlock) text], @"Found it. Nothing to erase.");
    XCTAssertEqual([APAs(blocks[3], APToolCallBlock) toolCall].status, APToolCallStatusVetoed);

    APMarkdownDocument * document = [APAs(blocks[4], APTextBlock) document];
    XCTAssertEqualObjects(document.source, @"## Pour-Over\n\nUse a **1:16** ratio.\n\n- bloom\n- pour");
    XCTAssertEqual(document.blocks.count, 4u);
    XCTAssertEqual([APAs(document.blocks[0], APMarkdownHeadingBlock) level], 2);
    XCTAssertEqualObjects([APAs(document.blocks[3], APMarkdownListItemBlock) text].string, @"pour");
}

- (void)testUserMessageWithAttachment {
    APAttachment * file = [APAttachment attachmentWithName:@"notes.md" text:@"# Notes" sizeBytes:7
                                     contentTypeIdentifier:@"net.daringfireball.markdown"];
    APMessage * message = [APMessage messageWithRole:APRoleUser
                                             content:@[ [APContent attachmentContent:file],
                                                        [APContent textContent:@"What does it say?"] ]];
    NSArray<APChatBlock *> * blocks = [APDisplayMessage displayMessageWithMessage:message].blocks;
    XCTAssertEqual(blocks.count, 2u);
    XCTAssertEqual([APAs(blocks[0], APAttachmentBlock) attachment], file);
    XCTAssertEqualObjects([[APAs(blocks[1], APTextBlock) document].blocks.firstObject
                           isKindOfClass:APMarkdownParagraphBlock.class] ? @"paragraph" : @"other", @"paragraph");
}

- (void)testUserMessageWithImageBecomesAnImageBlock {
    APImage * present = [APImage imageWithIdentifier:@"abc" mediaType:@"image/png" byteCount:1234
                                           pixelSize:CGSizeMake(640, 400) originalName:@"sketch.png"
                                             fileURL:[NSURL fileURLWithPath:@"/tmp/abc.png"]];
    APImage * missing = [APImage imageWithIdentifier:@"def" mediaType:@"image/jpeg" byteCount:2048
                                           pixelSize:CGSizeZero originalName:nil fileURL:nil];
    APMessage * message = [APMessage messageWithRole:APRoleUser
                                             content:@[ [APContent imageContent:present],
                                                        [APContent imageContent:missing],
                                                        [APContent textContent:@"What is this?"] ]];
    NSArray<APChatBlock *> * blocks = [APDisplayMessage displayMessageWithMessage:message].blocks;
    XCTAssertEqual(blocks.count, 3u, @"an image is a block of its own, never dropped");
    APImageBlock * first = APAs(blocks[0], APImageBlock);
    XCTAssertEqual(first.image, present);
    XCTAssertEqualObjects(first.displayName, @"sketch.png");
    XCTAssertEqualObjects(first.detail, @"640×400 · 1 KB");
    APImageBlock * second = APAs(blocks[1], APImageBlock);
    XCTAssertNil(second.image.fileURL, @"a missing file still shows as a placeholder");
    XCTAssertEqualObjects(second.displayName, @"Image");
    XCTAssertEqualObjects(second.detail, @"2 KB", @"no dimensions known");
    XCTAssertNotNil(APAs(blocks[2], APTextBlock));
}

#pragma mark - Tool calls

- (void)testToolCallBlockLaysOutArgumentsAndResult {
    APToolCallBlock * block = [[APToolCallBlock alloc] initWithToolCall:[self searchCall]];
    XCTAssertEqualObjects(block.argumentsSnippet.language, @"json");
    XCTAssertEqualObjects(block.argumentsSnippet.code, @"{\n  \"limit\": 3,\n  \"query\": \"coffee preferences\"\n}");
    XCTAssertTrue([APTokens(block.argumentsSnippet.highlighted) containsObject:APToken(@"\"query\"", APCodeTokenKindAttribute)]);
    XCTAssertEqualObjects(block.resultPreview.code, @"Prefers light roasts, uses a V60.");
    XCTAssertNil(block.resultPreview.language, @"Prose results are plain");
    XCTAssertFalse(block.resultTruncated);

    APToolCallBlock * vetoed = [[APToolCallBlock alloc] initWithToolCall:[self vetoedCall]];
    XCTAssertEqualObjects(vetoed.argumentsSnippet.code, @"{}");
    XCTAssertNil(vetoed.resultPreview);
    XCTAssertFalse(vetoed.resultTruncated);
}

- (void)testToolResultJSONIsReadableAndTruncated {
    APToolCall * json = [APToolCall toolCallWithName:@"man" arguments:@{} status:APToolCallStatusSucceeded
                                              result:@"{\"man\": \"Usage:\\n  arc list\", \"ok\": true}" duration:0.1];
    APToolCallBlock * readable = [[APToolCallBlock alloc] initWithToolCall:json];
    XCTAssertEqualObjects(readable.resultPreview.language, @"json");
    XCTAssertEqualObjects(readable.resultPreview.code, @"{\n  \"man\":\n    Usage:\n      arc list\n  \"ok\": true\n}");

    APToolCall * flat = [APToolCall toolCallWithName:@"t" arguments:@{} status:APToolCallStatusSucceeded
                                              result:@"[1, 2, 3]" duration:0.1];
    XCTAssertEqualObjects([[APToolCallBlock alloc] initWithToolCall:flat].resultPreview.language, @"json",
                          @"Single-line JSON stays as written, but highlighted");

    NSString * longResult = [@"" stringByPaddingToLength:APToolCallBlock.resultPreviewLimit + 10 withString:@"x" startingAtIndex:0];
    APToolCall * failed = [APToolCall toolCallWithName:@"t" arguments:@{} status:APToolCallStatusFailed
                                                result:longResult duration:2];
    APToolCallBlock * truncated = [[APToolCallBlock alloc] initWithToolCall:failed];
    XCTAssertTrue(truncated.resultTruncated);
    XCTAssertEqual(truncated.resultPreview.code.length, APToolCallBlock.resultPreviewLimit);
    XCTAssertNil(truncated.resultPreview.language);
}

#pragma mark - Live

- (void)testLiveMessageShowsTheArchivedBlocks {
    NSArray<APResponseDelta *> * deltas = [self interleavedDeltas];
    APLiveMessage * live = [[APLiveMessage alloc] init];
    NSUInteger previousCount = 0;
    for (APResponseDelta * delta in deltas) {
        [live appendDelta:delta];
        XCTAssertGreaterThanOrEqual(live.blocks.count, previousCount, @"Blocks only accumulate");
        previousCount = live.blocks.count;
    }

    APMessage * recorded = [self recordedMessageFromDeltas:deltas];
    XCTAssertEqualObjects(APDumpChatBlocks(live.blocks),
                          APDumpChatBlocks([APDisplayMessage displayMessageWithMessage:recorded].blocks));

    // The live message finalizes to the recorder's message, byte for byte in the transcript.
    APMessage * finished = [live messageWithStats:nil modelIdentifier:@"google/gemma-4-12b-it-qat"];
    XCTAssertEqualObjects([APTranscript dataFromMessages:@[ finished ]], [APTranscript dataFromMessages:@[ recorded ]]);
    XCTAssertEqual(finished.content.count, 5u);
    XCTAssertEqualObjects(finished.modelIdentifier, @"google/gemma-4-12b-it-qat");
}

- (void)testLiveMessageStreamedCharacterByCharacter {
    // The whole sample answer in one-character deltas, after a reasoning part and a call.
    APLiveMessage * live = [[APLiveMessage alloc] init];
    [live appendDelta:[APResponseDelta reasoningDelta:@"Plan the answer."]];
    [live appendDelta:[APResponseDelta toolCallDelta:[self searchCall]]];
    for (NSUInteger i = 0; i < APSampleMarkdown.length; i++) {
        [live appendDelta:[APResponseDelta textDelta:[APSampleMarkdown substringWithRange:NSMakeRange(i, 1)]]];
    }
    XCTAssertEqual(live.blocks.count, 3u);

    APTurnRecorder * recorder = [[APTurnRecorder alloc] initWithSession:nil deltaHandler:nil];
    [recorder emit:[APResponseDelta reasoningDelta:@"Plan the answer."]];
    [recorder emit:[APResponseDelta toolCallDelta:[self searchCall]]];
    [recorder emit:[APResponseDelta textDelta:APSampleMarkdown]];
    APMessage * recorded = [recorder assistantMessageWithStats:nil modelIdentifier:nil];
    XCTAssertEqualObjects(APDumpChatBlocks(live.blocks),
                          APDumpChatBlocks([APDisplayMessage displayMessageWithMessage:recorded].blocks));
    XCTAssertEqualObjects([APTranscript dataFromMessages:@[ [live messageWithStats:nil modelIdentifier:nil] ]],
                          [APTranscript dataFromMessages:@[ recorded ]]);
}

- (void)testEmptyLiveMessage {
    APLiveMessage * live = [[APLiveMessage alloc] init];
    XCTAssertEqual(live.blocks.count, 0u);
    XCTAssertEqual(live.content.count, 0u);
    APMessage * message = [live messageWithStats:nil modelIdentifier:nil];
    XCTAssertEqual(message.role, APRoleAssistant);
    XCTAssertEqual(message.content.count, 0u);
}

@end
