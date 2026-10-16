//  APStreamingBenchmarkTests — what a delta costs while a long Markdown answer streams.
//
//  The view re-reads `blocks` after every delta, on the callback queue, so the per-delta
//  cost is what bounds the frame budget. Streams a ~25 KB answer (headings, styled
//  paragraphs, nested lists, Swift code, tables, quotes) in token-sized pieces through an
//  APLiveMessage and asserts the per-delta percentiles stay under budget; alongside, it
//  samples what re-parsing the whole text would have cost, so the log shows the gain.
#import "APChatModelTestSupport.h"

@interface APStreamingBenchmarkTests : XCTestCase
@end

@implementation APStreamingBenchmarkTests

/// The budgets, in milliseconds. Measured on this Mac (M4 Max, Debug build, 2026-10-15):
/// 8444 deltas over 33.8 K characters, incremental p50 0.015 ms, p95 0.03 ms, worst
/// 0.07 ms; a whole-text re-parse of the same growing answer sat at 1.6 ms per delta
/// (3 ms by the end). Generous multiples of the measurement, so a loaded machine does not
/// fail the suite while a regression to O(length) per delta still does.
static const double APBudgetP95Milliseconds = 0.5;
static const double APBudgetMaxMilliseconds = 2.0;

- (NSString *)longAnswer {
    NSMutableString * answer = [NSMutableString string];
    for (NSInteger section = 1; section <= 40; section++) {
        [answer appendFormat:@"## Section %ld: Brewing variable %ld\n\n", (long)section, (long)section];
        [answer appendString:@"A clean cup comes down to **ratio**, **grind**, and *patience*; see "
                             @"[the guide](https://sca.coffee) and use `20 g` of coffee for ~~300~~ 320 g of water. "
                             @"Pour slowly, in stages, and keep the bed level so extraction stays even.\n\n"];
        [answer appendFormat:@"1. Rinse the filter.\n2. Add coffee (section %ld).\n3. Pour in stages:\n"
                             @"   - Bloom with 40 g for 30 s\n   - Pour to 320 g in slow circles\n4. Let it drain.\n\n",
                             (long)section];
        [answer appendString:@"```swift\nstruct Brew {\n    let ratio: Double // water per gram\n    var stages: [String] = [\"bloom\", \"pour\", \"drain\"]\n"
                             @"    func run() async throws {\n        for stage in stages { print(\"Now: \\(stage)\") }\n    }\n}\n```\n\n"];
        [answer appendString:@"| Setting | Value | Why |\n|---|---|---|\n| Ratio | 1:16 | Balanced strength |\n"
                             @"| Grind | Medium-fine | Controls flow rate |\n| Water | 94 °C | Extracts without bitterness |\n\n"];
        [answer appendString:@"> Tip: If it tastes sour, grind finer.\n\n---\n\n"];
    }
    return answer;
}

/// Token-sized pieces (about four characters), never splitting a composed character.
- (NSArray<NSString *> *)tokensOf:(NSString *)text {
    NSMutableArray * tokens = [NSMutableArray array];
    NSUInteger offset = 0;
    while (offset < text.length) {
        NSRange piece = [text rangeOfComposedCharacterSequencesForRange:NSMakeRange(offset, MIN(4u, text.length - offset))];
        [tokens addObject:[text substringWithRange:piece]];
        offset = NSMaxRange(piece);
    }
    return tokens;
}

- (void)testPerDeltaCostStaysUnderBudget {
    NSString * answer = [self longAnswer];
    NSArray<NSString *> * tokens = [self tokensOf:answer];

    APLiveMessage * live = [[APLiveMessage alloc] init];
    [live appendDelta:[APResponseDelta reasoningDelta:@"Write a long, structured answer."]];

    NSMutableArray<NSNumber *> * costs = [NSMutableArray arrayWithCapacity:tokens.count];
    NSMutableArray<NSNumber *> * fullCosts = [NSMutableArray array];
    NSMutableString * soFar = [NSMutableString string];
    double total = 0;
    for (NSUInteger i = 0; i < tokens.count; i++) {
        NSString * token = tokens[i];
        NSDate * start = [NSDate date];
        [live appendDelta:[APResponseDelta textDelta:token]];
        (void)live.blocks;   // what the view reads after each delta
        double ms = -start.timeIntervalSinceNow * 1000;
        [costs addObject:@(ms)];
        total += ms;

        // Every 200th delta: the alternative, re-parsing everything so far.
        [soFar appendString:token];
        if (i % 200 == 199) {
            NSDate * fullStart = [NSDate date];
            (void)[APMarkdownDocument documentWithMarkdown:soFar].blocks;
            [fullCosts addObject:@(-fullStart.timeIntervalSinceNow * 1000)];
        }
    }

    NSArray<NSNumber *> * sorted = [costs sortedArrayUsingSelector:@selector(compare:)];
    double p50 = sorted[sorted.count / 2].doubleValue;
    double p95 = sorted[(NSUInteger)(sorted.count * 0.95)].doubleValue;
    double max = sorted.lastObject.doubleValue;
    NSArray<NSNumber *> * fullSorted = [fullCosts sortedArrayUsingSelector:@selector(compare:)];
    NSLog(@"APBENCH chars=%lu deltas=%lu blocks=%lu incremental: total=%.0fms mean=%.3fms p50=%.3fms p95=%.3fms max=%.2fms | "
          @"full re-parse sampled %lu×: p50=%.2fms max=%.2fms",
          (unsigned long)answer.length, (unsigned long)tokens.count,
          (unsigned long)[APAs(live.blocks.lastObject, APTextBlock) document].blocks.count,
          total, total / tokens.count, p50, p95, max,
          (unsigned long)fullCosts.count, fullSorted[fullSorted.count / 2].doubleValue, fullSorted.lastObject.doubleValue);

    XCTAssertLessThan(p95, APBudgetP95Milliseconds, @"p95 per delta");
    XCTAssertLessThan(max, APBudgetMaxMilliseconds, @"worst delta");

    // And the result is still the archived rendering.
    APMessage * archived = [APMessage messageWithRole:APRoleAssistant
                                              content:@[ [APContent reasoningContent:@"Write a long, structured answer."],
                                                         [APContent textContent:answer] ]
                                                stats:nil modelIdentifier:nil];
    XCTAssertEqualObjects(APDumpChatBlocks(live.blocks),
                          APDumpChatBlocks([APDisplayMessage displayMessageWithMessage:archived].blocks));
}

@end
