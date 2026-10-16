//  APMarkdownDocumentTests — Markdown blocks, semantic inline keys (ported from the
//  prototype's PCParsingTests), and the streaming parser: fed any way, it ends with the
//  blocks of the full parse, and its settled prefix only grows.
#import "APChatModelTestSupport.h"

@interface APMarkdownDocumentTests : XCTestCase
@end

@implementation APMarkdownDocumentTests

- (void)testBlockStructure {
    NSArray<APMarkdownBlock *> * blocks = [APMarkdownDocument documentWithMarkdown:APSampleMarkdown].blocks;

    APMarkdownHeadingBlock * heading = APAs(blocks.firstObject, APMarkdownHeadingBlock);
    XCTAssertEqual(heading.level, 2);
    XCTAssertEqualObjects(heading.text.string, @"Pour-Over in Five Steps");

    NSMutableArray<NSString *> * markers = [NSMutableArray array];
    NSMutableArray<NSNumber *> * depths = [NSMutableArray array];
    APMarkdownTableBlock * table = nil;
    APMarkdownParagraphBlock * quote = nil;
    APMarkdownCodeBlock * code = nil;
    NSUInteger rules = 0;
    for (APMarkdownBlock * block in blocks) {
        if ([block isKindOfClass:APMarkdownListItemBlock.class]) {
            APMarkdownListItemBlock * item = (APMarkdownListItemBlock *)block;
            [markers addObject:item.marker];
            [depths addObject:@(item.depth)];
        } else if ([block isKindOfClass:APMarkdownTableBlock.class]) {
            table = (APMarkdownTableBlock *)block;
        } else if ([block isKindOfClass:APMarkdownCodeBlock.class]) {
            code = (APMarkdownCodeBlock *)block;
        } else if ([block isKindOfClass:APMarkdownRuleBlock.class]) {
            rules++;
        } else if ([block isKindOfClass:APMarkdownParagraphBlock.class] && block.quoteLevel > 0) {
            quote = (APMarkdownParagraphBlock *)block;
        }
    }
    XCTAssertEqualObjects(markers, (@[ @"1.", @"2.", @"3.", @"•", @"•", @"4.", @"5." ]));
    XCTAssertEqualObjects(depths, (@[ @1, @1, @1, @2, @2, @1, @1 ]));

    XCTAssertEqual(table.table.columnCount, 3);
    XCTAssertEqual(table.table.rows.count, 3u);
    XCTAssertEqualObjects(table.table.header.firstObject.string, @"Setting");
    XCTAssertEqualObjects(table.table.rows[2][1].string, @"94 °C");

    XCTAssertEqual(quote.quoteLevel, 1);
    XCTAssertEqualObjects(quote.text.string, @"Tip: If it tastes sour, grind finer.");
    XCTAssertEqual(rules, 1u);

    XCTAssertEqualObjects(code.snippet.language, @"swift");
    XCTAssertFalse([code.snippet.code hasSuffix:@"\n"]);
    XCTAssertTrue([APTokens(code.snippet.highlighted) containsObject:APToken(@"for", APCodeTokenKindKeyword)]);
}

- (void)testInlineSemantics {
    NSArray<APMarkdownBlock *> * blocks = [APMarkdownDocument documentWithMarkdown:APSampleMarkdown].blocks;
    NSAttributedString * intro = [APAs(blocks[1], APMarkdownParagraphBlock) text];
    XCTAssertEqualObjects(APAttribute(intro, APSemanticFontWeightAttributeName, @"ratio"), @(APFontWeightBold));
    XCTAssertEqualObjects(APAttribute(intro, APSemanticItalicAttributeName, @"patience"), @YES);
    XCTAssertNil(APAttribute(intro, APSemanticFontWeightAttributeName, @"A clean cup"));
    // Semantic only: nothing visual, and nothing from the parser, is stored.
    XCTAssertEqual([intro attributesAtIndex:0 effectiveRange:NULL].count, 0u);

    APMarkdownListItemBlock * step2 = nil;
    for (APMarkdownBlock * block in blocks) {
        APMarkdownListItemBlock * item = APAs(block, APMarkdownListItemBlock);
        if ([item.marker isEqualToString:@"2."]) step2 = item;
    }
    APMarkdownFlowBlock * link = APAs(blocks.lastObject, APMarkdownFlowBlock);
    XCTAssertEqualObjects(APAttribute(step2.text, APSemanticCodeAttributeName, @"20 g"), @YES);
    XCTAssertEqualObjects(APAttribute(link.text, APSemanticLinkAttributeName, @"the SCA guide"), @"https://sca.coffee");
}

- (void)testEdgeCases {
    APMarkdownDocument * document = [APMarkdownDocument documentWithMarkdown:
        @"| A | B | C |\n|---|---|---|\n| one<br>two | | z |\n| x | ~~y~~ |\n\n- item\n\n  more\n\n> > deep"];
    NSArray<APMarkdownBlock *> * blocks = document.blocks;
    APMarkdownTable * table = [APAs(blocks[0], APMarkdownTableBlock) table];
    XCTAssertNotNil(table);
    XCTAssertEqualObjects(table.rows[0][0].string, @"one two", @"<br> becomes a separator");
    XCTAssertEqualObjects(table.rows[0][1].string, @"", @"Empty cells are padded");
    XCTAssertEqualObjects(table.rows[0][2].string, @"z");
    XCTAssertEqualObjects(APAttribute(table.rows[1][1], APSemanticStrikethroughAttributeName, @"y"), @YES);

    APMarkdownListItemBlock * item = APAs(blocks[1], APMarkdownListItemBlock);
    APMarkdownListItemBlock * continuation = APAs(blocks[2], APMarkdownListItemBlock);
    XCTAssertEqualObjects(item.marker, @"•");
    XCTAssertEqualObjects(continuation.marker, @"", @"Continuation paragraph in the same item");
    XCTAssertEqual(blocks.lastObject.quoteLevel, 2);

    XCTAssertEqual([APMarkdownDocument documentWithMarkdown:@""].blocks.count, 0u);
}

#pragma mark - Streaming

/// Documents whose blocks the stream must reproduce exactly however they are chunked:
/// blank lines inside fenced code, a fence inside a list item, loose and nested lists,
/// numbering that must survive a split, tables, quotes, and a trailing open fence.
- (NSArray<NSString *> *)streamingCorpus {
    return @[
        APSampleMarkdown,
        @"```python\ndef f():\n    pass\n\n\nprint(f())\n```\n\nafter",
        @"- outer\n\n  ```sh\n  echo a\n\n  echo b\n  ```\n\n- second\n\n  continued\n\nparagraph",
        @"1. one\n\n2. two\n\n   nested para\n\n3. three\n\n   - a\n   - b\n\n7. seven",
        @"> quoted\n> more\n\n> again\n\n> > deeper\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n# Heading\n\ntext",
        @"Setext\n=====\n\ntext\n\n~~~\ntilde fence with ``` inside\n\n~~~\n\n```\nunclosed to the end\n\nstill code",
        @"\n\n\nleading blanks\n\n\n\nand many between\n\n",
        @"intro\n\n* star\n\n+ plus\n\n3) paren\n\n10. ten\n\n1.not a list\n\n-not a list either\n\nend",
    ];
}

- (void)testStreamMatchesFullParseAtEveryChunkSize {
    for (NSString * source in [self streamingCorpus]) {
        NSString * expected = APDumpMarkdownBlocks([APMarkdownDocument documentWithMarkdown:source].blocks);
        for (NSUInteger chunk = 1; chunk <= 7; chunk += 2) {
            APMarkdownStream * stream = [[APMarkdownStream alloc] init];
            NSUInteger settled = 0;
            for (NSUInteger offset = 0; offset < source.length; offset += chunk) {
                NSRange piece = NSMakeRange(offset, MIN(chunk, source.length - offset));
                [stream appendString:[source substringWithRange:piece]];
                XCTAssertGreaterThanOrEqual(stream.settledLength, settled, @"Settled text never unsettles");
                XCTAssertLessThanOrEqual(stream.settledLength, stream.source.length);
                settled = stream.settledLength;
            }
            XCTAssertEqualObjects(stream.source, source);
            XCTAssertEqualObjects(APDumpMarkdownBlocks(stream.blocks), expected,
                                  @"chunk %lu of %@", (unsigned long)chunk, source);
            XCTAssertEqualObjects(APDumpMarkdownBlocks([stream document].blocks), expected);
        }
    }
}

- (void)testStreamSettlesAtTopLevelBoundariesOnly {
    APMarkdownStream * stream = [[APMarkdownStream alloc] init];
    [stream appendString:@"para\n\n"];
    XCTAssertEqual(stream.settledLength, 0u, @"A blank line alone settles nothing: the next line decides");
    [stream appendString:@"  indented continuation\n\n"];
    XCTAssertEqual(stream.settledLength, 0u, @"An indented line may continue the block before it");
    [stream appendString:@"```\ncode\n\n"];
    XCTAssertEqual(stream.settledLength, [stream.source rangeOfString:@"```"].location,
                   @"The fence opener starts a new top-level block");
    [stream appendString:@"more code\n```\n\n- item\n\n- item\n\nnext\n"];
    XCTAssertEqual(stream.settledLength, [stream.source rangeOfString:@"next"].location,
                   @"Nothing settles inside the fence or at a list item; the paragraph after does");
    XCTAssertEqual(stream.blocks.count, 6u);
    XCTAssertEqualObjects([APAs(stream.blocks[2], APMarkdownCodeBlock) snippet].code, @"code\n\nmore code");
    XCTAssertEqualObjects([APAs(stream.blocks[4], APMarkdownListItemBlock) marker], @"•");
}

@end
