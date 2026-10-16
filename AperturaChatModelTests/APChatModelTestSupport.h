//  APChatModelTestSupport — helpers shared by the AperturaChatModel tests: token and
//  attribute inspection, typed block access, and a structural dump of blocks so two block
//  arrays (live vs. archived) can be compared as strings.
#import <XCTest/XCTest.h>
#import <AperturaKit/AperturaKit.h>
#import <AperturaChatModel/AperturaChatModel.h>

NS_ASSUME_NONNULL_BEGIN

/// The token kinds in `highlighted`, as "text=kind" strings in order.
static inline NSArray<NSString *> * APTokens(NSAttributedString * highlighted) {
    NSMutableArray * tokens = [NSMutableArray array];
    [highlighted enumerateAttribute:APCodeTokenAttributeName inRange:NSMakeRange(0, highlighted.length) options:0
                         usingBlock:^(NSNumber * kind, NSRange range, BOOL * stop) {
        if (kind) [tokens addObject:[NSString stringWithFormat:@"%@=%ld",
                                     [highlighted.string substringWithRange:range], (long)kind.integerValue]];
    }];
    return tokens;
}

static inline NSString * APToken(NSString * text, APCodeTokenKind kind) {
    return [NSString stringWithFormat:@"%@=%ld", text, (long)kind];
}

/// `object` when it is a `cls`, else nil, so a wrong block type fails an assertion instead of crashing.
static inline id _Nullable APAsKindOf(id _Nullable object, Class cls) {
    return [object isKindOfClass:cls] ? object : nil;
}

/// `object` typed as `Type *` when it is one, else nil.
#define APAs(object, Type) ((Type *)APAsKindOf(object, Type.class))

/// The value of `key` over the first occurrence of `substring`, if it covers all of it.
static inline id _Nullable APAttribute(NSAttributedString * text, NSAttributedStringKey key, NSString * substring) {
    NSRange range = [text.string rangeOfString:substring];
    if (range.location == NSNotFound) return nil;
    NSRange effective;
    id value = [text attribute:key atIndex:range.location longestEffectiveRange:&effective inRange:range];
    return NSEqualRanges(effective, range) ? value : nil;
}

/// A line per block: class, quote level, and the block's own fields. Attributed text is
/// dumped with its attributes, so inline semantics take part in the comparison.
static inline NSString * APDumpMarkdownBlocks(NSArray<APMarkdownBlock *> * blocks) {
    NSMutableString * dump = [NSMutableString string];
    for (APMarkdownBlock * block in blocks) {
        [dump appendFormat:@"%@ q%ld", NSStringFromClass(block.class), (long)block.quoteLevel];
        if ([block isKindOfClass:APMarkdownHeadingBlock.class]) {
            [dump appendFormat:@" h%ld", (long)((APMarkdownHeadingBlock *)block).level];
        } else if ([block isKindOfClass:APMarkdownListItemBlock.class]) {
            APMarkdownListItemBlock * item = (APMarkdownListItemBlock *)block;
            [dump appendFormat:@" d%ld '%@'", (long)item.depth, item.marker];
        }
        if ([block isKindOfClass:APMarkdownFlowBlock.class]) {
            [dump appendFormat:@" %@", ((APMarkdownFlowBlock *)block).text];
        } else if ([block isKindOfClass:APMarkdownCodeBlock.class]) {
            APCodeSnippet * snippet = ((APMarkdownCodeBlock *)block).snippet;
            [dump appendFormat:@" [%@] %@ %@", snippet.language, snippet.code, APTokens(snippet.highlighted)];
        } else if ([block isKindOfClass:APMarkdownTableBlock.class]) {
            APMarkdownTable * table = ((APMarkdownTableBlock *)block).table;
            [dump appendFormat:@" %ld %@ %@", (long)table.columnCount, table.header, table.rows];
        }
        [dump appendString:@"\n"];
    }
    return dump;
}

static inline NSString * APDumpChatBlocks(NSArray<APChatBlock *> * blocks) {
    NSMutableString * dump = [NSMutableString string];
    for (APChatBlock * block in blocks) {
        [dump appendFormat:@"== %@\n", NSStringFromClass(block.class)];
        if ([block isKindOfClass:APTextBlock.class]) {
            APMarkdownDocument * document = ((APTextBlock *)block).document;
            [dump appendFormat:@"%@\n%@", document.source, APDumpMarkdownBlocks(document.blocks)];
        } else if ([block isKindOfClass:APReasoningBlock.class]) {
            [dump appendFormat:@"%@\n", ((APReasoningBlock *)block).text];
        } else if ([block isKindOfClass:APToolCallBlock.class]) {
            APToolCallBlock * call = (APToolCallBlock *)block;
            [dump appendFormat:@"%@ %ld %@ | %@ | %d\n", call.toolCall.name, (long)call.toolCall.status,
             call.argumentsSnippet.code, call.resultPreview.code, call.resultTruncated];
        } else if ([block isKindOfClass:APAttachmentBlock.class]) {
            APAttachment * attachment = ((APAttachmentBlock *)block).attachment;
            [dump appendFormat:@"%@ %llu\n", attachment.name, attachment.sizeBytes];
        }
    }
    return dump;
}

/// The prototype's sample answer: every block kind and inline style in one document.
static NSString * const APSampleMarkdown =
    @"## Pour-Over in Five Steps\n\n"
    @"A clean cup comes down to **ratio**, **grind**, and *patience*.\n\n"
    @"| Setting | Value | Why |\n|---|---|---|\n"
    @"| Ratio | 1:16 | Balanced strength |\n| Grind | Medium-fine | Controls flow rate |\n"
    @"| Water | 94 °C | Extracts without bitterness |\n\n"
    @"1. Rinse the filter with hot water.\n2. Add `20 g` of coffee.\n3. Pour in stages:\n"
    @"   - Bloom with 40 g for 30 s\n   - Pour to 320 g in slow circles\n4. Let it drain.\n5. Enjoy.\n\n"
    @"> Tip: If it tastes sour, grind finer.\n\n---\n\n### Timer Script\n\n"
    @"```swift\nfor stage in [\"bloom\", \"pour\", \"drain\"] {\n    print(\"Now: \\(stage)\")\n}\n```\n\n"
    @"See [the SCA guide](https://sca.coffee) for more.";

NS_ASSUME_NONNULL_END
