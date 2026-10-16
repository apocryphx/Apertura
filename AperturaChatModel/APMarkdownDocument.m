#import "APMarkdownInternal.h"
#import "APCodeHighlighter.h"
#import "APSemanticText.h"

#pragma mark - APMarkdownTable

@implementation APMarkdownTable

- (instancetype)initWithHeader:(NSArray<NSAttributedString *> *)header
                          rows:(NSArray<NSArray<NSAttributedString *> *> *)rows {
    if ((self = [super init])) {
        // Deep copies: the builder hands in mutable cells.
        _header = [[NSArray alloc] initWithArray:header copyItems:YES];
        NSMutableArray * copiedRows = [NSMutableArray arrayWithCapacity:rows.count];
        for (NSArray * row in rows) [copiedRows addObject:[[NSArray alloc] initWithArray:row copyItems:YES]];
        _rows = [copiedRows copy];
        NSInteger columns = (NSInteger)header.count;
        for (NSArray * row in rows) columns = MAX(columns, (NSInteger)row.count);
        _columnCount = columns;
    }
    return self;
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end

#pragma mark - Blocks

@implementation APMarkdownBlock

- (instancetype)initWithQuoteLevel:(NSInteger)quoteLevel {
    if ((self = [super init])) _quoteLevel = quoteLevel;
    return self;
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end

@implementation APMarkdownFlowBlock

- (instancetype)initWithText:(NSAttributedString *)text quoteLevel:(NSInteger)quoteLevel {
    if ((self = [super initWithQuoteLevel:quoteLevel])) _text = [text copy];
    return self;
}

@end

@implementation APMarkdownHeadingBlock

- (instancetype)initWithLevel:(NSInteger)level text:(NSAttributedString *)text quoteLevel:(NSInteger)quoteLevel {
    if ((self = [super initWithText:text quoteLevel:quoteLevel])) _level = level;
    return self;
}

@end

@implementation APMarkdownParagraphBlock
@end

@implementation APMarkdownListItemBlock

- (instancetype)initWithDepth:(NSInteger)depth marker:(NSString *)marker text:(NSAttributedString *)text
                   quoteLevel:(NSInteger)quoteLevel {
    if ((self = [super initWithText:text quoteLevel:quoteLevel])) {
        _depth = depth;
        _marker = [marker copy];
    }
    return self;
}

@end

@implementation APMarkdownCodeBlock

- (instancetype)initWithSnippet:(APCodeSnippet *)snippet quoteLevel:(NSInteger)quoteLevel {
    if ((self = [super initWithQuoteLevel:quoteLevel])) _snippet = snippet;
    return self;
}

@end

@implementation APMarkdownTableBlock

- (instancetype)initWithTable:(APMarkdownTable *)table quoteLevel:(NSInteger)quoteLevel {
    if ((self = [super initWithQuoteLevel:quoteLevel])) _table = table;
    return self;
}

@end

@implementation APMarkdownRuleBlock
@end

#pragma mark - Block builder

static BOOL APIntentIsList(NSPresentationIntent * intent) {
    return intent.intentKind == NSPresentationIntentKindOrderedList
        || intent.intentKind == NSPresentationIntentKindUnorderedList;
}

static BOOL APIntentIsTableRow(NSPresentationIntent * intent) {
    return intent.intentKind == NSPresentationIntentKindTableHeaderRow
        || intent.intentKind == NSPresentationIntentKindTableRow;
}

/// The first component (innermost first) of the given kind, or nil.
static NSPresentationIntent * _Nullable APFirstIntent(NSArray<NSPresentationIntent *> * components,
                                                      NSPresentationIntentKind kind) {
    for (NSPresentationIntent * component in components) {
        if (component.intentKind == kind) return component;
    }
    return nil;
}

/// Walks presentation-intent runs in document order and gathers them into blocks; table
/// cells accumulate until a run outside the table arrives.
@interface APMarkdownBlockBuilder : NSObject
- (void)addText:(NSAttributedString *)text intent:(nullable NSPresentationIntent *)intent;
- (NSArray<APMarkdownBlock *> *)finish;
@end

@implementation APMarkdownBlockBuilder {
    NSMutableArray<APMarkdownBlock *> * _blocks;

    // The table being gathered, cell by cell.
    NSNumber * _tableIdentity;
    NSInteger _tableQuoteLevel;
    NSMutableArray<NSMutableAttributedString *> * _tableHeader;
    NSMutableArray<NSMutableArray<NSMutableAttributedString *> *> * _tableRows;
    NSNumber * _tableRowIdentity;

    NSNumber * _lastListItemIdentity;
}

- (instancetype)init {
    if ((self = [super init])) _blocks = [NSMutableArray array];
    return self;
}

- (void)addText:(NSAttributedString *)text intent:(NSPresentationIntent *)intent {
    // Components run from innermost to outermost.
    NSMutableArray<NSPresentationIntent *> * components = [NSMutableArray array];
    NSInteger quoteLevel = 0;
    for (NSPresentationIntent * component = intent; component; component = component.parentIntent) {
        [components addObject:component];
        if (component.intentKind == NSPresentationIntentKindBlockQuote) quoteLevel++;
    }

    NSPresentationIntent * table = APFirstIntent(components, NSPresentationIntentKindTable);
    if (table) {
        [self addTableCell:text components:components tableIdentity:table.identity quoteLevel:quoteLevel];
        return;
    }
    [self flushTable];

    NSPresentationIntent * first = components.firstObject;
    switch (first.intentKind) {
        case NSPresentationIntentKindCodeBlock: {
            NSString * code = text.string;
            while ([code hasSuffix:@"\n"]) code = [code substringToIndex:code.length - 1];
            APCodeSnippet * snippet = [APCodeSnippet snippetWithLanguage:first.languageHint code:code];
            [_blocks addObject:[[APMarkdownCodeBlock alloc] initWithSnippet:snippet quoteLevel:quoteLevel]];
            return;
        }
        case NSPresentationIntentKindHeader:
            [_blocks addObject:[[APMarkdownHeadingBlock alloc] initWithLevel:first.headerLevel text:text
                                                                  quoteLevel:quoteLevel]];
            return;
        case NSPresentationIntentKindThematicBreak:
            [_blocks addObject:[[APMarkdownRuleBlock alloc] initWithQuoteLevel:quoteLevel]];
            return;
        default:
            break;
    }

    NSUInteger itemIndex = [components indexOfObjectPassingTest:^BOOL(NSPresentationIntent * component,
                                                                      NSUInteger idx, BOOL * stop) {
        return component.intentKind == NSPresentationIntentKindListItem;
    }];
    if (itemIndex == NSNotFound) {
        [_blocks addObject:[[APMarkdownParagraphBlock alloc] initWithText:text quoteLevel:quoteLevel]];
        return;
    }

    NSPresentationIntent * item = components[itemIndex];
    NSInteger depth = 0;
    for (NSPresentationIntent * component in components) if (APIntentIsList(component)) depth++;
    BOOL isOrdered = itemIndex + 1 < components.count
        && components[itemIndex + 1].intentKind == NSPresentationIntentKindOrderedList;
    NSString * marker = [_lastListItemIdentity isEqual:@(item.identity)]
        ? @""
        : (isOrdered ? [NSString stringWithFormat:@"%ld.", (long)item.ordinal] : @"•");
    _lastListItemIdentity = @(item.identity);
    [_blocks addObject:[[APMarkdownListItemBlock alloc] initWithDepth:depth marker:marker text:text
                                                           quoteLevel:quoteLevel]];
}

- (NSArray<APMarkdownBlock *> *)finish {
    [self flushTable];
    return [_blocks copy];
}

- (void)addTableCell:(NSAttributedString *)text
          components:(NSArray<NSPresentationIntent *> *)components
       tableIdentity:(NSInteger)tableIdentity
          quoteLevel:(NSInteger)quoteLevel {
    if (![_tableIdentity isEqual:@(tableIdentity)]) {
        [self flushTable];
        _tableIdentity = @(tableIdentity);
        _tableQuoteLevel = quoteLevel;
        _tableHeader = [NSMutableArray array];
        _tableRows = [NSMutableArray array];
        _tableRowIdentity = nil;
    }
    NSPresentationIntent * cell = APFirstIntent(components, NSPresentationIntentKindTableCell);
    NSInteger column = cell ? cell.column : 0;
    NSUInteger rowIndex = [components indexOfObjectPassingTest:^BOOL(NSPresentationIntent * component,
                                                                     NSUInteger idx, BOOL * stop) {
        return APIntentIsTableRow(component);
    }];
    if (rowIndex == NSNotFound) return;
    NSPresentationIntent * row = components[rowIndex];

    if (row.intentKind == NSPresentationIntentKindTableHeaderRow) {
        [self.class setCell:_tableHeader column:column text:text];
    } else {
        if (![_tableRowIdentity isEqual:@(row.identity)]) {
            [_tableRows addObject:[NSMutableArray array]];
            _tableRowIdentity = @(row.identity);
        }
        [self.class setCell:_tableRows.lastObject column:column text:text];
    }
}

- (void)flushTable {
    if (!_tableIdentity) return;
    APMarkdownTable * table = [[APMarkdownTable alloc] initWithHeader:_tableHeader rows:_tableRows];
    [_blocks addObject:[[APMarkdownTableBlock alloc] initWithTable:table quoteLevel:_tableQuoteLevel]];
    _tableIdentity = nil;
    _tableHeader = nil;
    _tableRows = nil;
    _tableRowIdentity = nil;
}

/// Empty cells produce no runs, so place cells by column index and pad the gaps.
+ (void)setCell:(NSMutableArray<NSMutableAttributedString *> *)cells column:(NSInteger)column
           text:(NSAttributedString *)text {
    while ((NSInteger)cells.count <= column) [cells addObject:[[NSMutableAttributedString alloc] init]];
    [cells[column] appendAttributedString:text];
}

@end

#pragma mark - APMarkdownDocument

@implementation APMarkdownDocument

- (instancetype)initWithSource:(NSString *)source blocks:(NSArray<APMarkdownBlock *> *)blocks {
    if ((self = [super init])) {
        _source = [source copy];
        _blocks = [blocks copy];
    }
    return self;
}

+ (instancetype)documentWithMarkdown:(NSString *)source {
    return [[self alloc] initWithSource:source blocks:[self blocksByParsing:source]];
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

+ (NSArray<APMarkdownBlock *> *)blocksByParsing:(NSString *)source {
    if (source.length == 0) return @[];

    // Models often use <br> for line breaks inside table cells.
    static NSRegularExpression * lineBreakTag;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        lineBreakTag = [NSRegularExpression regularExpressionWithPattern:@"<br\\s*/?>" options:0 error:NULL];
    });
    NSString * prepared = [lineBreakTag stringByReplacingMatchesInString:source options:0
                                                                   range:NSMakeRange(0, source.length)
                                                            withTemplate:@" "];

    NSAttributedStringMarkdownParsingOptions * options = [[NSAttributedStringMarkdownParsingOptions alloc] init];
    options.interpretedSyntax = NSAttributedStringMarkdownInterpretedSyntaxFull;
    options.failurePolicy = NSAttributedStringMarkdownParsingFailureReturnPartiallyParsedIfPossible;
    NSAttributedString * parsed = [[NSAttributedString alloc] initWithMarkdownString:prepared options:options
                                                                             baseURL:nil error:NULL];
    if (!parsed) {
        return @[ [[APMarkdownParagraphBlock alloc] initWithText:[[NSAttributedString alloc] initWithString:source]
                                                      quoteLevel:0] ];
    }

    APMarkdownBlockBuilder * builder = [[APMarkdownBlockBuilder alloc] init];
    [parsed enumerateAttribute:NSPresentationIntentAttributeName
                       inRange:NSMakeRange(0, parsed.length)
                       options:0
                    usingBlock:^(NSPresentationIntent * intent, NSRange range, BOOL * stop) {
        [builder addText:[self semanticTextFrom:parsed range:range] intent:intent];
    }];
    return [builder finish];
}

/// The key the Markdown parser stores link destinations under. It is NSLinkAttributeName,
/// which AppKit and UIKit declare but Foundation does not; the string is the same on both.
static NSAttributedStringKey const APParsedLinkAttributeName = @"NSLink";

/// Copies the run's characters, mapping inline intents and links to semantic keys.
+ (NSAttributedString *)semanticTextFrom:(NSAttributedString *)parsed range:(NSRange)range {
    NSMutableAttributedString * text =
        [[NSMutableAttributedString alloc] initWithString:[parsed.string substringWithRange:range]];
    [parsed enumerateAttributesInRange:range options:0
                            usingBlock:^(NSDictionary<NSAttributedStringKey, id> * attributes, NSRange run, BOOL * stop) {
        NSRange local = NSMakeRange(run.location - range.location, run.length);

        NSUInteger inlineIntent = [attributes[NSInlinePresentationIntentAttributeName] unsignedIntegerValue];
        if (inlineIntent & NSInlinePresentationIntentStronglyEmphasized) {
            [text addAttribute:APSemanticFontWeightAttributeName value:@(APFontWeightBold) range:local];
        }
        if (inlineIntent & NSInlinePresentationIntentEmphasized) {
            [text addAttribute:APSemanticItalicAttributeName value:@YES range:local];
        }
        if (inlineIntent & NSInlinePresentationIntentCode) {
            [text addAttribute:APSemanticCodeAttributeName value:@YES range:local];
        }
        if (inlineIntent & NSInlinePresentationIntentStrikethrough) {
            [text addAttribute:APSemanticStrikethroughAttributeName value:@YES range:local];
        }

        id link = attributes[APParsedLinkAttributeName];
        NSString * urlString = [link isKindOfClass:NSURL.class] ? [link absoluteString]
                             : [link isKindOfClass:NSString.class] ? link : nil;
        if (urlString) [text addAttribute:APSemanticLinkAttributeName value:urlString range:local];
    }];
    return text;
}

@end
