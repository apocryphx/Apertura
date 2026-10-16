//  APMarkdownDocument — Markdown split into renderable blocks.
//
//  Parsed with Foundation's CommonMark parser (-[NSAttributedString initWithMarkdownString:
//  options:baseURL:error:]); the NSPresentationIntent runs it produces become one block
//  each — heading, paragraph, list item, code block, table, thematic break — nested in
//  however many block quotes. Inline intent (bold, italic, code, strikethrough, links)
//  stays in each block's text as SEMANTIC attributes (APSemanticText.h): no font, no color,
//  so a view resolves appearance at display time and a theme change needs no re-parse.
//  Code blocks are tokenized here, once (APCodeHighlighter).
#import <Foundation/Foundation.h>

@class APCodeSnippet;

NS_ASSUME_NONNULL_BEGIN

#pragma mark - Table

@interface APMarkdownTable : NSObject <NSCopying>

/// Semantic text per cell. Empty when the table has no header row.
@property (readonly) NSArray<NSAttributedString *> *header;
@property (readonly) NSArray<NSArray<NSAttributedString *> *> *rows;
/// The widest row; shorter rows are missing trailing cells.
@property (readonly) NSInteger columnCount;

- (instancetype)initWithHeader:(NSArray<NSAttributedString *> *)header
                          rows:(NSArray<NSArray<NSAttributedString *> *> *)rows NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@end

#pragma mark - Blocks

/// One renderable block. Abstract: every block is one of the subclasses below.
@interface APMarkdownBlock : NSObject <NSCopying>
/// How many block quotes the block is nested in; 0 outside any quote.
@property (readonly) NSInteger quoteLevel;
- (instancetype)initWithQuoteLevel:(NSInteger)quoteLevel NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// A block of inline text. Abstract: consecutive flow blocks at one quote level can be
/// shown as one flowing text, so a selection may span them.
@interface APMarkdownFlowBlock : APMarkdownBlock
/// Semantic inline text (APSemanticText.h).
@property (readonly) NSAttributedString *text;
- (instancetype)initWithText:(NSAttributedString *)text quoteLevel:(NSInteger)quoteLevel NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithQuoteLevel:(NSInteger)quoteLevel NS_UNAVAILABLE;
@end

@interface APMarkdownHeadingBlock : APMarkdownFlowBlock
/// 1 for `#`, 2 for `##`, …
@property (readonly) NSInteger level;
- (instancetype)initWithLevel:(NSInteger)level text:(NSAttributedString *)text
                   quoteLevel:(NSInteger)quoteLevel NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithText:(NSAttributedString *)text quoteLevel:(NSInteger)quoteLevel NS_UNAVAILABLE;
@end

@interface APMarkdownParagraphBlock : APMarkdownFlowBlock
@end

@interface APMarkdownListItemBlock : APMarkdownFlowBlock
/// 1 for a top-level item.
@property (readonly) NSInteger depth;
/// "•" or "3."; empty for a continuation paragraph inside the same item.
@property (readonly) NSString *marker;
- (instancetype)initWithDepth:(NSInteger)depth marker:(NSString *)marker text:(NSAttributedString *)text
                   quoteLevel:(NSInteger)quoteLevel NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithText:(NSAttributedString *)text quoteLevel:(NSInteger)quoteLevel NS_UNAVAILABLE;
@end

@interface APMarkdownCodeBlock : APMarkdownBlock
@property (readonly) APCodeSnippet *snippet;
- (instancetype)initWithSnippet:(APCodeSnippet *)snippet quoteLevel:(NSInteger)quoteLevel NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithQuoteLevel:(NSInteger)quoteLevel NS_UNAVAILABLE;
@end

@interface APMarkdownTableBlock : APMarkdownBlock
@property (readonly) APMarkdownTable *table;
- (instancetype)initWithTable:(APMarkdownTable *)table quoteLevel:(NSInteger)quoteLevel NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithQuoteLevel:(NSInteger)quoteLevel NS_UNAVAILABLE;
@end

/// A thematic break (`---`).
@interface APMarkdownRuleBlock : APMarkdownBlock
@end

#pragma mark - Document

@interface APMarkdownDocument : NSObject <NSCopying>

/// Parses `source` in full. For text that is still arriving, see APMarkdownStream.
+ (instancetype)documentWithMarkdown:(NSString *)source;

/// The Markdown text — what gets stored; `blocks` is derived from it.
@property (readonly) NSString *source;
@property (readonly) NSArray<APMarkdownBlock *> *blocks;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
