//  APMarkdownStream — Markdown that is still arriving.
//
//  A model streams its answer a few characters at a time, and a view wants blocks after
//  every delta. Re-parsing the whole text each time costs O(length) per delta; instead the
//  stream SETTLES the text up to the last block boundary it can trust — a blank line
//  followed by an unindented line that is not a list item, outside any fenced code block —
//  parses that prefix once, and re-parses only the open tail (the paragraph, list, table or
//  code block still being written) on each append. Settled blocks never change.
//
//  The split is exact for CommonMark constructs that cannot span such a line: paragraphs,
//  headings, tables, thematic breaks, quotes, and lists (a list spans blank lines and
//  numbers its items from the first, so a list is never split). Two constructs can span it:
//  a link reference definition used before it is defined, and an HTML block type that spans
//  blank lines. Both are rare in model output and degrade to literal text; an archived
//  message (APMarkdownDocument) parses them in full.
#import <Foundation/Foundation.h>
#import <AperturaChatModel/APMarkdownDocument.h>

NS_ASSUME_NONNULL_BEGIN

/// Not thread-safe: append and read from one queue (the session's callback queue).
@interface APMarkdownStream : NSObject

- (instancetype)init NS_DESIGNATED_INITIALIZER;

/// Appends `text` and re-parses the open tail. Cost is proportional to the tail, not the
/// whole source.
- (void)appendString:(NSString *)text;

/// Everything appended so far.
@property (readonly) NSString *source;
/// The blocks of `source` as of the last append: the settled prefix's, then the tail's.
@property (readonly) NSArray<APMarkdownBlock *> *blocks;
/// How many characters of `source` are settled — their blocks will not change again.
@property (readonly) NSUInteger settledLength;

/// An immutable snapshot (no parsing: the blocks are the ones already built).
- (APMarkdownDocument *)document;

@end

NS_ASSUME_NONNULL_END
