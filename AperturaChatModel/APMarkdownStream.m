#import "APMarkdownStream.h"
#import "APMarkdownInternal.h"

@implementation APMarkdownStream {
    NSMutableString * _source;
    NSMutableArray<APMarkdownBlock *> * _settledBlocks;
    NSUInteger _settledLength;
    NSArray<APMarkdownBlock *> * _tailBlocks;
    NSArray<APMarkdownBlock *> * _blocks;   // settled + tail, built on demand
}

- (instancetype)init {
    if ((self = [super init])) {
        _source = [NSMutableString string];
        _settledBlocks = [NSMutableArray array];
        _tailBlocks = @[];
    }
    return self;
}

- (NSString *)source { return [_source copy]; }
- (NSUInteger)settledLength { return _settledLength; }

- (NSArray<APMarkdownBlock *> *)blocks {
    if (!_blocks) _blocks = [_settledBlocks arrayByAddingObjectsFromArray:_tailBlocks];
    return _blocks;
}

- (APMarkdownDocument *)document {
    return [[APMarkdownDocument alloc] initWithSource:_source blocks:self.blocks];
}

- (void)appendString:(NSString *)text {
    if (text.length == 0) return;
    [_source appendString:text];

    NSUInteger boundary = [self lastBoundary];
    if (boundary != NSNotFound) {
        NSString * chunk = [_source substringWithRange:NSMakeRange(_settledLength, boundary - _settledLength)];
        [_settledBlocks addObjectsFromArray:[APMarkdownDocument blocksByParsing:chunk]];
        _settledLength = boundary;
    }
    _tailBlocks = [APMarkdownDocument blocksByParsing:[_source substringFromIndex:_settledLength]];
    _blocks = nil;
}

#pragma mark - Boundaries

/// Whether `line` is only whitespace.
static BOOL APLineIsBlank(NSString * source, NSRange line) {
    for (NSUInteger i = line.location; i < NSMaxRange(line); i++) {
        unichar c = [source characterAtIndex:i];
        if (c != ' ' && c != '\t' && c != '\r') return NO;
    }
    return YES;
}

/// If `line` opens or closes a fenced code block, the fence character and its length;
/// otherwise 0. Leading whitespace of any depth counts (fences inside list items).
static NSUInteger APLineFence(NSString * source, NSRange line, unichar * fenceChar) {
    NSUInteger i = line.location, end = NSMaxRange(line);
    while (i < end && ([source characterAtIndex:i] == ' ' || [source characterAtIndex:i] == '\t')) i++;
    if (i >= end) return 0;
    unichar c = [source characterAtIndex:i];
    if (c != '`' && c != '~') return 0;
    NSUInteger count = 0;
    while (i < end && [source characterAtIndex:i] == c) { i++; count++; }
    if (count < 3) return 0;
    *fenceChar = c;
    return count;
}

/// Whether `line` starts a list item: a bullet or an ordered marker followed by a space.
static BOOL APLineStartsListItem(NSString * source, NSRange line) {
    NSUInteger i = line.location, end = NSMaxRange(line);
    if (i >= end) return NO;
    unichar c = [source characterAtIndex:i];
    if (c == '-' || c == '*' || c == '+') {
        i++;
    } else {
        NSUInteger digits = 0;
        while (i < end && digits < 9 && isdigit([source characterAtIndex:i])) { i++; digits++; }
        if (digits == 0 || i >= end) return NO;
        unichar after = [source characterAtIndex:i];
        if (after != '.' && after != ')') return NO;
        i++;
    }
    return i >= end || [source characterAtIndex:i] == ' ' || [source characterAtIndex:i] == '\t';
}

/// The start of the last COMPLETE line after `_settledLength` that begins a new top-level
/// block: preceded by a blank line, not indented, not a list item (a list spans blank lines,
/// and its numbering is counted from the first item), not inside a fenced code block. An
/// unterminated line is not judged — "7" may still become "7. seven". NSNotFound if none.
/// The settled prefix always ends at such a line, so the scan starts outside any fence.
- (NSUInteger)lastBoundary {
    NSUInteger length = _source.length;
    NSUInteger boundary = NSNotFound;
    unichar openFence = 0;
    NSUInteger openFenceLength = 0;
    BOOL previousBlank = NO;

    NSUInteger lineStart = _settledLength;
    while (lineStart < length) {
        NSUInteger lineEnd = [_source rangeOfString:@"\n" options:0
                                              range:NSMakeRange(lineStart, length - lineStart)].location;
        if (lineEnd == NSNotFound) break;   // the open line: not yet a block of its own
        NSRange line = NSMakeRange(lineStart, lineEnd - lineStart);

        unichar fenceChar = 0;
        NSUInteger fence = APLineFence(_source, line, &fenceChar);
        if (openFence) {
            if (fence >= openFenceLength && fenceChar == openFence) openFence = 0;
            previousBlank = NO;
        } else if (APLineIsBlank(_source, line)) {
            previousBlank = YES;
        } else {
            unichar first = [_source characterAtIndex:lineStart];
            if (previousBlank && first != ' ' && first != '\t' && !APLineStartsListItem(_source, line)) {
                boundary = lineStart;
            }
            previousBlank = NO;
            if (fence) { openFence = fenceChar; openFenceLength = fence; }
        }
        lineStart = lineEnd + 1;
    }
    return boundary;
}

@end
