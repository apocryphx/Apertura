#import "APChatBlock.h"
#import "APMarkdownDocument.h"
#import "APCodeHighlighter.h"
#import "APReadableJSON.h"

@implementation APChatBlock
@end

@implementation APTextBlock

- (instancetype)initWithDocument:(APMarkdownDocument *)document {
    if ((self = [super init])) _document = document;
    return self;
}

@end

@implementation APReasoningBlock

- (instancetype)initWithText:(NSString *)text {
    if ((self = [super init])) _text = [text copy];
    return self;
}

@end

@implementation APToolCallBlock

+ (NSUInteger)resultPreviewLimit { return 4000; }

- (instancetype)initWithToolCall:(APToolCall *)toolCall {
    if ((self = [super init])) {
        _toolCall = toolCall;
        _argumentsSnippet = [APCodeSnippet snippetWithLanguage:@"json"
                                                   highlighted:[APReadableJSON formatValue:toolCall.arguments]];
        if (toolCall.result) [self buildPreviewOfResult:toolCall.result];
    }
    return self;
}

- (void)buildPreviewOfResult:(NSString *)result {
    NSUInteger limit = APToolCallBlock.resultPreviewLimit;
    NSAttributedString * readable = [APReadableJSON formatText:result];
    if (readable) {
        _resultTruncated = readable.length > limit;
        if (_resultTruncated) {
            NSRange head = [readable.string rangeOfComposedCharacterSequencesForRange:NSMakeRange(0, limit)];
            readable = [readable attributedSubstringFromRange:head];
        }
        _resultPreview = [APCodeSnippet snippetWithLanguage:@"json" highlighted:readable];
        return;
    }

    _resultTruncated = result.length > limit;
    NSString * preview = _resultTruncated
        ? [result substringWithRange:[result rangeOfComposedCharacterSequencesForRange:NSMakeRange(0, limit)]]
        : result;
    NSString * start = [preview stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    BOOL looksLikeJSON = [start hasPrefix:@"{"] || [start hasPrefix:@"["];
    _resultPreview = [APCodeSnippet snippetWithLanguage:looksLikeJSON ? @"json" : nil code:preview];
}

@end

@implementation APAttachmentBlock

- (instancetype)initWithAttachment:(APAttachment *)attachment {
    if ((self = [super init])) _attachment = attachment;
    return self;
}

@end
