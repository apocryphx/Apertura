#import "APDisplayMessage.h"
#import "APMarkdownDocument.h"

@implementation APDisplayMessage

- (instancetype)initWithMessage:(APMessage *)message {
    if ((self = [super init])) {
        _message = message;
        _blocks = [self.class blocksForContent:message.content];
    }
    return self;
}

+ (instancetype)displayMessageWithMessage:(APMessage *)message {
    return [[self alloc] initWithMessage:message];
}

+ (NSArray<APChatBlock *> *)blocksForContent:(NSArray<APContent *> *)content {
    NSMutableArray<APChatBlock *> * blocks = [NSMutableArray arrayWithCapacity:content.count];
    for (APContent * part in content) {
        APChatBlock * block = [self blockForContent:part];
        if (block) [blocks addObject:block];
    }
    return [blocks copy];
}

+ (nullable APChatBlock *)blockForContent:(APContent *)part {
    switch (part.kind) {
        case APContentKindText:
            return [[APTextBlock alloc] initWithDocument:[APMarkdownDocument documentWithMarkdown:part.text ?: @""]];
        case APContentKindReasoning:
            return [[APReasoningBlock alloc] initWithText:part.text ?: @""];
        case APContentKindToolCall:
            return part.toolCall ? [[APToolCallBlock alloc] initWithToolCall:part.toolCall] : nil;
        case APContentKindAttachment:
            return part.attachment ? [[APAttachmentBlock alloc] initWithAttachment:part.attachment] : nil;
        case APContentKindImage:
        case APContentKindAudio:
            return nil;   // reserved: nothing to show until the kit can construct them
    }
    return nil;
}

@end
