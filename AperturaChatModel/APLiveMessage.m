#import "APLiveMessage.h"
#import "APMarkdownStream.h"

@implementation APLiveMessage {
    NSMutableArray<APContent *> * _closedParts;
    NSMutableArray<APChatBlock *> * _closedBlocks;

    // The trailing text or reasoning part, still growing. Exactly one of the two is set.
    APMarkdownStream * _openText;
    NSMutableString * _openReasoning;

    NSArray<APChatBlock *> * _blocks;   // closed + open, built on demand
}

- (instancetype)init {
    if ((self = [super init])) {
        _closedParts = [NSMutableArray array];
        _closedBlocks = [NSMutableArray array];
    }
    return self;
}

#pragma mark - Deltas

- (void)appendDelta:(APResponseDelta *)delta {
    switch (delta.kind) {
        case APResponseDeltaKindToolCall:
            [self closeOpenPart];
            if (delta.toolCall) {
                [_closedParts addObject:[APContent toolCallContent:delta.toolCall]];
                [_closedBlocks addObject:[[APToolCallBlock alloc] initWithToolCall:delta.toolCall]];
            }
            break;
        case APResponseDeltaKindText:
            if (_openReasoning) [self closeOpenPart];
            if (!_openText) _openText = [[APMarkdownStream alloc] init];
            [_openText appendString:delta.text];
            break;
        case APResponseDeltaKindReasoning:
            if (_openText) [self closeOpenPart];
            if (!_openReasoning) _openReasoning = [NSMutableString string];
            [_openReasoning appendString:delta.text];
            break;
    }
    _blocks = nil;
}

- (void)closeOpenPart {
    if (_openText) {
        [_closedParts addObject:[APContent textContent:_openText.source]];
        [_closedBlocks addObject:[[APTextBlock alloc] initWithDocument:[_openText document]]];
        _openText = nil;
    } else if (_openReasoning) {
        [_closedParts addObject:[APContent reasoningContent:_openReasoning]];
        [_closedBlocks addObject:[[APReasoningBlock alloc] initWithText:_openReasoning]];
        _openReasoning = nil;
    }
}

#pragma mark - State

- (NSArray<APChatBlock *> *)blocks {
    if (!_blocks) {
        APChatBlock * open = _openText ? [[APTextBlock alloc] initWithDocument:[_openText document]]
                           : _openReasoning ? [[APReasoningBlock alloc] initWithText:_openReasoning]
                           : nil;
        _blocks = open ? [_closedBlocks arrayByAddingObject:open] : [_closedBlocks copy];
    }
    return _blocks;
}

- (NSArray<APContent *> *)content {
    APContent * open = _openText ? [APContent textContent:_openText.source]
                     : _openReasoning ? [APContent reasoningContent:_openReasoning]
                     : nil;
    return open ? [_closedParts arrayByAddingObject:open] : [_closedParts copy];
}

- (APMessage *)messageWithStats:(APResponseStats *)stats modelIdentifier:(NSString *)modelIdentifier {
    return [APMessage messageWithRole:APRoleAssistant content:self.content
                                stats:stats modelIdentifier:modelIdentifier];
}

@end
