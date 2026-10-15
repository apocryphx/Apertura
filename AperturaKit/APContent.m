#import "APContent.h"

#pragma mark - APAttachment

@implementation APAttachment

- (instancetype)initWithName:(NSString *)name text:(NSString *)text
                   sizeBytes:(unsigned long long)sizeBytes
       contentTypeIdentifier:(NSString *)contentTypeIdentifier {
    if ((self = [super init])) {
        _name = [name copy];
        _text = [text copy];
        _sizeBytes = sizeBytes;
        _contentTypeIdentifier = [contentTypeIdentifier copy];
    }
    return self;
}

+ (instancetype)attachmentWithName:(NSString *)name text:(NSString *)text
                         sizeBytes:(unsigned long long)sizeBytes
             contentTypeIdentifier:(NSString *)contentTypeIdentifier {
    return [[self alloc] initWithName:name text:text sizeBytes:sizeBytes
                contentTypeIdentifier:contentTypeIdentifier];
}

- (NSString *)promptText {
    return [NSString stringWithFormat:@"--- attached file: %@ ---\n%@\n--- end of %@ ---\n\n",
            _name, _text, _name];
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end

#pragma mark - APToolCall

@implementation APToolCall

- (instancetype)initWithName:(NSString *)name arguments:(NSDictionary<NSString *, id> *)arguments
                      status:(APToolCallStatus)status result:(NSString *)result
                    duration:(NSTimeInterval)duration {
    if ((self = [super init])) {
        _name = [name copy];
        _arguments = [arguments copy] ?: @{};
        _status = status;
        _result = [result copy];
        _duration = duration;
    }
    return self;
}

+ (instancetype)toolCallWithName:(NSString *)name arguments:(NSDictionary<NSString *, id> *)arguments
                          status:(APToolCallStatus)status result:(NSString *)result
                        duration:(NSTimeInterval)duration {
    return [[self alloc] initWithName:name arguments:arguments status:status
                               result:result duration:duration];
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end

#pragma mark - APContent

@implementation APContent

- (instancetype)initWithKind:(APContentKind)kind text:(NSString *)text
                    toolCall:(APToolCall *)toolCall attachment:(APAttachment *)attachment {
    if ((self = [super init])) {
        _kind = kind;
        _text = [text copy];
        _toolCall = toolCall;
        _attachment = attachment;
    }
    return self;
}

+ (instancetype)textContent:(NSString *)text {
    return [[self alloc] initWithKind:APContentKindText text:text toolCall:nil attachment:nil];
}

+ (instancetype)reasoningContent:(NSString *)text {
    return [[self alloc] initWithKind:APContentKindReasoning text:text toolCall:nil attachment:nil];
}

+ (instancetype)toolCallContent:(APToolCall *)toolCall {
    return [[self alloc] initWithKind:APContentKindToolCall text:nil toolCall:toolCall attachment:nil];
}

+ (instancetype)attachmentContent:(APAttachment *)attachment {
    return [[self alloc] initWithKind:APContentKindAttachment text:nil toolCall:nil attachment:attachment];
}

- (NSString *)promptText {
    switch (_kind) {
        case APContentKindText:       return _text;
        case APContentKindAttachment: return _attachment.promptText;
        case APContentKindReasoning:
        case APContentKindToolCall:
        case APContentKindImage:
        case APContentKindAudio:      return nil;
    }
    return nil;
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end
