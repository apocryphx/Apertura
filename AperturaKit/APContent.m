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

#pragma mark - APImage

@implementation APImage

- (instancetype)initWithIdentifier:(NSString *)identifier mediaType:(NSString *)mediaType
                         byteCount:(unsigned long long)byteCount pixelSize:(CGSize)pixelSize
                      originalName:(NSString *)originalName fileURL:(NSURL *)fileURL {
    if ((self = [super init])) {
        _identifier = [identifier copy];
        _mediaType = [mediaType copy];
        _byteCount = byteCount;
        _pixelSize = pixelSize;
        _originalName = [originalName copy];
        _fileURL = [fileURL copy];
    }
    return self;
}

+ (instancetype)imageWithIdentifier:(NSString *)identifier mediaType:(NSString *)mediaType
                          byteCount:(unsigned long long)byteCount pixelSize:(CGSize)pixelSize
                       originalName:(NSString *)originalName fileURL:(NSURL *)fileURL {
    return [[self alloc] initWithIdentifier:identifier mediaType:mediaType byteCount:byteCount
                                  pixelSize:pixelSize originalName:originalName fileURL:fileURL];
}

- (instancetype)imageWithFileURL:(NSURL *)fileURL {
    return [[self.class alloc] initWithIdentifier:_identifier mediaType:_mediaType byteCount:_byteCount
                                        pixelSize:_pixelSize originalName:_originalName fileURL:fileURL];
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
                    toolCall:(APToolCall *)toolCall attachment:(APAttachment *)attachment
                       image:(APImage *)image {
    if ((self = [super init])) {
        _kind = kind;
        _text = [text copy];
        _toolCall = toolCall;
        _attachment = attachment;
        _image = image;
    }
    return self;
}

+ (instancetype)textContent:(NSString *)text {
    return [[self alloc] initWithKind:APContentKindText text:text toolCall:nil attachment:nil image:nil];
}

+ (instancetype)reasoningContent:(NSString *)text {
    return [[self alloc] initWithKind:APContentKindReasoning text:text toolCall:nil attachment:nil image:nil];
}

+ (instancetype)toolCallContent:(APToolCall *)toolCall {
    return [[self alloc] initWithKind:APContentKindToolCall text:nil toolCall:toolCall attachment:nil image:nil];
}

+ (instancetype)attachmentContent:(APAttachment *)attachment {
    return [[self alloc] initWithKind:APContentKindAttachment text:nil toolCall:nil attachment:attachment image:nil];
}

+ (instancetype)imageContent:(APImage *)image {
    return [[self alloc] initWithKind:APContentKindImage text:nil toolCall:nil attachment:nil image:image];
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
