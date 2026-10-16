#import "APMessage.h"
#import "APResponse.h"

static NSString * const kRoleSystem    = @"system";
static NSString * const kRoleUser      = @"user";
static NSString * const kRoleAssistant = @"assistant";
static NSString * const kRoleTool      = @"tool";

NSString * APRoleName(APRole role) {
    switch (role) {
        case APRoleSystem:    return kRoleSystem;
        case APRoleUser:      return kRoleUser;
        case APRoleAssistant: return kRoleAssistant;
        case APRoleTool:      return kRoleTool;
    }
    return nil;   // a case added without updating this switch: refuse to guess
}

BOOL APRoleFromName(NSString * name, APRole * outRole) {
    if      ([name isEqualToString:kRoleSystem])    { *outRole = APRoleSystem;    return YES; }
    else if ([name isEqualToString:kRoleUser])      { *outRole = APRoleUser;      return YES; }
    else if ([name isEqualToString:kRoleAssistant]) { *outRole = APRoleAssistant; return YES; }
    else if ([name isEqualToString:kRoleTool])      { *outRole = APRoleTool;      return YES; }
    return NO;
}

@implementation APMessage

- (instancetype)initWithRole:(APRole)role content:(NSArray<APContent *> *)content
                       stats:(APResponseStats *)stats modelIdentifier:(NSString *)modelIdentifier {
    if ((self = [super init])) {
        _role = role;
        _content = [content copy];
        _stats = stats;
        _modelIdentifier = [modelIdentifier copy];
    }
    return self;
}

+ (instancetype)messageWithRole:(APRole)role content:(NSArray<APContent *> *)content {
    return [[self alloc] initWithRole:role content:content stats:nil modelIdentifier:nil];
}

+ (instancetype)messageWithRole:(APRole)role content:(NSArray<APContent *> *)content
                          stats:(APResponseStats *)stats modelIdentifier:(NSString *)modelIdentifier {
    return [[self alloc] initWithRole:role content:content stats:stats modelIdentifier:modelIdentifier];
}

+ (instancetype)systemMessageWithText:(NSString *)text {
    return [self messageWithRole:APRoleSystem content:@[ [APContent textContent:text] ]];
}
+ (instancetype)userMessageWithText:(NSString *)text {
    return [self messageWithRole:APRoleUser content:@[ [APContent textContent:text] ]];
}
+ (instancetype)assistantMessageWithText:(NSString *)text {
    return [self messageWithRole:APRoleAssistant content:@[ [APContent textContent:text] ]];
}

- (NSString *)textRepresentation {
    NSMutableString * s = [NSMutableString string];
    for (APContent * c in _content) {
        NSString * prompt = c.promptText;
        if (prompt) [s appendString:prompt];
    }
    return s;
}

- (NSString *)displayText {
    NSMutableString * s = [NSMutableString string];
    for (APContent * c in _content)
        if (c.kind == APContentKindText && c.text) [s appendString:c.text];
    return s;
}

- (NSString *)reasoning {
    NSMutableString * s = nil;
    for (APContent * c in _content) {
        if (c.kind != APContentKindReasoning || !c.text) continue;
        if (!s) s = [NSMutableString string];
        [s appendString:c.text];
    }
    return s.length ? s : nil;
}

- (NSArray<APToolCall *> *)toolCalls {
    NSMutableArray * calls = [NSMutableArray array];
    for (APContent * c in _content)
        if (c.kind == APContentKindToolCall && c.toolCall) [calls addObject:c.toolCall];
    return calls;
}

- (NSArray<APAttachment *> *)attachments {
    NSMutableArray * files = [NSMutableArray array];
    for (APContent * c in _content)
        if (c.kind == APContentKindAttachment && c.attachment) [files addObject:c.attachment];
    return files;
}

- (NSArray<APImage *> *)images {
    NSMutableArray * images = [NSMutableArray array];
    for (APContent * c in _content)
        if (c.kind == APContentKindImage && c.image) [images addObject:c.image];
    return images;
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end
