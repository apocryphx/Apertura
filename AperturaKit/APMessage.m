#import "APMessage.h"

@implementation APMessage

- (instancetype)initWithRole:(APRole)role content:(NSArray<APContent *> *)content {
    if ((self = [super init])) {
        _role = role;
        _content = [content copy];
    }
    return self;
}

+ (instancetype)messageWithRole:(APRole)role content:(NSArray<APContent *> *)content {
    return [[self alloc] initWithRole:role content:content];
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
        if (c.kind == APContentKindText && c.text) [s appendString:c.text];
    }
    return s;
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end
