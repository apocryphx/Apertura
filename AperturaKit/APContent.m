#import "APContent.h"

@implementation APContent

- (instancetype)initWithKind:(APContentKind)kind text:(NSString *)text {
    if ((self = [super init])) {
        _kind = kind;
        _text = [text copy];
    }
    return self;
}

+ (instancetype)textContent:(NSString *)text {
    return [[self alloc] initWithKind:APContentKindText text:text];
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end
