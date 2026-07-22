#import "APModelConfiguration.h"

@implementation APModelConfiguration

+ (instancetype)defaultConfiguration { return [[self alloc] init]; }

- (instancetype)init {
    if ((self = [super init])) {
        _headBits = 8;
        _prefillChunkLength = 512;
        _maximumContextLength = 0;   // model maximum
        _instrumented = NO;
    }
    return self;
}

- (id)copyWithZone:(NSZone *)zone {
    APModelConfiguration * c = [[[self class] allocWithZone:zone] init];
    c.headBits = self.headBits;
    c.prefillChunkLength = self.prefillChunkLength;
    c.maximumContextLength = self.maximumContextLength;
    c.instrumented = self.instrumented;
    return c;
}

@end
