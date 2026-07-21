#import "APGenerationOptions.h"

@implementation APGenerationOptions

+ (instancetype)deterministicOptions {
    APGenerationOptions * o = [[self alloc] init];
    o.temperature = 0;
    return o;
}

+ (instancetype)defaultOptions { return [[self alloc] init]; }

- (instancetype)init {
    if ((self = [super init])) {
        _temperature = 0.7f;
        _topK = 64;
        _topP = 0.95f;
        _maximumResponseTokens = 0;
    }
    return self;
}

- (id)copyWithZone:(NSZone *)zone {
    APGenerationOptions * o = [[[self class] allocWithZone:zone] init];
    o.temperature = self.temperature;
    o.topK = self.topK;
    o.topP = self.topP;
    o.maximumResponseTokens = self.maximumResponseTokens;
    return o;
}

@end
