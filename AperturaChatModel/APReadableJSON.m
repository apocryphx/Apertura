#import "APReadableJSON.h"
#import "APSemanticText.h"

static BOOL APIsMultilineString(id value) {
    return [value isKindOfClass:NSString.class] && [value containsString:@"\n"];
}

static BOOL APContainsMultilineString(id value) {
    if ([value isKindOfClass:NSString.class]) return APIsMultilineString(value);
    if ([value isKindOfClass:NSArray.class]) {
        for (id item in value) if (APContainsMultilineString(item)) return YES;
    } else if ([value isKindOfClass:NSDictionary.class]) {
        for (id item in [value allValues]) if (APContainsMultilineString(item)) return YES;
    }
    return NO;
}

static BOOL APIsBoolean(NSNumber * number) {
    return CFGetTypeID((__bridge CFTypeRef)number) == CFBooleanGetTypeID();
}

@interface APReadableJSONWriter : NSObject
@property (readonly) NSMutableAttributedString *output;
- (void)write:(id)value indent:(NSInteger)indent;
@end

@implementation APReadableJSONWriter

- (instancetype)init {
    if ((self = [super init])) _output = [[NSMutableAttributedString alloc] init];
    return self;
}

- (void)write:(id)value indent:(NSInteger)indent {
    if ([value isKindOfClass:NSDictionary.class]) {
        NSDictionary * object = value;
        NSArray * keys = [object.allKeys sortedArrayUsingSelector:@selector(compare:)];
        NSMutableArray * values = [NSMutableArray arrayWithCapacity:keys.count];
        for (NSString * key in keys) [values addObject:object[key]];
        [self writeContainerOpen:@"{" close:@"}" keys:keys values:values indent:indent];
    } else if ([value isKindOfClass:NSArray.class]) {
        [self writeContainerOpen:@"[" close:@"]" keys:nil values:value indent:indent];
    } else if (APIsMultilineString(value)) {
        // A text block: starts on its own line, one level deeper than the key. Left
        // untagged so it reads as prose in the normal text color.
        for (NSString * line in [value componentsSeparatedByString:@"\n"]) {
            [self emit:@"\n" kind:nil];
            if (line.length > 0) {
                [self emit:[self padding:indent + 1] kind:nil];
                [self emit:line kind:nil];
            }
        }
    } else if ([value isKindOfClass:NSString.class]) {
        NSString * escaped = [value stringByReplacingOccurrencesOfString:@"\"" withString:@"\\\""];
        [self emit:[NSString stringWithFormat:@"\"%@\"", escaped] kind:@(APCodeTokenKindString)];
    } else if ([value isKindOfClass:NSNumber.class]) {
        NSNumber * number = value;
        if (APIsBoolean(number)) {
            [self emit:number.boolValue ? @"true" : @"false" kind:@(APCodeTokenKindKeyword)];
        } else {
            double d = number.doubleValue;
            BOOL isWhole = round(d) == d && fabs(d) < 1e15;
            [self emit:isWhole ? [NSString stringWithFormat:@"%lld", (long long)d] : number.stringValue
                  kind:@(APCodeTokenKindNumber)];
        }
    } else {
        [self emit:@"null" kind:@(APCodeTokenKindKeyword)];
    }
}

- (void)writeContainerOpen:(NSString *)open close:(NSString *)close
                      keys:(nullable NSArray<NSString *> *)keys values:(NSArray *)values
                    indent:(NSInteger)indent {
    if (values.count == 0) {
        [self emit:[open stringByAppendingString:close] kind:nil];
        return;
    }
    [self emit:open kind:nil];
    for (NSUInteger index = 0; index < values.count; index++) {
        id value = values[index];
        [self emit:[@"\n" stringByAppendingString:[self padding:indent + 1]] kind:nil];
        if (keys) {
            [self emit:[NSString stringWithFormat:@"\"%@\"", keys[index]] kind:@(APCodeTokenKindAttribute)];
            [self emit:@":" kind:nil];
            if (!APIsMultilineString(value)) [self emit:@" " kind:nil];
        }
        [self write:value indent:indent + 1];
        // A comma after a text block would dangle at the end of its last line.
        if (index < values.count - 1 && !APIsMultilineString(value)) [self emit:@"," kind:nil];
    }
    [self emit:[NSString stringWithFormat:@"\n%@%@", [self padding:indent], close] kind:nil];
}

- (void)emit:(NSString *)text kind:(nullable NSNumber *)kind {
    NSDictionary * attributes = kind ? @{ APCodeTokenAttributeName : kind } : nil;
    [self.output appendAttributedString:[[NSAttributedString alloc] initWithString:text attributes:attributes]];
}

- (NSString *)padding:(NSInteger)level {
    return [@"" stringByPaddingToLength:(NSUInteger)(level * 2) withString:@" " startingAtIndex:0];
}

@end

@implementation APReadableJSON

+ (NSAttributedString *)formatText:(NSString *)text {
    NSData * data = [text dataUsingEncoding:NSUTF8StringEncoding];
    if (!data) return nil;
    id value = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:NULL];
    if (!value || !APContainsMultilineString(value)) return nil;
    return [self formatValue:value];
}

+ (NSAttributedString *)formatValue:(id)value {
    APReadableJSONWriter * writer = [[APReadableJSONWriter alloc] init];
    [writer write:value indent:0];
    return [writer.output copy];
}

@end
