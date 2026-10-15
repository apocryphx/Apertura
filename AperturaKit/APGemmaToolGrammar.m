//  APGemmaToolGrammar — see header. Moved verbatim from APLocalSession.mm's static
//  helpers (2026-10-14); behavior unchanged by design (--tools-verify).
#import "APGemmaToolGrammar.h"

@implementation APGemmaToolGrammar

+ (NSString *)sanitize:(NSString *)s {
    return [[s stringByReplacingOccurrencesOfString:@"<|" withString:@"< |"]
                stringByReplacingOccurrencesOfString:@"|>" withString:@"| >"];
}

+ (NSString *)compactValue:(id)v {
    if ([v isKindOfClass:NSString.class]) {
        return [NSString stringWithFormat:@"<|\"|>%@<|\"|>", [self sanitize:v]];
    }
    if ([v isKindOfClass:NSNumber.class]) {
        NSNumber * n = v;
        if (n == (id) kCFBooleanTrue)  return @"true";
        if (n == (id) kCFBooleanFalse) return @"false";
        return n.stringValue;
    }
    if ([v isKindOfClass:NSArray.class]) {
        NSMutableArray * parts = [NSMutableArray array];
        for (id e in (NSArray *) v) [parts addObject:[self compactValue:e]];
        return [NSString stringWithFormat:@"[%@]", [parts componentsJoinedByString:@","]];
    }
    if ([v isKindOfClass:NSDictionary.class]) {
        NSDictionary * d = v;
        NSArray * keys = [d.allKeys sortedArrayUsingSelector:@selector(compare:)];
        NSMutableArray * parts = [NSMutableArray array];
        for (NSString * k in keys)
            [parts addObject:[NSString stringWithFormat:@"%@:%@", k, [self compactValue:d[k]]]];
        return [NSString stringWithFormat:@"{%@}", [parts componentsJoinedByString:@","]];
    }
    return @"null";
}

+ (id)uppercaseSchemaTypes:(id)node {
    if ([node isKindOfClass:NSDictionary.class]) {
        NSMutableDictionary * out = [NSMutableDictionary dictionaryWithCapacity:
                                     [(NSDictionary *)node count]];
        [(NSDictionary *)node enumerateKeysAndObjectsUsingBlock:^(NSString * key, id v, BOOL * stop) {
            if ([key isEqualToString:@"type"]) {
                if ([v isKindOfClass:NSString.class]) { out[key] = [v uppercaseString]; return; }
                if ([v isKindOfClass:NSArray.class]) {
                    NSMutableArray * types = [NSMutableArray array];
                    for (id t in (NSArray *)v)
                        [types addObject:[t isKindOfClass:NSString.class] ? [t uppercaseString] : t];
                    out[key] = types; return;
                }
            }
            out[key] = [self uppercaseSchemaTypes:v];
        }];
        return out;
    }
    if ([node isKindOfClass:NSArray.class]) {
        NSMutableArray * out = [NSMutableArray arrayWithCapacity:[(NSArray *)node count]];
        for (id v in (NSArray *)node) [out addObject:[self uppercaseSchemaTypes:v]];
        return out;
    }
    return node;
}

+ (NSString *)declarationForTool:(id<APTool>)tool {
    return [NSString stringWithFormat:@"declaration:%@{description:%@,parameters:%@}",
            tool.name, [self compactValue:tool.toolDescription],
            [self compactValue:[self uppercaseSchemaTypes:tool.parameterSchema ?: @{}]]];
}

+ (NSString *)JSONEscape:(NSString *)s {
    NSMutableString * out = [NSMutableString stringWithCapacity:s.length + 16];
    for (NSUInteger i = 0; i < s.length; i++) {
        unichar c = [s characterAtIndex:i];
        switch (c) {
            case '\\': [out appendString:@"\\\\"]; break;
            case '"':  [out appendString:@"\\\""]; break;
            case '\n': [out appendString:@"\\n"];  break;
            case '\r': [out appendString:@"\\r"];  break;
            case '\t': [out appendString:@"\\t"];  break;
            default:
                if (c < 0x20) [out appendFormat:@"\\u%04x", c];
                else [out appendFormat:@"%C", c];
        }
    }
    return out;
}

+ (NSDictionary<NSString *, id> *)parseArguments:(NSString *)rawBody {
    NSArray<NSString *> * parts =
        [(rawBody ?: @"") componentsSeparatedByString:@"<|\"|>"];
    BOOL needsBraces = ![[parts[0] stringByTrimmingCharactersInSet:
                          NSCharacterSet.whitespaceAndNewlineCharacterSet] hasPrefix:@"{"];
    NSRegularExpression * bareKey =
        [NSRegularExpression regularExpressionWithPattern:@"([{,]\\s*)([A-Za-z_][A-Za-z0-9_]*)\\s*:"
                                                  options:0 error:nil];
    NSMutableString * jsonish = [NSMutableString string];
    if (needsBraces) [jsonish appendString:@"{"];
    for (NSUInteger i = 0; i < parts.count; i++) {
        if (i % 2) {   // inside a quoted span: verbatim string content
            [jsonish appendFormat:@"\"%@\"", [self JSONEscape:parts[i]]];
        } else {       // structure: quote bare keys, leave everything else alone
            NSString * st = [bareKey stringByReplacingMatchesInString:parts[i] options:0
                                                                range:NSMakeRange(0, parts[i].length)
                                                         withTemplate:@"$1\"$2\":"];
            if (i == 0 && needsBraces) {   // key at the very start has no [{,] anchor
                NSRange first = [st rangeOfString:@"^\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*:"
                                          options:NSRegularExpressionSearch];
                if (first.location != NSNotFound) {
                    NSString * head = [st substringWithRange:first];
                    NSString * name = [[head stringByReplacingOccurrencesOfString:@":" withString:@""]
                        stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
                    st = [NSString stringWithFormat:@"\"%@\":%@", name,
                          [st substringFromIndex:NSMaxRange(first)]];
                }
            }
            [jsonish appendString:st];
        }
    }
    if (needsBraces) [jsonish appendString:@"}"];
    NSData * data = [jsonish dataUsingEncoding:NSUTF8StringEncoding];
    id parsed = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    if ([parsed isKindOfClass:NSDictionary.class]) return parsed;
    return @{ @"_raw" : rawBody ?: @"" };
}

@end
