#import "APCodeHighlighter.h"
#import "APSemanticText.h"

#pragma mark - APCodeSnippet

@implementation APCodeSnippet

- (instancetype)initWithLanguage:(NSString *)language highlighted:(NSAttributedString *)highlighted {
    if ((self = [super init])) {
        _language = [language copy];
        _highlighted = [highlighted copy];
        _code = [highlighted.string copy];
    }
    return self;
}

+ (instancetype)snippetWithLanguage:(NSString *)language code:(NSString *)code {
    return [[self alloc] initWithLanguage:language
                              highlighted:[APCodeHighlighter highlightCode:code language:language]];
}

+ (instancetype)snippetWithLanguage:(NSString *)language highlighted:(NSAttributedString *)highlighted {
    return [[self alloc] initWithLanguage:language highlighted:highlighted];
}

- (id)copyWithZone:(NSZone *)zone { return self; }   // immutable

@end

#pragma mark - Grammars

/// A rule's kind: a token kind, or APRuleIdentifier for a word classified afterwards.
static const NSInteger APRuleIdentifier = -1;

@interface APCodeGrammar : NSObject
/// NSNumber (APCodeTokenKind or APRuleIdentifier), one per capture group.
@property (copy) NSArray<NSNumber *> *ruleKinds;
@property (copy) NSSet<NSString *> *keywords;
@property (copy) NSSet<NSString *> *types;
/// Treat capitalized words as types (Swift, Objective-C, Python classes).
@property BOOL capitalizedAreTypes;
/// Match keywords and types ignoring case (SQL); both lists must then be lowercase.
@property BOOL caseInsensitive;
@property NSRegularExpression *regex;
@end

@implementation APCodeGrammar

/// `rules` alternates pattern and kind: @[pattern, @(kind), pattern, @(kind), …]. All groups
/// inside patterns are non-capturing so rule indices line up with match groups.
+ (instancetype)grammarWithRules:(NSArray *)rules
                        keywords:(NSArray<NSString *> *)keywords
                           types:(NSArray<NSString *> *)types
             capitalizedAreTypes:(BOOL)capitalizedAreTypes {
    APCodeGrammar * grammar = [[APCodeGrammar alloc] init];
    NSMutableArray<NSString *> * patterns = [NSMutableArray array];
    NSMutableArray<NSNumber *> * kinds = [NSMutableArray array];
    for (NSUInteger i = 0; i + 1 < rules.count; i += 2) {
        [patterns addObject:[NSString stringWithFormat:@"(%@)", rules[i]]];
        [kinds addObject:rules[i + 1]];
    }
    grammar.ruleKinds = kinds;
    grammar.keywords = [NSSet setWithArray:keywords];
    grammar.types = [NSSet setWithArray:types];
    grammar.capitalizedAreTypes = capitalizedAreTypes;
    // One alternation, earlier rules win: comments and strings shadow anything inside them.
    NSError * error = nil;
    grammar.regex = [NSRegularExpression regularExpressionWithPattern:[patterns componentsJoinedByString:@"|"]
                                                              options:NSRegularExpressionAnchorsMatchLines
                                                                error:&error];
    NSAssert(grammar.regex, @"Invalid grammar pattern: %@", error);
    return grammar;
}

/// The token kind for a word, or nil when it is plain.
- (nullable NSNumber *)classify:(NSString *)word {
    NSString * key = self.caseInsensitive ? word.lowercaseString : word;
    if ([self.keywords containsObject:key]) return @(APCodeTokenKindKeyword);
    if ([self.types containsObject:key]) return @(APCodeTokenKindType);
    if (self.capitalizedAreTypes && word.length > 0 &&
        [NSCharacterSet.uppercaseLetterCharacterSet characterIsMember:[word characterAtIndex:0]]) {
        return @(APCodeTokenKindType);
    }
    return nil;
}

+ (nullable APCodeGrammar *)grammarNamed:(NSString *)language {
    static NSDictionary<NSString *, APCodeGrammar *> * grammars;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        APCodeGrammar * swift = [self swift], * python = [self python], * objc = [self objectiveC];
        APCodeGrammar * json = [self json], * shell = [self shell];
        APCodeGrammar * script = [self script], * rust = [self rust], * go = [self go];
        APCodeGrammar * markup = [self markup], * css = [self css], * yaml = [self yaml];
        APCodeGrammar * sql = [self sql], * java = [self java], * kotlin = [self kotlin];
        grammars = @{
            @"typescript": script, @"ts": script, @"tsx": script,
            @"javascript": script, @"js": script, @"jsx": script, @"mjs": script, @"cjs": script,
            @"rust": rust, @"rs": rust,
            @"go": go, @"golang": go,
            @"html": markup, @"htm": markup, @"xml": markup, @"svg": markup,
            @"css": css, @"scss": css,
            @"yaml": yaml, @"yml": yaml,
            @"sql": sql,
            @"java": java,
            @"kotlin": kotlin, @"kt": kotlin, @"kts": kotlin,
            @"swift": swift,
            @"python": python, @"py": python, @"python3": python,
            @"objc": objc, @"objective-c": objc, @"objectivec": objc, @"obj-c": objc,
            @"m": objc, @"c": objc, @"h": objc, @"cpp": objc, @"c++": objc,
            @"json": json, @"jsonc": json,
            @"sh": shell, @"bash": shell, @"zsh": shell, @"shell": shell,
            @"console": shell, @"shell-session": shell,
        };
    });
    return grammars[language.lowercaseString];
}

// Shared patterns (Objective-C has no raw strings: every regex backslash is doubled).
#define APDoubleQuoted @"\"(?:[^\"\\\\\\n]|\\\\.)*\""
#define APSingleQuoted @"'(?:[^'\\\\\\n]|\\\\.)*'"
#define APSlashComment @"//[^\\n]*"
#define APBlockComment @"/\\*[\\s\\S]*?\\*/"
#define APHashComment  @"#[^\\n]*"
#define APNumberBody   @"(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|\\d[\\d_]*(?:\\.\\d[\\d_]*)?(?:[eE][+-]?\\d+)?)"
#define APNumber       @"\\b" APNumberBody @"\\b"
#define APIdentifier   @"\\b[A-Za-z_][A-Za-z0-9_]*\\b"

#define KIND(k) @(APCodeTokenKind##k)
#define IDENT   @(APRuleIdentifier)

+ (APCodeGrammar *)swift {
    return [self grammarWithRules:@[
        APBlockComment, KIND(Comment),
        APSlashComment, KIND(Comment),
        @"#?\"\"\"[\\s\\S]*?\"\"\"#?", KIND(String),
        APDoubleQuoted, KIND(String),
        @"[@#][A-Za-z_]\\w*", KIND(Attribute),
        APNumber, KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"actor", @"any", @"as", @"associatedtype", @"async", @"await", @"break", @"case", @"catch", @"class",
        @"continue", @"convenience", @"default", @"defer", @"deinit", @"didSet", @"do", @"dynamic", @"else",
        @"enum", @"extension", @"fallthrough", @"false", @"fileprivate", @"final", @"for", @"func", @"get",
        @"guard", @"if", @"import", @"in", @"indirect", @"init", @"inout", @"internal", @"is", @"lazy", @"let",
        @"mutating", @"nil", @"nonisolated", @"nonmutating", @"open", @"operator", @"override", @"private",
        @"protocol", @"public", @"repeat", @"required", @"rethrows", @"return", @"self", @"Self", @"set",
        @"some", @"static", @"struct", @"subscript", @"super", @"switch", @"throw", @"throws", @"true", @"try",
        @"typealias", @"unowned", @"var", @"weak", @"where", @"while", @"willSet",
    ] types:@[] capitalizedAreTypes:YES];
}

+ (APCodeGrammar *)python {
    return [self grammarWithRules:@[
        APHashComment, KIND(Comment),
        @"\\b[rRbBfFuU]{0,2}(?:\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?''')", KIND(String),
        @"\\b[rRbBfFuU]{1,2}(?:" APDoubleQuoted @"|" APSingleQuoted @")", KIND(String),
        APDoubleQuoted, KIND(String),
        APSingleQuoted, KIND(String),
        @"^[ \\t]*@[\\w.]+", KIND(Attribute),
        APNumber, KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"False", @"None", @"True", @"and", @"as", @"assert", @"async", @"await", @"break", @"class", @"continue",
        @"def", @"del", @"elif", @"else", @"except", @"finally", @"for", @"from", @"global", @"if", @"import",
        @"in", @"is", @"lambda", @"match", @"case", @"nonlocal", @"not", @"or", @"pass", @"raise", @"return",
        @"self", @"try", @"while", @"with", @"yield",
    ] types:@[
        @"bool", @"bytes", @"dict", @"float", @"frozenset", @"int", @"list", @"object", @"set", @"str", @"tuple",
        @"type", @"len", @"print", @"range", @"enumerate", @"zip", @"map", @"filter", @"sorted", @"open",
        @"isinstance", @"super",
    ] capitalizedAreTypes:YES];
}

+ (APCodeGrammar *)objectiveC {
    return [self grammarWithRules:@[
        APBlockComment, KIND(Comment),
        APSlashComment, KIND(Comment),
        @"^[ \\t]*#[ \\t]*\\w+(?:[ \\t]+<[^>\\n]*>)?", KIND(Attribute),
        @"@?" APDoubleQuoted, KIND(String),
        APSingleQuoted, KIND(String),
        @"@[A-Za-z_]\\w*", KIND(Keyword),
        @"@?" APNumber, KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"auto", @"break", @"case", @"const", @"continue", @"default", @"do", @"else", @"enum", @"extern",
        @"for", @"goto", @"if", @"inline", @"register", @"restrict", @"return", @"sizeof", @"static", @"struct",
        @"switch", @"typedef", @"union", @"volatile", @"while", @"self", @"super", @"nil", @"Nil", @"NULL",
        @"YES", @"NO", @"true", @"false", @"nonatomic", @"atomic", @"strong", @"weak", @"copy", @"assign",
        @"readonly", @"readwrite", @"nullable", @"nonnull", @"instancetype", @"__block", @"__weak",
        @"__strong", @"class", @"namespace", @"template", @"typename", @"public", @"private", @"protected",
        @"virtual", @"new", @"delete", @"this", @"using",
    ] types:@[
        @"void", @"char", @"short", @"int", @"long", @"float", @"double", @"signed", @"unsigned", @"bool",
        @"id", @"BOOL", @"SEL", @"IMP", @"Class", @"size_t", @"uint8_t", @"uint16_t", @"uint32_t", @"uint64_t",
        @"int8_t", @"int16_t", @"int32_t", @"int64_t", @"CGFloat",
    ] capitalizedAreTypes:YES];
}

+ (APCodeGrammar *)json {
    return [self grammarWithRules:@[
        APDoubleQuoted @"(?=\\s*:)", KIND(Attribute),
        APDoubleQuoted, KIND(String),
        @"-?" APNumber, KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[@"true", @"false", @"null"] types:@[] capitalizedAreTypes:NO];
}

+ (APCodeGrammar *)shell {
    return [self grammarWithRules:@[
        @"(?:^|(?<=\\s))#[^\\n]*", KIND(Comment),
        APDoubleQuoted, KIND(String),
        @"'[^']*'", KIND(String),
        @"\\$(?:\\{[^}\\n]*\\}|[A-Za-z_]\\w*|[@#?$!*0-9])", KIND(Attribute),
        @"(?<=\\s)--?[A-Za-z][\\w-]*", KIND(Type),
        APNumber, KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"if", @"then", @"else", @"elif", @"fi", @"for", @"while", @"until", @"do", @"done", @"case", @"esac",
        @"in", @"function", @"return", @"export", @"local", @"readonly", @"source", @"alias", @"unset",
        @"sudo", @"cd", @"echo", @"exit", @"set",
    ] types:@[] capitalizedAreTypes:NO];
}

/// TypeScript and JavaScript (JSX included; regex literals are not recognized).
+ (APCodeGrammar *)script {
    return [self grammarWithRules:@[
        APBlockComment, KIND(Comment),
        APSlashComment, KIND(Comment),
        @"`(?:[^`\\\\]|\\\\[\\s\\S])*`", KIND(String),
        APDoubleQuoted, KIND(String),
        APSingleQuoted, KIND(String),
        @"@[A-Za-z_]\\w*", KIND(Attribute),
        APNumber, KIND(Number),
        @"\\b[A-Za-z_$][A-Za-z0-9_$]*\\b", IDENT,
    ] keywords:@[
        @"abstract", @"as", @"async", @"await", @"break", @"case", @"catch", @"class", @"const", @"continue",
        @"debugger", @"declare", @"default", @"delete", @"do", @"else", @"enum", @"export", @"extends",
        @"false", @"finally", @"for", @"from", @"function", @"get", @"if", @"implements", @"import", @"in",
        @"instanceof", @"interface", @"keyof", @"let", @"namespace", @"new", @"null", @"of", @"override",
        @"private", @"protected", @"public", @"readonly", @"return", @"satisfies", @"set", @"static",
        @"super", @"switch", @"this", @"throw", @"true", @"try", @"type", @"typeof", @"undefined", @"var",
        @"void", @"while", @"with", @"yield",
    ] types:@[
        @"any", @"bigint", @"boolean", @"never", @"number", @"object", @"string", @"symbol", @"unknown",
    ] capitalizedAreTypes:YES];
}

+ (APCodeGrammar *)rust {
    return [self grammarWithRules:@[
        APBlockComment, KIND(Comment),
        APSlashComment, KIND(Comment),
        @"\\bb?r#*\"[\\s\\S]*?\"#*", KIND(String),
        APDoubleQuoted, KIND(String),
        @"'(?:[^'\\\\\\n]|\\\\.[^'\\n]*)'", KIND(String),
        @"'[A-Za-z_]\\w*", KIND(Attribute),
        @"#!?\\[[^\\]\\n]*\\]", KIND(Attribute),
        @"\\b[A-Za-z_]\\w*!(?!=)", KIND(Attribute),
        @"\\b" APNumberBody @"(?:[iu](?:8|16|32|64|128|size)|f32|f64)?\\b", KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"as", @"async", @"await", @"break", @"const", @"continue", @"crate", @"dyn", @"else", @"enum",
        @"extern", @"false", @"fn", @"for", @"if", @"impl", @"in", @"let", @"loop", @"match", @"mod",
        @"move", @"mut", @"pub", @"ref", @"return", @"self", @"Self", @"static", @"struct", @"super",
        @"trait", @"true", @"type", @"unsafe", @"use", @"where", @"while",
    ] types:@[
        @"bool", @"char", @"f32", @"f64", @"i8", @"i16", @"i32", @"i64", @"i128", @"isize", @"str", @"u8",
        @"u16", @"u32", @"u64", @"u128", @"usize",
    ] capitalizedAreTypes:YES];
}

+ (APCodeGrammar *)go {
    return [self grammarWithRules:@[
        APBlockComment, KIND(Comment),
        APSlashComment, KIND(Comment),
        @"`[^`]*`", KIND(String),
        APDoubleQuoted, KIND(String),
        APSingleQuoted, KIND(String),
        APNumber, KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"break", @"case", @"chan", @"const", @"continue", @"default", @"defer", @"else", @"fallthrough",
        @"false", @"for", @"func", @"go", @"goto", @"if", @"import", @"interface", @"iota", @"map",
        @"nil", @"package", @"range", @"return", @"select", @"struct", @"switch", @"true", @"type", @"var",
    ] types:@[
        @"any", @"bool", @"byte", @"complex64", @"complex128", @"error", @"float32", @"float64", @"int",
        @"int8", @"int16", @"int32", @"int64", @"rune", @"string", @"uint", @"uint8", @"uint16",
        @"uint32", @"uint64", @"uintptr", @"append", @"cap", @"close", @"copy", @"delete", @"len",
        @"make", @"new", @"panic", @"print", @"println", @"recover",
    ] capitalizedAreTypes:YES];
}

/// HTML and XML: tag names are keywords, attribute names attributes; strings count only as
/// attribute values so apostrophes in text stay plain.
+ (APCodeGrammar *)markup {
    return [self grammarWithRules:@[
        @"<!--[\\s\\S]*?-->", KIND(Comment),
        @"<!\\[CDATA\\[[\\s\\S]*?\\]\\]>", KIND(String),
        @"<![A-Za-z][^>]*>", KIND(Keyword),
        @"<\\?[\\s\\S]*?\\?>", KIND(Attribute),
        @"(?<==)\\s*(?:\"[^\"]*\"|'[^']*')", KIND(String),
        @"(?<=</|<)[A-Za-z][\\w:.-]*", KIND(Keyword),
        @"[A-Za-z_:@][\\w:.@-]*(?=\\s*=)", KIND(Attribute),
    ] keywords:@[] types:@[] capitalizedAreTypes:NO];
}

/// CSS and SCSS. Properties are recognized by a colon followed by whitespace, which leaves
/// pseudo-classes (`a:hover`) alone.
+ (APCodeGrammar *)css {
    return [self grammarWithRules:@[
        APBlockComment, KIND(Comment),
        APSlashComment, KIND(Comment),
        @"url\\([^)'\"\\n]*\\)", KIND(String),
        APDoubleQuoted, KIND(String),
        APSingleQuoted, KIND(String),
        @"@[A-Za-z-]+", KIND(Keyword),
        @"!\\s*important\\b", KIND(Keyword),
        @"#[0-9a-fA-F]{3,8}\\b", KIND(Number),
        @"(?<![\\w#-])-?(?:\\d+\\.?\\d*|\\.\\d+)(?:%|[A-Za-z]+)?", KIND(Number),
        @"\\$[\\w-]+", KIND(Attribute),
        @"[A-Za-z-][\\w-]*(?=\\s*:\\s)", KIND(Attribute),
        @"[.#][A-Za-z_-][\\w-]*", KIND(Type),
    ] keywords:@[] types:@[] capitalizedAreTypes:NO];
}

+ (APCodeGrammar *)yaml {
    return [self grammarWithRules:@[
        APHashComment, KIND(Comment),
        @"(?:\"(?:[^\"\\\\\\n]|\\\\.)*\"|'[^'\\n]*')(?=\\s*:(?:\\s|$))", KIND(Attribute),
        @"(?<!\\w)" APDoubleQuoted, KIND(String),
        @"(?<!\\w)" APSingleQuoted, KIND(String),
        @"^(?:---|\\.\\.\\.)(?=\\s|$)", KIND(Keyword),
        @"(?<![\\w.$/-])[\\w.$/-]+(?=\\s*:(?:\\s|$))", KIND(Attribute),
        @"[&*][\\w-]+", KIND(Type),
        @"(?<![\\w.:-])-?\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?(?![\\w.:-])", KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"true", @"false", @"null", @"True", @"False", @"Null", @"TRUE", @"FALSE", @"NULL",
    ] types:@[] capitalizedAreTypes:NO];
}

+ (APCodeGrammar *)sql {
    APCodeGrammar * grammar = [self grammarWithRules:@[
        @"--[^\\n]*", KIND(Comment),
        APBlockComment, KIND(Comment),
        @"'(?:[^']|'')*'", KIND(String),
        @"\"(?:[^\"]|\"\")*\"|`[^`]*`", KIND(Attribute),
        APNumber, KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"add", @"all", @"alter", @"and", @"as", @"asc", @"begin", @"between", @"by", @"case", @"check",
        @"commit", @"constraint", @"create", @"cross", @"default", @"delete", @"desc", @"distinct",
        @"drop", @"else", @"end", @"exists", @"false", @"foreign", @"from", @"full", @"group", @"having",
        @"in", @"index", @"inner", @"insert", @"into", @"is", @"join", @"key", @"left", @"like", @"limit",
        @"not", @"null", @"offset", @"on", @"or", @"order", @"outer", @"primary", @"references",
        @"returning", @"right", @"rollback", @"select", @"set", @"table", @"then", @"true", @"union",
        @"unique", @"update", @"values", @"view", @"when", @"where", @"with",
    ] types:@[
        @"bigint", @"blob", @"boolean", @"char", @"date", @"decimal", @"double", @"float", @"int",
        @"integer", @"json", @"numeric", @"real", @"serial", @"smallint", @"text", @"time", @"timestamp",
        @"uuid", @"varchar", @"count", @"sum", @"avg", @"min", @"max",
    ] capitalizedAreTypes:NO];
    grammar.caseInsensitive = YES;
    return grammar;
}

+ (APCodeGrammar *)java {
    return [self grammarWithRules:@[
        APBlockComment, KIND(Comment),
        APSlashComment, KIND(Comment),
        @"\"\"\"[\\s\\S]*?\"\"\"", KIND(String),
        APDoubleQuoted, KIND(String),
        APSingleQuoted, KIND(String),
        @"@[A-Za-z_]\\w*", KIND(Attribute),
        @"\\b" APNumberBody @"[lLfFdD]?\\b", KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"abstract", @"assert", @"break", @"case", @"catch", @"class", @"const", @"continue", @"default",
        @"do", @"else", @"enum", @"extends", @"false", @"final", @"finally", @"for", @"goto", @"if",
        @"implements", @"import", @"instanceof", @"interface", @"native", @"new", @"null", @"package",
        @"permits", @"private", @"protected", @"public", @"record", @"return", @"sealed", @"static",
        @"strictfp", @"super", @"switch", @"synchronized", @"this", @"throw", @"throws", @"transient",
        @"true", @"try", @"var", @"volatile", @"while", @"yield",
    ] types:@[
        @"boolean", @"byte", @"char", @"double", @"float", @"int", @"long", @"short", @"void",
    ] capitalizedAreTypes:YES];
}

+ (APCodeGrammar *)kotlin {
    return [self grammarWithRules:@[
        APBlockComment, KIND(Comment),
        APSlashComment, KIND(Comment),
        @"\"\"\"[\\s\\S]*?\"\"\"", KIND(String),
        APDoubleQuoted, KIND(String),
        APSingleQuoted, KIND(String),
        @"@[A-Za-z_]\\w*", KIND(Attribute),
        @"\\b" APNumberBody @"[lLfFuU]?\\b", KIND(Number),
        APIdentifier, IDENT,
    ] keywords:@[
        @"abstract", @"annotation", @"as", @"break", @"by", @"catch", @"class", @"companion", @"const",
        @"constructor", @"continue", @"crossinline", @"data", @"do", @"else", @"enum", @"false",
        @"final", @"finally", @"for", @"fun", @"get", @"if", @"import", @"in", @"infix", @"init",
        @"inline", @"inner", @"interface", @"internal", @"is", @"lateinit", @"noinline", @"null",
        @"object", @"open", @"operator", @"out", @"override", @"package", @"private", @"protected",
        @"public", @"reified", @"return", @"sealed", @"set", @"super", @"suspend", @"tailrec", @"this",
        @"throw", @"true", @"try", @"typealias", @"val", @"var", @"vararg", @"when", @"where", @"while",
    ] types:@[] capitalizedAreTypes:YES];
}

@end

#pragma mark - APCodeHighlighter

@implementation APCodeHighlighter

+ (NSAttributedString *)highlightCode:(NSString *)code language:(NSString *)language {
    NSMutableAttributedString * result = [[NSMutableAttributedString alloc] initWithString:code];
    APCodeGrammar * grammar = language ? [APCodeGrammar grammarNamed:language] : nil;
    if (!grammar) return result;

    NSArray<NSTextCheckingResult *> * matches =
        [grammar.regex matchesInString:code options:0 range:NSMakeRange(0, code.length)];
    for (NSTextCheckingResult * match in matches) {
        for (NSUInteger rule = 0; rule < grammar.ruleKinds.count; rule++) {
            NSRange range = [match rangeAtIndex:rule + 1];
            if (range.location == NSNotFound) continue;

            NSNumber * kind = grammar.ruleKinds[rule];
            if (kind.integerValue == APRuleIdentifier) {
                kind = [grammar classify:[code substringWithRange:range]];
            }
            if (kind) [result addAttribute:APCodeTokenAttributeName value:kind range:range];
            break;
        }
    }
    return result;
}

@end
