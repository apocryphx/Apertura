//  APGemmaToolGrammar — the Gemma-4 compact tool grammar, as text.
//
//  PROJECT header (never public). Pure Objective-C, stateless: the string-level codec
//  between JSON-ish tool values and the grammar the reference chat template renders
//  (declarations advertised in the system turn, argument bodies the model emits, and
//  the response bodies spliced back). No engine, no tokens — the token-level work
//  (encodeQuoted, marker ids) stays in ESChatTemplate; APLocalSession is the only
//  caller and --tools-verify gates the whole loop. Every method here is deterministic
//  and model-free, so the unit tests cover it without a bundle.
#import <Foundation/Foundation.h>
#import "APTool.h"

NS_ASSUME_NONNULL_BEGIN

@interface APGemmaToolGrammar : NSObject

/// Strings that reach the grammar must not smuggle marker text: "<|" and "|>" are
/// broken with a space.
+ (NSString *)sanitize:(NSString *)string;

/// Render a JSON-ish value (NSString / NSNumber / NSArray / NSDictionary) into the
/// compact grammar: bare keys (sorted), strings in <|"|> quote markers, booleans as
/// true/false, arrays and objects in their plain bracketed forms; anything else → null.
+ (NSString *)compactValue:(nullable id)value;

/// Deep copy of a JSON-Schema fragment with every `type` VALUE uppercased (STRING,
/// OBJECT, …; arrays of type names too) — `value['type'] | upper` in the official Jinja.
/// Everything else (descriptions, enums, property names) keeps its case.
+ (id)uppercaseSchemaTypes:(id)node;

/// declaration:NAME{description:<|"|>…<|"|>,parameters:{…}} — the system-turn line for
/// one tool (the CLI --tools file format, generated).
+ (NSString *)declarationForTool:(id<APTool>)tool;

/// JSON string-escape (backslash, quote, \n \r \t, other controls as \uXXXX).
+ (NSString *)JSONEscape:(NSString *)string;

/// Best-effort parse of the model's compact argument body `key:val,...` into a
/// dictionary. The body alternates structure and <|"|>-quoted string spans; span
/// contents are JSON-escaped verbatim (newlines, quotes, braces and all), bare keys in
/// the structure parts get quoted, then JSON parses. Falls back to {"_raw": body} so
/// tools always receive something actionable.
+ (NSDictionary<NSString *, id> *)parseArguments:(nullable NSString *)rawBody;

@end

NS_ASSUME_NONNULL_END
