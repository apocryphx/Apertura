//  APCodeHighlighter — token-kind tagging for code shown in a chat.
//
//  A small regex grammar per language, good enough for reading code in a transcript
//  (not an editor): comments and strings shadow what is inside them, identifiers are
//  classified against keyword and type lists. Output carries APCodeTokenAttributeName
//  runs only (APSemanticText.h); color is the view's business.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// A piece of code tokenized once, when its block was parsed.
@interface APCodeSnippet : NSObject <NSCopying>

/// Tokenizes `code` with the grammar for `language`; plain when the language is unknown.
+ (instancetype)snippetWithLanguage:(nullable NSString *)language code:(NSString *)code;
/// For text that was tagged while it was written, like APReadableJSON's output.
+ (instancetype)snippetWithLanguage:(nullable NSString *)language
                        highlighted:(NSAttributedString *)highlighted;

/// The fence's info string ("swift", "json"), as written; nil when there was none.
@property (nullable, readonly) NSString *language;
@property (readonly) NSString *code;
/// `code` with APCodeTokenAttributeName runs.
@property (readonly) NSAttributedString *highlighted;

- (instancetype)init NS_UNAVAILABLE;

@end

@interface APCodeHighlighter : NSObject

/// `code` with APCodeTokenAttributeName runs; untagged when `language` has no grammar.
/// Grammars (language names are case-insensitive):
///   swift
///   python (py, python3)
///   objective-c (objc, objectivec, obj-c, m, c, h, cpp, c++)
///   json (jsonc)
///   shell (sh, bash, zsh, console, shell-session)
///   typescript / javascript (ts, tsx, js, jsx, mjs, cjs)
///   rust (rs)
///   go (golang)
///   html / xml (htm, svg)
///   css (scss)
///   yaml (yml)
///   sql
///   java
///   kotlin (kt, kts)
+ (NSAttributedString *)highlightCode:(NSString *)code language:(nullable NSString *)language;

@end

NS_ASSUME_NONNULL_END
