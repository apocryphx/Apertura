//  APReadableJSON — JSON laid out for reading.
//
//  Tool results often wrap multi-line text in JSON (`{"man": "Available commands:\n  arc …"}`),
//  which is unreadable as one escaped string. Here multi-line strings become indented text
//  blocks with real line breaks; everything else keeps JSON syntax. Tokens are tagged with
//  APCodeTokenAttributeName as they are written, so no regex pass is needed afterwards.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface APReadableJSON : NSObject

/// The readable layout of a JSON object or array, or nil when `text` is not JSON or has no
/// multi-line string anywhere in it — then the original is the better rendering.
+ (nullable NSAttributedString *)formatText:(NSString *)text;

/// The readable layout of an already-parsed JSON value (as NSJSONSerialization produces
/// it): JSON syntax, keys sorted, multi-line strings as text blocks. Never nil.
+ (NSAttributedString *)formatValue:(id)value;

@end

NS_ASSUME_NONNULL_END
