//  APSemanticText — attribute keys for parsed chat text.
//
//  Parsed text records INTENT ("this run is bold", "this token is a keyword"), never
//  fonts or colors. Appearance is resolved by whatever displays the text, so a theme or
//  text-size change needs no re-parse, and this framework needs nothing from AppKit or
//  UIKit. The inline keys use the names and values of the TextKit2-Editor's SXDeltaKey*
//  (Ferrixcore), so semantic runs carry over between the two unchanged.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

#pragma mark - Font weight

/// A font weight on the CoreText scale (-1 … 1), the same numbers NSFontWeight,
/// UIFontWeight and SwiftUI's Font.Weight use, carried as a plain double so this
/// framework stays Foundation-only. A view maps it onto its toolkit's weight type.
typedef double APFontWeight NS_SWIFT_NAME(FontWeight);

FOUNDATION_EXPORT const APFontWeight APFontWeightRegular;    //  0.0
FOUNDATION_EXPORT const APFontWeight APFontWeightMedium;     //  0.23
FOUNDATION_EXPORT const APFontWeight APFontWeightSemibold;   //  0.3
FOUNDATION_EXPORT const APFontWeight APFontWeightBold;       //  0.4

#pragma mark - Inline keys

/// NSNumber (APFontWeight). Render bold when >= APFontWeightSemibold.
FOUNDATION_EXPORT NSAttributedStringKey const APSemanticFontWeightAttributeName;
/// NSNumber (BOOL).
FOUNDATION_EXPORT NSAttributedStringKey const APSemanticItalicAttributeName;
/// NSNumber (BOOL). An inline code span.
FOUNDATION_EXPORT NSAttributedStringKey const APSemanticCodeAttributeName;
/// NSNumber (BOOL).
FOUNDATION_EXPORT NSAttributedStringKey const APSemanticStrikethroughAttributeName;
/// NSString (the URL as written).
FOUNDATION_EXPORT NSAttributedStringKey const APSemanticLinkAttributeName;

#pragma mark - Code tokens

/// What a piece of code is, independent of how it gets colored.
typedef NS_ENUM(NSInteger, APCodeTokenKind) {
    APCodeTokenKindKeyword   = 0,
    APCodeTokenKindType      = 1,
    APCodeTokenKindString    = 2,
    APCodeTokenKindComment   = 3,
    APCodeTokenKindNumber    = 4,
    /// Attributes, decorators, preprocessor directives, JSON keys, shell variables.
    APCodeTokenKindAttribute = 5,
};

/// NSNumber (APCodeTokenKind). Marks runs of highlighted code; untagged runs are plain.
FOUNDATION_EXPORT NSAttributedStringKey const APCodeTokenAttributeName;

NS_ASSUME_NONNULL_END
