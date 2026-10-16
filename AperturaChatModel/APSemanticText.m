#import "APSemanticText.h"

const APFontWeight APFontWeightRegular  = 0.0;
const APFontWeight APFontWeightMedium   = 0.23;
const APFontWeight APFontWeightSemibold = 0.3;
const APFontWeight APFontWeightBold     = 0.4;

// The TextKit2-Editor's key names; keep them.
NSAttributedStringKey const APSemanticFontWeightAttributeName    = @"fontWeight";
NSAttributedStringKey const APSemanticItalicAttributeName        = @"italic";
NSAttributedStringKey const APSemanticCodeAttributeName          = @"code";
NSAttributedStringKey const APSemanticStrikethroughAttributeName = @"strikethrough";
NSAttributedStringKey const APSemanticLinkAttributeName          = @"link";

NSAttributedStringKey const APCodeTokenAttributeName = @"AP_codeToken";
