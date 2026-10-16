//  APGoogleSessionInternal — PROJECT header (never public). The request-construction
//  surface of the Gemini backend, so AperturaKitTests can check what a turn would send
//  without a network: which parts a message becomes, and the request body those parts
//  ride in.
#pragma once
#import "APGoogleSession.h"
#import "APSessionSubclass.h"

NS_ASSUME_NONNULL_BEGIN

/// Longest edge, in pixels, an image is sent at. Larger images go as a downscaled COPY
/// built per request; the stored original is never touched. (Gemini tiles images at
/// 768 px; beyond ~2K there is nothing to gain and megabytes of base64 to lose.)
FOUNDATION_EXPORT const CGFloat APGoogleSessionMaximumImageEdge;

/// Media types the API takes as inlineData; anything else is re-encoded as PNG.
FOUNDATION_EXPORT NSSet<NSString *> * APGoogleSessionAcceptedImageMediaTypes(void);

@interface APGoogleSession (Internal)

/// The `parts` array of one user (or replayed model) turn: every image as an inlineData
/// part first (bytes read from APImage.fileURL, downscaled or transcoded when needed),
/// then the text. `missing` decides what happens to an image whose bytes are not there:
/// YES substitutes a text placeholder (replayed history survives a lost file), NO fails
/// with APErrorFileMissing (a live turn must not silently lose its picture). nil only
/// on failure.
- (nullable NSArray<NSDictionary *> *)partsForMessage:(APMessage *)message
                                   placeholderForMissing:(BOOL)missing
                                                   error:(NSError **)error;

/// The streamGenerateContent request for `contents`; nil (with the error) when the key
/// provider returns nothing or the body cannot be encoded.
- (nullable NSURLRequest *)requestWithContents:(NSArray<NSDictionary *> *)contents
                                         tools:(NSArray<id<APTool>> *)activeTools
                                       options:(APGenerationOptions *)opts
                                         error:(NSError **)error;

/// The API-side history after prime (the seed turns) — what the next request replays.
@property (readonly) NSArray<NSDictionary *> *contentsForTesting;

@end

NS_ASSUME_NONNULL_END
