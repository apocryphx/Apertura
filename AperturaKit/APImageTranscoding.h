//  APImageTranscoding — PROJECT header (never public). ImageIO helpers the store and the
//  vision backends share: sniff bytes for their type and size, and produce a bounded copy
//  for a request without touching the stored original.
#pragma once
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/// YES when `data` decodes as an image. `outMediaType` receives the IANA type ImageIO
/// identified ("image/png"), `outPixelSize` the stored pixel dimensions.
FOUNDATION_EXPORT BOOL APImageSniff(NSData *data,
                                    NSString *_Nullable __autoreleasing *_Nullable outMediaType,
                                    CGSize *_Nullable outPixelSize);

/// The bytes to send for an image whose long edge may exceed `maxLongEdge` pixels, or whose
/// type is not among `acceptedMediaTypes`: the original data when it already qualifies,
/// otherwise a re-encoded copy (JPEG for a JPEG source, PNG for anything else) scaled so
/// its long edge is at most `maxLongEdge`. `outMediaType` receives the type of what is
/// returned. nil when the data cannot be decoded.
FOUNDATION_EXPORT NSData *_Nullable APImageDataForTransport(NSData *data,
                                                            NSString *mediaType,
                                                            CGFloat maxLongEdge,
                                                            NSSet<NSString *> *acceptedMediaTypes,
                                                            NSString *_Nullable __autoreleasing *_Nullable outMediaType);

#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_END
