//  APImageTranscoding — see header.
#import "APImageTranscoding.h"
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

static NSString * const kJPEG = @"image/jpeg";
static NSString * const kPNG  = @"image/png";

BOOL APImageSniff(NSData * data, NSString * __autoreleasing * outMediaType, CGSize * outPixelSize) {
    if (data.length == 0) return NO;
    CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
    if (!source) return NO;
    BOOL ok = NO;
    CFStringRef uti = CGImageSourceGetType(source);
    NSString * mediaType = uti ? [UTType typeWithIdentifier:(__bridge NSString *)uti].preferredMIMEType : nil;
    if (mediaType && CGImageSourceGetCount(source) > 0) {
        NSDictionary * props = CFBridgingRelease(CGImageSourceCopyPropertiesAtIndex(source, 0, NULL));
        NSNumber * w = props[(__bridge NSString *)kCGImagePropertyPixelWidth];
        NSNumber * h = props[(__bridge NSString *)kCGImagePropertyPixelHeight];
        // A source whose first frame has no readable properties is not something we can
        // show or send; treat it as undecodable rather than storing a mystery.
        if (w && h) {
            ok = YES;
            if (outMediaType) *outMediaType = mediaType;
            if (outPixelSize) *outPixelSize = CGSizeMake(w.doubleValue, h.doubleValue);
        }
    }
    CFRelease(source);
    return ok;
}

NSData * APImageDataForTransport(NSData * data, NSString * mediaType, CGFloat maxLongEdge,
                                 NSSet<NSString *> * acceptedMediaTypes,
                                 NSString * __autoreleasing * outMediaType) {
    NSString * sniffed = nil;
    CGSize size = CGSizeZero;
    if (!APImageSniff(data, &sniffed, &size)) return nil;
    BOOL typeOK = [acceptedMediaTypes containsObject:mediaType];
    BOOL sizeOK = maxLongEdge <= 0 || MAX(size.width, size.height) <= maxLongEdge;
    if (typeOK && sizeOK) {
        if (outMediaType) *outMediaType = mediaType;
        return data;
    }

    CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, NULL);
    if (!source) return nil;
    CGImageRef image = NULL;
    if (sizeOK) {
        image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    } else {
        // ImageIO's thumbnail path is the bounded, orientation-aware decode: it never
        // materializes the full-size bitmap for a large JPEG.
        NSDictionary * opts = @{ (__bridge NSString *)kCGImageSourceCreateThumbnailFromImageAlways : @YES,
                                 (__bridge NSString *)kCGImageSourceThumbnailMaxPixelSize : @(maxLongEdge),
                                 (__bridge NSString *)kCGImageSourceCreateThumbnailWithTransform : @YES };
        image = CGImageSourceCreateThumbnailAtIndex(source, 0, (__bridge CFDictionaryRef)opts);
    }
    CFRelease(source);
    if (!image) return nil;

    // JPEG stays JPEG (photos; keeps the payload small); everything else becomes PNG so
    // alpha and line art survive.
    BOOL jpeg = [mediaType isEqualToString:kJPEG];
    NSString * outType = jpeg ? kJPEG : kPNG;
    UTType * uti = [UTType typeWithMIMEType:outType];
    NSMutableData * out = [NSMutableData data];
    CGImageDestinationRef dest = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)out,
                                                                  (__bridge CFStringRef)uti.identifier, 1, NULL);
    if (!dest) { CGImageRelease(image); return nil; }
    NSDictionary * props = jpeg ? @{ (__bridge NSString *)kCGImageDestinationLossyCompressionQuality : @0.85 } : @{};
    CGImageDestinationAddImage(dest, image, (__bridge CFDictionaryRef)props);
    BOOL ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(image);
    if (!ok) return nil;
    if (outMediaType) *outMediaType = outType;
    return out;
}
