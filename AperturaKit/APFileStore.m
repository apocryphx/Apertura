//  APFileStore — see header.
#import "APFileStore.h"
#import "APError.h"
#import "APImageTranscoding.h"
#import <CommonCrypto/CommonDigest.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

const unsigned long long APFileStoreMaximumImageBytes = 20ULL * 1024 * 1024;

static NSString * apSHA256Hex(NSData * data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString * hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [hex appendFormat:@"%02x", digest[i]];
    return hex;
}

/// The file extension for a media type: the type's preferred one, else a safe fallback so a
/// reference with an unknown type still names a file.
static NSString * apExtensionForMediaType(NSString * mediaType) {
    UTType * type = [UTType typeWithMIMEType:mediaType];
    return type.preferredFilenameExtension ?: @"bin";
}

@implementation APFileStore

- (instancetype)initWithDirectoryURL:(NSURL *)directoryURL {
    if ((self = [super init])) {
        _directoryURL = [directoryURL copy];
    }
    return self;
}

- (BOOL)ensureDirectoryWithError:(NSError **)error {
    return [NSFileManager.defaultManager createDirectoryAtURL:_directoryURL
                                  withIntermediateDirectories:YES attributes:nil error:error];
}

- (NSURL *)fileURLForIdentifier:(NSString *)identifier mediaType:(NSString *)mediaType {
    NSString * name = [identifier stringByAppendingPathExtension:apExtensionForMediaType(mediaType)];
    return [_directoryURL URLByAppendingPathComponent:name isDirectory:NO];
}

#pragma mark - Images

- (APImage *)storeImageData:(NSData *)data originalName:(NSString *)originalName error:(NSError **)error {
    if (data.length == 0) {
        if (error) *error = APMakeError(APErrorFileUnreadable, @"the image is empty");
        return nil;
    }
    if (data.length > APFileStoreMaximumImageBytes) {
        if (error) *error = APMakeError(APErrorFileTooLarge, [NSString stringWithFormat:
            @"%@ is %@ — images are limited to %@",
            originalName ?: @"the image",
            [NSByteCountFormatter stringFromByteCount:(long long)data.length
                                           countStyle:NSByteCountFormatterCountStyleFile],
            [NSByteCountFormatter stringFromByteCount:(long long)APFileStoreMaximumImageBytes
                                           countStyle:NSByteCountFormatterCountStyleFile]]);
        return nil;
    }
    NSString * mediaType = nil;
    CGSize pixelSize = CGSizeZero;
    if (!APImageSniff(data, &mediaType, &pixelSize)) {
        if (error) *error = APMakeError(APErrorFileUnreadable, [NSString stringWithFormat:
            @"%@ is not an image this system can decode", originalName ?: @"the file"]);
        return nil;
    }

    NSString * identifier = apSHA256Hex(data);
    NSURL * url = [self fileURLForIdentifier:identifier mediaType:mediaType];
    @synchronized(self) {
        if (![self ensureDirectoryWithError:error]) return nil;
        if (![NSFileManager.defaultManager fileExistsAtPath:url.path]) {
            // Atomic: a crash mid-write leaves no half file under a name that claims a hash.
            if (![data writeToURL:url options:NSDataWritingAtomic error:error]) return nil;
        }
    }
    return [APImage imageWithIdentifier:identifier mediaType:mediaType
                              byteCount:data.length pixelSize:pixelSize
                           originalName:originalName fileURL:url];
}

- (APImage *)stageImageAtURL:(NSURL *)fileURL error:(NSError **)error {
    NSNumber * size = nil;
    [fileURL getResourceValue:&size forKey:NSURLFileSizeKey error:NULL];
    if (size.unsignedLongLongValue > APFileStoreMaximumImageBytes) {
        // Checked before reading: a 2 GB file is refused without being loaded.
        if (error) *error = APMakeError(APErrorFileTooLarge, [NSString stringWithFormat:
            @"%@ is %@ — images are limited to %@", fileURL.lastPathComponent,
            [NSByteCountFormatter stringFromByteCount:(long long)size.unsignedLongLongValue
                                           countStyle:NSByteCountFormatterCountStyleFile],
            [NSByteCountFormatter stringFromByteCount:(long long)APFileStoreMaximumImageBytes
                                           countStyle:NSByteCountFormatterCountStyleFile]]);
        return nil;
    }
    NSData * data = [NSData dataWithContentsOfURL:fileURL options:0 error:error];
    if (!data) return nil;
    return [self storeImageData:data originalName:fileURL.lastPathComponent error:error];
}

- (NSURL *)URLForIdentifier:(NSString *)identifier mediaType:(NSString *)mediaType {
    if (identifier.length == 0 || mediaType.length == 0) return nil;
    NSURL * url = [self fileURLForIdentifier:identifier mediaType:mediaType];
    return [NSFileManager.defaultManager fileExistsAtPath:url.path] ? url : nil;
}

- (NSURL *)URLForImage:(APImage *)image {
    return [self URLForIdentifier:image.identifier mediaType:image.mediaType];
}

- (NSData *)dataForImage:(APImage *)image error:(NSError **)error {
    NSURL * url = [self URLForImage:image];
    if (!url) {
        if (error) *error = APMakeError(APErrorFileMissing, [NSString stringWithFormat:
            @"%@ is not in the file store", image.originalName ?: image.identifier]);
        return nil;
    }
    return [NSData dataWithContentsOfURL:url options:0 error:error];
}

- (APImage *)resolveImage:(APImage *)image {
    return [image imageWithFileURL:[self URLForImage:image]];
}

- (BOOL)removeImage:(APImage *)image error:(NSError **)error {
    NSURL * url = [self fileURLForIdentifier:image.identifier mediaType:image.mediaType];
    @synchronized(self) {
        if (![NSFileManager.defaultManager fileExistsAtPath:url.path]) return YES;
        return [NSFileManager.defaultManager removeItemAtURL:url error:error];
    }
}

@end
