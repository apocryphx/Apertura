//  APFileStore — the sidecar store for bytes that do not belong in a transcript.
//
//  A transcript is a JSON document (APTranscript) that rides Core Data and CloudKit; an
//  image is megabytes. So the transcript carries a REFERENCE (APImage) and the bytes live
//  here, in one flat directory, content-addressed: the identifier is the SHA-256 of the
//  bytes and the file is `<identifier>.<extension>` for the sniffed media type. Storing the
//  same bytes twice is a no-op, two conversations attaching the same picture share one
//  file, and a reference can always be checked against what it names.
//
//  The app passes …/Application Support/Apertura/Files; tests use a temporary directory.
#import <Foundation/Foundation.h>
#import <AperturaKit/APContent.h>

NS_ASSUME_NONNULL_BEGIN

/// Largest image the store accepts (20 MB). Bigger ones fail with APErrorFileTooLarge —
/// this is a chat attachment, not an asset pipeline.
FOUNDATION_EXPORT const unsigned long long APFileStoreMaximumImageBytes;

@interface APFileStore : NSObject

/// The directory is created on first use. One store per directory; thread-safe.
- (instancetype)initWithDirectoryURL:(NSURL *)directoryURL NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@property (readonly) NSURL *directoryURL;

#pragma mark - Images

/// Store image bytes. The data is sniffed (ImageIO): anything that is not a decodable
/// image fails with APErrorFileUnreadable, anything over APFileStoreMaximumImageBytes
/// with APErrorFileTooLarge. Returns the reference, resolved to its file.
- (nullable APImage *)storeImageData:(NSData *)data
                        originalName:(nullable NSString *)originalName
                               error:(NSError **)error;

/// Read a file and store it as an image (its last path component is the original name).
- (nullable APImage *)stageImageAtURL:(NSURL *)fileURL error:(NSError **)error;

/// Where `image`'s bytes would be in this store; nil when the file is not there.
- (nullable NSURL *)URLForImage:(APImage *)image;
/// Where bytes with this identifier and media type would be; nil when not there.
- (nullable NSURL *)URLForIdentifier:(NSString *)identifier mediaType:(NSString *)mediaType;

/// The bytes of `image` from this store, or nil with APErrorFileMissing.
- (nullable NSData *)dataForImage:(APImage *)image error:(NSError **)error;

/// `image` resolved against this store: the same reference with `fileURL` set when the
/// file is present, nil `fileURL` when it is not.
- (APImage *)resolveImage:(APImage *)image;

/// Delete the bytes of `image`. YES when the file is gone afterwards (missing counts).
- (BOOL)removeImage:(APImage *)image error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
