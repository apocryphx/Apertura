//  APFileStoreTests — the sidecar store: content addressing, sniffing, the size cap, the
//  image reference round trip through a message and the v2 transcript (with and without
//  a store), and what the Gemini backend builds from an image turn (no network).
#import <XCTest/XCTest.h>
#import <AperturaKit/AperturaKit.h>
#import <ImageIO/ImageIO.h>
#import "APGoogleSessionInternal.h"
#import "APImageTranscoding.h"

@interface APFileStoreTests : XCTestCase
@end

@implementation APFileStoreTests {
    NSURL * _dir;
    APFileStore * _store;
}

- (void)setUp {
    _dir = [NSFileManager.defaultManager.temporaryDirectory
            URLByAppendingPathComponent:[NSString stringWithFormat:@"APFileStoreTests-%@", NSUUID.UUID.UUIDString]
                            isDirectory:YES];
    _store = [[APFileStore alloc] initWithDirectoryURL:_dir];
}

- (void)tearDown {
    [NSFileManager.defaultManager removeItemAtURL:_dir error:NULL];
}

#pragma mark - Fixtures

/// A solid PNG of the given size, drawn with CoreGraphics (no asset needed).
static NSData * apPNG(size_t width, size_t height) {
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(NULL, width, height, 8, 0, space, (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
    CGContextSetRGBFillColor(ctx, 0.2, 0.4, 0.8, 1);
    CGContextFillRect(ctx, CGRectMake(0, 0, width, height));
    CGContextSetRGBFillColor(ctx, 1, 1, 1, 1);
    CGContextFillRect(ctx, CGRectMake(width / 4, height / 4, width / 2, height / 2));
    CGImageRef image = CGBitmapContextCreateImage(ctx);
    NSMutableData * out = [NSMutableData data];
    CGImageDestinationRef dest = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)out,
                                                                  CFSTR("public.png"), 1, NULL);
    CGImageDestinationAddImage(dest, image, NULL);
    CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(image);
    CGContextRelease(ctx);
    CGColorSpaceRelease(space);
    return out;
}

static NSString * apSHA256(NSData * data) {
    // Independent of the store's own hashing: shell out to the system digest.
    NSTask * task = [[NSTask alloc] init];
    task.executableURL = [NSURL fileURLWithPath:@"/usr/bin/shasum"];
    task.arguments = @[ @"-a", @"256" ];
    NSPipe * in = NSPipe.pipe, * out = NSPipe.pipe;
    task.standardInput = in; task.standardOutput = out;
    [task launch];
    [in.fileHandleForWriting writeData:data];
    [in.fileHandleForWriting closeFile];
    NSData * result = [out.fileHandleForReading readDataToEndOfFile];
    [task waitUntilExit];
    NSString * line = [[NSString alloc] initWithData:result encoding:NSUTF8StringEncoding];
    return [line componentsSeparatedByString:@" "].firstObject;
}

#pragma mark - Store

- (void)testStoreIsContentAddressedAndIdempotent {
    NSData * png = apPNG(64, 48);
    NSError * error = nil;
    APImage * a = [_store storeImageData:png originalName:@"a.png" error:&error];
    XCTAssertNotNil(a, @"%@", error);
    XCTAssertEqualObjects(a.identifier, apSHA256(png));
    XCTAssertEqualObjects(a.mediaType, @"image/png");
    XCTAssertEqual(a.byteCount, png.length);
    XCTAssertEqual(a.pixelSize.width, 64);
    XCTAssertEqual(a.pixelSize.height, 48);
    XCTAssertEqualObjects(a.originalName, @"a.png");
    XCTAssertEqualObjects(a.fileURL.lastPathComponent, [a.identifier stringByAppendingString:@".png"]);
    XCTAssertEqualObjects([NSData dataWithContentsOfURL:a.fileURL], png);

    // Same bytes under another name: same file, no second copy, the new name on the reference.
    NSDate * mtime = [NSFileManager.defaultManager attributesOfItemAtPath:a.fileURL.path error:NULL][NSFileModificationDate];
    APImage * b = [_store storeImageData:png originalName:@"b.png" error:&error];
    XCTAssertEqualObjects(b.identifier, a.identifier);
    XCTAssertEqualObjects(b.fileURL, a.fileURL);
    XCTAssertEqualObjects(b.originalName, @"b.png");
    XCTAssertEqualObjects([NSFileManager.defaultManager attributesOfItemAtPath:a.fileURL.path error:NULL][NSFileModificationDate], mtime);
    NSArray * files = [NSFileManager.defaultManager contentsOfDirectoryAtPath:_dir.path error:NULL];
    XCTAssertEqual(files.count, 1u);

    // Resolution and data.
    XCTAssertEqualObjects([_store URLForImage:a], a.fileURL);
    XCTAssertEqualObjects([_store URLForIdentifier:a.identifier mediaType:@"image/png"], a.fileURL);
    XCTAssertEqualObjects([_store dataForImage:a error:NULL], png);
    XCTAssertTrue([_store removeImage:a error:&error], @"%@", error);
    XCTAssertNil([_store URLForImage:a]);
    XCTAssertNil([_store dataForImage:a error:&error]);
    XCTAssertEqual(error.code, APErrorFileMissing);
    XCTAssertTrue([_store removeImage:a error:NULL], @"removing a missing file is fine");
    XCTAssertNil([_store resolveImage:a].fileURL);
}

- (void)testStageReadsSniffsAndNamesFromTheURL {
    NSData * png = apPNG(10, 10);
    NSURL * src = [NSFileManager.defaultManager.temporaryDirectory
                   URLByAppendingPathComponent:[NSString stringWithFormat:@"%@.image", NSUUID.UUID.UUIDString]];
    XCTAssertTrue([png writeToURL:src atomically:YES]);   // deliberately not named .png: sniffed, not trusted
    NSError * error = nil;
    APImage * image = [_store stageImageAtURL:src error:&error];
    XCTAssertNotNil(image, @"%@", error);
    XCTAssertEqualObjects(image.mediaType, @"image/png");
    XCTAssertEqualObjects(image.originalName, src.lastPathComponent);
    XCTAssertTrue([image.fileURL.path hasSuffix:@".png"]);
    [NSFileManager.defaultManager removeItemAtURL:src error:NULL];
}

- (void)testStoreRefusesNonImagesEmptyAndOversized {
    NSError * error = nil;
    XCTAssertNil([_store storeImageData:[@"hello" dataUsingEncoding:NSUTF8StringEncoding]
                           originalName:@"notes.txt" error:&error]);
    XCTAssertEqualObjects(error.domain, APErrorDomain);
    XCTAssertEqual(error.code, APErrorFileUnreadable);
    XCTAssertNil([_store storeImageData:[NSData data] originalName:nil error:&error]);
    XCTAssertEqual(error.code, APErrorFileUnreadable);

    // Over the cap: refused by size BEFORE sniffing (the bytes are not even an image).
    NSMutableData * big = [NSMutableData dataWithLength:(NSUInteger)APFileStoreMaximumImageBytes + 1];
    XCTAssertNil([_store storeImageData:big originalName:@"huge.png" error:&error]);
    XCTAssertEqual(error.code, APErrorFileTooLarge);
    XCTAssertTrue([error.localizedDescription containsString:@"huge.png"]);
    // …and at the URL entry too, without reading the file.
    NSURL * src = [NSFileManager.defaultManager.temporaryDirectory
                   URLByAppendingPathComponent:[NSString stringWithFormat:@"%@.png", NSUUID.UUID.UUIDString]];
    XCTAssertTrue([big writeToURL:src atomically:YES]);
    XCTAssertNil([_store stageImageAtURL:src error:&error]);
    XCTAssertEqual(error.code, APErrorFileTooLarge);
    [NSFileManager.defaultManager removeItemAtURL:src error:NULL];
    XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:_dir.path], @"nothing was stored");
}

#pragma mark - Message + transcript

- (void)testImageContentIsNotTextAndRoundTripsThroughTheTranscript {
    APImage * image = [_store storeImageData:apPNG(32, 16) originalName:@"tiny.png" error:NULL];
    APMessage * user = [APMessage messageWithRole:APRoleUser
                                          content:@[ [APContent imageContent:image],
                                                     [APContent textContent:@"what is this?"] ]];
    XCTAssertNil(user.content[0].promptText, @"nothing textual reaches the model for an image");
    XCTAssertEqualObjects(user.textRepresentation, @"what is this?");
    XCTAssertEqualObjects(user.displayText, @"what is this?");
    XCTAssertEqual(user.images.count, 1u);
    XCTAssertEqual(user.images.firstObject, image);
    XCTAssertEqual(user.attachments.count, 0u);

    NSData * data = [APTranscript dataFromMessages:@[ user ]];
    NSDictionary * doc = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
    NSDictionary * part = doc[@"messages"][0][@"content"][0];
    XCTAssertEqualObjects(part[@"kind"], @"image");
    XCTAssertEqualObjects(part[@"id"], image.identifier);
    XCTAssertEqualObjects(part[@"type"], @"image/png");
    XCTAssertEqualObjects(part[@"size"], @(image.byteCount));
    XCTAssertEqualObjects(part[@"width"], @32);
    XCTAssertEqualObjects(part[@"height"], @16);
    XCTAssertEqualObjects(part[@"name"], @"tiny.png");
    XCTAssertNil(part[@"data"], @"the reference only — never the bytes");
    XCTAssertNil(part[@"text"]);
    XCTAssertFalse([[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding]
                    containsString:[image.fileURL.path stringByDeletingLastPathComponent]],
                   @"no machine-specific path in the document");

    // With the store: the file URL comes back. Without: a placeholder reference, kept.
    int16_t version = 0;
    APMessage * withStore = [APTranscript messagesFromData:data version:&version fileStore:_store].firstObject;
    XCTAssertEqual(version, 2);
    APImage * back = withStore.images.firstObject;
    XCTAssertEqualObjects(back.identifier, image.identifier);
    XCTAssertEqualObjects(back.fileURL, image.fileURL);
    XCTAssertEqual(back.pixelSize.width, 32);
    XCTAssertEqual(back.pixelSize.height, 16);
    XCTAssertEqualObjects(back.originalName, @"tiny.png");
    XCTAssertEqualObjects([APTranscript dataFromMessages:@[ withStore ]], data, @"deterministic bytes");

    APMessage * noStore = [APTranscript messagesFromData:data version:&version].firstObject;
    XCTAssertEqual(noStore.content.count, 2u, @"the image part is kept, not dropped");
    XCTAssertEqual(noStore.content[0].kind, APContentKindImage);
    XCTAssertNil(noStore.images.firstObject.fileURL);
    XCTAssertEqualObjects(noStore.images.firstObject.identifier, image.identifier);
    XCTAssertEqualObjects([APTranscript dataFromMessages:@[ noStore ]], data, @"re-saving loses nothing");

    // Missing file: the reference survives with no URL.
    [_store removeImage:image error:NULL];
    APMessage * missing = [APTranscript messagesFromData:data version:&version fileStore:_store].firstObject;
    XCTAssertEqual(missing.images.count, 1u);
    XCTAssertNil(missing.images.firstObject.fileURL);
    XCTAssertEqualObjects([APTranscript dataFromMessages:@[ missing ]], data);
}

- (void)testMalformedImagePartsAreSkipped {
    NSString * doc = @"{\"version\":2,\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"kind\":\"image\",\"type\":\"image/png\"},"          // no id
        "{\"kind\":\"image\",\"id\":\"abc\"},"                  // no type
        "{\"kind\":\"image\",\"id\":\"abc\",\"type\":\"image/png\"},"   // minimal: kept
        "{\"kind\":\"text\",\"text\":\"t\"}]}]}";
    int16_t v = 0;
    APMessage * m = [APTranscript messagesFromData:[doc dataUsingEncoding:NSUTF8StringEncoding]
                                           version:&v fileStore:_store].firstObject;
    XCTAssertEqual(m.content.count, 2u);
    XCTAssertEqualObjects(m.images.firstObject.identifier, @"abc");
    XCTAssertEqual(m.images.firstObject.pixelSize.width, 0);
    XCTAssertNil(m.images.firstObject.originalName);
}

#pragma mark - Backends

- (void)testLocalValidationRefusesImagesAndGoogleAccepts {
    APImage * image = [_store storeImageData:apPNG(8, 8) originalName:@"p.png" error:NULL];
    APMessage * turn = [APMessage messageWithRole:APRoleUser
                                          content:@[ [APContent imageContent:image],
                                                     [APContent textContent:@"?"] ]];
    // The base rule, as the local backend inherits it (no model needed to check it).
    APSession * base = [[APSession alloc] init];
    XCTAssertFalse(base.acceptsImageContent);
    NSError * e = [base validationErrorForRespondMessage:turn];
    XCTAssertEqualObjects(e.domain, APErrorDomain);
    XCTAssertEqual(e.code, APErrorUnsupportedContent);
    XCTAssertTrue([e.localizedDescription containsString:@"text-only"], @"%@", e);
    XCTAssertEqual([base validationErrorForPrimeMessages:@[ turn ]].code, APErrorUnsupportedContent);

    APGoogleSession * google = [[APGoogleSession alloc] initWithModelName:@"gemma-4-31b-it"
                                                           apiKeyProvider:^NSString * { return @"k"; }];
    XCTAssertTrue(google.acceptsImageContent);
    XCTAssertNil([google validationErrorForRespondMessage:turn]);
    XCTAssertNil([google validationErrorForPrimeMessages:@[ turn ]]);
}

- (void)testGoogleUserPartsPutInlineDataBeforeText {
    NSData * png = apPNG(20, 12);
    APImage * image = [_store storeImageData:png originalName:@"p.png" error:NULL];
    APMessage * turn = [APMessage messageWithRole:APRoleUser
                                          content:@[ [APContent imageContent:image],
                                                     [APContent attachmentContent:
                                                      [APAttachment attachmentWithName:@"n.md" text:@"body" sizeBytes:4
                                                                 contentTypeIdentifier:nil]],
                                                     [APContent textContent:@"describe"] ]];
    APGoogleSession * google = [[APGoogleSession alloc] initWithModelName:@"gemma-4-31b-it"
                                                           apiKeyProvider:^NSString * { return @"k"; }];
    NSError * error = nil;
    NSArray<NSDictionary *> * parts = [google partsForMessage:turn placeholderForMissing:NO error:&error];
    XCTAssertEqual(parts.count, 2u, @"%@", error);
    XCTAssertEqualObjects(parts[0][@"inlineData"][@"mimeType"], @"image/png");
    XCTAssertEqualObjects([[NSData alloc] initWithBase64EncodedString:parts[0][@"inlineData"][@"data"] options:0], png,
                          @"small PNG: the stored bytes verbatim");
    XCTAssertEqualObjects(parts[1][@"text"], turn.textRepresentation);
    XCTAssertTrue([parts[1][@"text"] hasPrefix:@"--- attached file: n.md ---"]);

    // The request body carries the same parts.
    NSURLRequest * request = [google requestWithContents:@[ @{ @"role" : @"user", @"parts" : parts } ]
                                                   tools:@[] options:[APGenerationOptions defaultOptions] error:&error];
    XCTAssertNotNil(request, @"%@", error);
    NSDictionary * body = [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
    XCTAssertEqualObjects(body[@"contents"][0][@"parts"][0][@"inlineData"][@"mimeType"], @"image/png");
    XCTAssertEqualObjects(request.allHTTPHeaderFields[@"x-goog-api-key"], @"k");

    // A live turn whose file is gone fails; replayed history gets a placeholder.
    [_store removeImage:image error:NULL];
    XCTAssertNil([google partsForMessage:turn placeholderForMissing:NO error:&error]);
    XCTAssertEqual(error.code, APErrorFileMissing);
    NSArray * replay = [google partsForMessage:turn placeholderForMissing:YES error:NULL];
    XCTAssertEqual(replay.count, 2u);
    XCTAssertTrue([replay[0][@"text"] containsString:@"p.png"]);
    XCTAssertTrue([replay[0][@"text"] containsString:@"no longer available"]);
}

- (void)testGoogleDownscalesACopyForTheRequestAndKeepsTheOriginal {
    // A PNG wider than the edge cap. Transport gets a scaled copy; the store is untouched.
    NSData * png = apPNG(3000, 600);
    APImage * image = [_store storeImageData:png originalName:@"wide.png" error:NULL];
    XCTAssertEqual(image.pixelSize.width, 3000);
    APGoogleSession * google = [[APGoogleSession alloc] initWithModelName:@"m"
                                                           apiKeyProvider:^NSString * { return @"k"; }];
    NSArray * parts = [google partsForMessage:[APMessage messageWithRole:APRoleUser content:@[ [APContent imageContent:image] ]]
                        placeholderForMissing:NO error:NULL];
    XCTAssertEqual(parts.count, 1u, @"an image-only turn is a valid turn: no empty text part");
    NSData * sent = [[NSData alloc] initWithBase64EncodedString:parts[0][@"inlineData"][@"data"] options:0];
    NSString * type = nil; CGSize size = CGSizeZero;
    XCTAssertTrue(APImageSniff(sent, &type, &size));
    XCTAssertEqualObjects(type, @"image/png");
    XCTAssertEqual(size.width, APGoogleSessionMaximumImageEdge);
    XCTAssertEqual(size.height, 410);   // 600 * 2048 / 3000, rounded by ImageIO
    XCTAssertEqualObjects([NSData dataWithContentsOfURL:image.fileURL], png, @"original untouched");

    // A type the API does not take (TIFF) is transcoded to PNG even when small.
    NSString * outType = nil;
    NSData * tiff = ({
        CGImageSourceRef s = CGImageSourceCreateWithData((__bridge CFDataRef)apPNG(5, 5), NULL);
        CGImageRef cg = CGImageSourceCreateImageAtIndex(s, 0, NULL);
        NSMutableData * out = [NSMutableData data];
        CGImageDestinationRef d = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)out, CFSTR("public.tiff"), 1, NULL);
        CGImageDestinationAddImage(d, cg, NULL);
        CGImageDestinationFinalize(d);
        CFRelease(d); CGImageRelease(cg); CFRelease(s);
        out;
    });
    NSData * converted = APImageDataForTransport(tiff, @"image/tiff", 2048,
                                                 APGoogleSessionAcceptedImageMediaTypes(), &outType);
    XCTAssertEqualObjects(outType, @"image/png");
    XCTAssertTrue(APImageSniff(converted, &type, NULL));
    XCTAssertEqualObjects(type, @"image/png");
}

- (void)testGooglePrimeReplaysImagesAsInlineData {
    NSData * png = apPNG(6, 6);
    APImage * image = [_store storeImageData:png originalName:@"p.png" error:NULL];
    APGoogleSession * google = [[APGoogleSession alloc] initWithModelName:@"m"
                                                           apiKeyProvider:^NSString * { return @"k"; }];
    google.callbackQueue = dispatch_queue_create("test.prime", DISPATCH_QUEUE_SERIAL);
    XCTestExpectation * primed = [self expectationWithDescription:@"primed"];
    [google primeWithMessages:@[ [APMessage systemMessageWithText:@"persona"],
                                 [APMessage messageWithRole:APRoleUser content:@[ [APContent imageContent:image],
                                                                                  [APContent textContent:@"look"] ]],
                                 [APMessage assistantMessageWithText:@"a square"] ]
                   completion:^(NSError * error) { XCTAssertNil(error); [primed fulfill]; }];
    [self waitForExpectations:@[ primed ] timeout:5];
    NSArray<NSDictionary *> * contents = google.contentsForTesting;
    XCTAssertEqual(contents.count, 2u);
    XCTAssertEqualObjects(contents[0][@"role"], @"user");
    XCTAssertEqualObjects(contents[0][@"parts"][0][@"inlineData"][@"mimeType"], @"image/png");
    XCTAssertEqualObjects(contents[0][@"parts"][1][@"text"], @"look");
    XCTAssertEqualObjects(contents[1][@"parts"][0][@"text"], @"a square");
    XCTAssertEqual(google.transcript.count, 3u);
}

@end
