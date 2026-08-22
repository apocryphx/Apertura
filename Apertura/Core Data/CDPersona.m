//
//  CDPersona.m
//  Apertura
//

#import "CDPersona.h"
#import <CommonCrypto/CommonDigest.h>

NSString * const CDPersonaSectionSeparator = @"\n\n---\n\n";

static NSString * const kEntityName = @"CDPersona";

static NSString * apPersonaSHA256Hex(NSString * text) {
    NSData * utf8 = [text dataUsingEncoding:NSUTF8StringEncoding];
    if (utf8.length == 0) return nil;
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(utf8.bytes, (CC_LONG)utf8.length, digest);
    NSMutableString * hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [hex appendFormat:@"%02x", digest[i]];
    return hex;
}

@implementation CDPersona

#pragma mark - Lifecycle

+ (instancetype)insertInContext:(NSManagedObjectContext *)context {
    return [NSEntityDescription insertNewObjectForEntityForName:kEntityName
                                         inManagedObjectContext:context];
}

+ (NSFetchRequest<CDPersona *> *)currentPersonasFetchRequest {
    NSFetchRequest<CDPersona *> * request = [self fetchRequest];
    request.predicate = [NSPredicate predicateWithFormat:@"nextVersion == nil"];
    request.sortDescriptors = @[ [NSSortDescriptor sortDescriptorWithKey:@"name" ascending:YES
                                    selector:@selector(localizedStandardCompare:)] ];
    return request;
}

+ (nullable instancetype)personaWithIdentifier:(NSUUID *)identifier
                                     inContext:(NSManagedObjectContext *)context
                                         error:(NSError **)error {
    NSFetchRequest<CDPersona *> * request = [self fetchRequest];
    request.predicate = [NSPredicate predicateWithFormat:@"identifier == %@", identifier];
    request.fetchLimit = 1;
    return [context executeFetchRequest:request error:error].firstObject;
}

+ (nullable instancetype)importFromFileURLs:(NSArray<NSURL *> *)fileURLs
                                       name:(NSString *)name
                                  inContext:(NSManagedObjectContext *)context
                                      error:(NSError **)error {
    NSMutableArray<NSString *> * sections = [NSMutableArray arrayWithCapacity:fileURLs.count];
    for (NSURL * url in fileURLs) {
        NSString * text = [NSString stringWithContentsOfURL:url
                                                   encoding:NSUTF8StringEncoding error:error];
        if (!text) return nil;
        [sections addObject:text];
    }
    CDPersona * persona = [self insertInContext:context];
    persona.name = name;
    [persona updateBody:[sections componentsJoinedByString:CDPersonaSectionSeparator]];
    return persona;
}

- (void)awakeFromInsert {
    [super awakeFromInsert];
    NSDate * now = [NSDate date];
    [self setPrimitiveValue:[NSUUID UUID] forKey:@"identifier"];
    [self setPrimitiveValue:now forKey:@"dateCreated"];
    [self setPrimitiveValue:now forKey:@"dateModified"];
}

#pragma mark - Editing

- (CDPersona *)snapshotBeforeEditWithNote:(NSString *)note author:(NSString *)author {
    CDPersona * frozen = [CDPersona insertInContext:self.managedObjectContext];
    frozen.name = self.name;
    frozen.body = self.body;
    frozen.sha256 = self.sha256;
    // The frozen row keeps the timestamps of the state it preserves; its `notes` say why
    // the edit that displaced it happened.
    frozen.dateCreated = self.dateCreated;
    frozen.dateModified = self.dateModified;
    if (note.length || author.length)
        frozen.notes = author.length ? [NSString stringWithFormat:@"%@: %@", author, note ?: @""]
                                     : note;
    // Splice behind the head: head -> frozen -> (old chain).
    frozen.previousVersion = self.previousVersion;
    self.previousVersion = frozen;
    return frozen;
}

- (void)updateBody:(NSString *)body {
    self.body = body;
    self.sha256 = apPersonaSHA256Hex(body);
    self.dateModified = [NSDate date];
}

- (NSArray<CDPersona *> *)versionChain {
    NSMutableArray<CDPersona *> * chain = [NSMutableArray array];
    for (CDPersona * v = self; v; v = v.previousVersion) {
        // A cycle would mean a corrupted splice; refuse to loop forever.
        if ([chain containsObject:v]) break;
        [chain addObject:v];
    }
    return chain;
}

- (BOOL)isCurrentVersion { return self.nextVersion == nil; }

@end
