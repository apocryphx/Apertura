//
//  CDChatSession.m
//  Apertura
//
//  Created by Kolja Wawrowsky on 8/6/26.
//

#import "CDChatSession.h"
#import <AperturaKit/AperturaKit.h>
#import <CommonCrypto/CommonDigest.h>

NSString * const CDChatSessionBackendLocal  = @"local";
NSString * const CDChatSessionBackendGoogle = @"google";

const int16_t CDChatSessionCurrentSchemaVersion = 1;

static NSString * const kEntityName = @"CDChatSession";

// Attribute names, used with the primitive accessors below.
static NSString * const kAttrIdentifier     = @"identifier";
static NSString * const kAttrDateCreated    = @"dateCreated";
static NSString * const kAttrDateModified   = @"dateModified";
static NSString * const kAttrTranscriptJSON = @"transcriptJSON";

// Document keys.
static NSString * const kKeyVersion  = @"version";
static NSString * const kKeyMessages = @"messages";
static NSString * const kKeyRole     = @"role";
static NSString * const kKeyContent  = @"content";
static NSString * const kKeyKind     = @"kind";
static NSString * const kKeyText     = @"text";

// Role and kind names. These strings are the wire format — never renumber, never rename.
static NSString * const kRoleSystem    = @"system";
static NSString * const kRoleUser      = @"user";
static NSString * const kRoleAssistant = @"assistant";
static NSString * const kRoleTool      = @"tool";
static NSString * const kKindText      = @"text";
static NSString * const kKindImage     = @"image";
static NSString * const kKindAudio     = @"audio";

/// Longest `title` derived from a first user turn, in composed characters.
static const NSUInteger kDerivedTitleLength = 60;

#pragma mark - Enum <-> name

static NSString * apRoleName(APRole role) {
    switch (role) {
        case APRoleSystem:    return kRoleSystem;
        case APRoleUser:      return kRoleUser;
        case APRoleAssistant: return kRoleAssistant;
        case APRoleTool:      return kRoleTool;
    }
    return nil;   // a case added without updating this switch: refuse to guess
}

static BOOL apRoleFromName(NSString * name, APRole * outRole) {
    if      ([name isEqualToString:kRoleSystem])    { *outRole = APRoleSystem;    return YES; }
    else if ([name isEqualToString:kRoleUser])      { *outRole = APRoleUser;      return YES; }
    else if ([name isEqualToString:kRoleAssistant]) { *outRole = APRoleAssistant; return YES; }
    else if ([name isEqualToString:kRoleTool])      { *outRole = APRoleTool;      return YES; }
    return NO;
}

static NSString * apContentKindName(APContentKind kind) {
    switch (kind) {
        case APContentKindText:  return kKindText;
        case APContentKindImage: return kKindImage;
        case APContentKindAudio: return kKindAudio;
    }
    return nil;
}

static NSString * apSHA256Hex(NSData * data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString * hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [hex appendFormat:@"%02x", digest[i]];
    return hex;
}

@implementation CDChatSession {
    NSArray<APMessage *> * _decodedMessages;       // valid only while _messagesDecoded
    int16_t                _decodedVersion;        // the WRITER's envelope version
    BOOL                   _messagesDecoded;
    BOOL                   _encodingFromMessages;  // -setMessages: is driving -setTranscriptJSON:
}

#pragma mark - Lifecycle

+ (instancetype)insertInContext:(NSManagedObjectContext *)context {
    return [NSEntityDescription insertNewObjectForEntityForName:kEntityName
                                         inManagedObjectContext:context];
}

+ (NSFetchRequest<CDChatSession *> *)recentSessionsFetchRequest {
    NSFetchRequest<CDChatSession *> * request = [self fetchRequest];
    request.sortDescriptors = @[ [NSSortDescriptor sortDescriptorWithKey:kAttrDateModified
                                                              ascending:NO] ];
    return request;
}

+ (nullable instancetype)sessionWithIdentifier:(NSUUID *)identifier
                                     inContext:(NSManagedObjectContext *)context
                                         error:(NSError **)error {
    NSFetchRequest<CDChatSession *> * request = [self fetchRequest];
    request.predicate = [NSPredicate predicateWithFormat:@"%K == %@", kAttrIdentifier, identifier];
    request.fetchLimit = 1;
    return [context executeFetchRequest:request error:error].firstObject;
}

- (void)awakeFromInsert {
    [super awakeFromInsert];
    // Primitive accessors: these are baseline values, not undoable edits (see
    // -awakeFromInsert's documentation).
    NSDate * now = [NSDate date];
    [self setPrimitiveValue:[NSUUID UUID] forKey:kAttrIdentifier];
    [self setPrimitiveValue:now forKey:kAttrDateCreated];
    [self setPrimitiveValue:now forKey:kAttrDateModified];
}

- (void)willSave {
    if (!self.isDeleted) {
        // Touch the timestamp through the KVO setter so Core Data records it as a change
        // and actually writes the column. That re-enters -willSave, so the delta guard is
        // load-bearing: on the second pass the existing value is already within a second
        // of now, nothing changes, and the recursion stops.
        NSDate * now = [NSDate date];
        NSDate * previous = self.dateModified;
        if (!previous || fabs([now timeIntervalSinceDate:previous]) > 1.0) {
            self.dateModified = now;
        }
    }
    [super willSave];
}

- (void)awakeFromFetch                                    { [super awakeFromFetch];              [self invalidateMessageCache]; }
- (void)didTurnIntoFault                                  { [super didTurnIntoFault];            [self invalidateMessageCache]; }
- (void)awakeFromSnapshotEvents:(NSSnapshotEventType)flags { [super awakeFromSnapshotEvents:flags]; [self invalidateMessageCache]; }

- (void)invalidateMessageCache {
    _decodedMessages = nil;
    _decodedVersion  = 0;
    _messagesDecoded = NO;
}

#pragma mark - The chat

// Custom accessor for a modeled attribute: same contract as the dynamic one, plus it
// drops the decoded cache. Anything that writes transcriptJSON directly goes through
// here, so `messages` can never answer from a stale decode.
- (void)setTranscriptJSON:(NSData *)transcriptJSON {
    [self willChangeValueForKey:kAttrTranscriptJSON];
    [self setPrimitiveValue:transcriptJSON forKey:kAttrTranscriptJSON];
    [self didChangeValueForKey:kAttrTranscriptJSON];
    [self invalidateMessageCache];

    if (_encodingFromMessages) return;   // -setMessages: sets the columns itself, exactly
    // A document written straight to the column — an import, a repair, a hand-edit. Decode
    // it now so the denormalized columns cannot drift from what is really stored, and keep
    // the WRITER's version rather than claiming this build produced it.
    self.messageCount = (int64_t)self.messages.count;
    self.transcriptSchemaVersion = _decodedVersion;
}

- (NSArray<APMessage *> *)messages {
    if (!_messagesDecoded) {
        int16_t version = 0;
        _decodedMessages = [self decodeTranscriptVersion:&version] ?: @[];
        _decodedVersion  = version;
        _messagesDecoded = YES;
    }
    return _decodedMessages;
}

- (void)setMessages:(NSArray<APMessage *> *)messages {
    NSArray<APMessage *> * turns = messages ? [messages copy] : @[];
    NSData * encoded = nil;
    if (turns.count > 0) {
        encoded = [CDChatSession dataFromMessages:turns];
        if (!encoded) return;   // encoding failed and said so — keep what is on disk
    }

    _encodingFromMessages = YES;
    self.transcriptJSON = encoded;          // invalidates the cache…
    _encodingFromMessages = NO;

    _decodedMessages = turns;               // …so prime it afterwards, not before
    _decodedVersion  = turns.count > 0 ? CDChatSessionCurrentSchemaVersion : 0;
    _messagesDecoded = YES;

    self.messageCount = (int64_t)turns.count;
    self.transcriptSchemaVersion = _decodedVersion;
}

- (void)appendMessage:(APMessage *)message {
    if (!message) return;
    self.messages = [self.messages arrayByAddingObject:message];
}

- (nullable NSString *)transcriptJSONString {
    NSData * data = self.transcriptJSON;
    if (data.length == 0) return nil;
    return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
}

#pragma mark - JSON codec

+ (nullable NSData *)dataFromMessages:(NSArray<APMessage *> *)messages {
    NSMutableArray<NSDictionary *> * turns = [NSMutableArray arrayWithCapacity:messages.count];
    for (APMessage * message in messages) {
        NSString * role = apRoleName(message.role);
        if (!role) {
            NSLog(@"CDChatSession: dropping a message with unknown role %ld", (long)message.role);
            continue;
        }
        NSMutableArray<NSDictionary *> * parts = [NSMutableArray arrayWithCapacity:message.content.count];
        for (APContent * content in message.content) {
            NSString * kind = apContentKindName(content.kind);
            if (!kind) continue;
            NSMutableDictionary * part = [NSMutableDictionary dictionaryWithObject:kind forKey:kKeyKind];
            if (content.text) part[kKeyText] = content.text;
            [parts addObject:part];
        }
        [turns addObject:@{ kKeyRole : role, kKeyContent : parts }];
    }

    NSDictionary * document = @{ kKeyVersion  : @(CDChatSessionCurrentSchemaVersion),
                                 kKeyMessages : turns };
    // Sorted keys: identical transcripts produce identical bytes, which keeps diffs and
    // any future content hashing honest.
    NSError * error = nil;
    NSData * data = [NSJSONSerialization dataWithJSONObject:document
                                                    options:NSJSONWritingSortedKeys
                                                      error:&error];
    if (!data) NSLog(@"CDChatSession: could not encode transcript — %@", error);
    return data;
}

/// nil when there is nothing stored or the stored bytes are not a transcript. `outVersion`
/// receives the version the document claims — 0 when there is none to read.
- (nullable NSArray<APMessage *> *)decodeTranscriptVersion:(int16_t *)outVersion {
    *outVersion = 0;
    NSData * data = self.transcriptJSON;
    if (data.length == 0) return nil;

    NSError * error = nil;
    id document = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![document isKindOfClass:NSDictionary.class]) {
        NSLog(@"CDChatSession %@: transcript is not a JSON object — %@",
              self.identifier, error ?: document);
        return nil;
    }
    id rawVersion = document[kKeyVersion];
    if ([rawVersion isKindOfClass:NSNumber.class]) *outVersion = [rawVersion shortValue];
    id rawTurns = document[kKeyMessages];
    if (![rawTurns isKindOfClass:NSArray.class]) {
        NSLog(@"CDChatSession %@: transcript has no messages array", self.identifier);
        return nil;
    }

    NSMutableArray<APMessage *> * messages = [NSMutableArray arrayWithCapacity:[rawTurns count]];
    for (id rawTurn in (NSArray *)rawTurns) {
        if (![rawTurn isKindOfClass:NSDictionary.class]) continue;
        id rawRole = rawTurn[kKeyRole];
        APRole role;
        if (![rawRole isKindOfClass:NSString.class] || !apRoleFromName(rawRole, &role)) {
            NSLog(@"CDChatSession %@: skipping a turn with unreadable role %@",
                  self.identifier, rawRole);
            continue;
        }

        NSMutableArray<APContent *> * content = [NSMutableArray array];
        id rawParts = rawTurn[kKeyContent];
        if ([rawParts isKindOfClass:NSArray.class]) {
            for (id rawPart in (NSArray *)rawParts) {
                if (![rawPart isKindOfClass:NSDictionary.class]) continue;
                id text = rawPart[kKeyText];
                // Text is all v1 can build. Image and audio parts are reserved in
                // APContent with no way to construct them, so they are skipped rather
                // than guessed at — `transcriptSchemaVersion` is how a caller learns the
                // row may hold more than it got back.
                if ([rawPart[kKeyKind] isEqual:kKindText] && [text isKindOfClass:NSString.class]) {
                    [content addObject:[APContent textContent:text]];
                }
            }
        }
        [messages addObject:[APMessage messageWithRole:role content:content]];
    }
    return messages;
}

#pragma mark - Capture

- (void)captureSession:(APSession *)session {
    [self captureSession:session model:nil];
}

- (void)captureSession:(APSession *)session model:(nullable APModel *)model {
    if (!session) return;

    // A session's transcript INCLUDES what it was primed with, so copying it wholesale
    // would write the persona into every row twice — once here and once in `personaText`,
    // which for Isolde is 52 KB of duplication per conversation. The standing prefix has
    // its own column; the transcript keeps the turns. Resume puts them back together.
    NSMutableArray<APMessage *> * turns = [NSMutableArray arrayWithCapacity:session.transcript.count];
    for (APMessage * message in session.transcript)
        if (message.role != APRoleSystem) [turns addObject:message];
    self.messages = turns;

    self.contextTokenCount = (int64_t)session.contextTokenCount;
    self.reasoningEnabled = session.reasoningEnabled;

    if ([session isKindOfClass:APGoogleSession.class]) {
        self.backend = CDChatSessionBackendGoogle;
        self.modelIdentifier = [(APGoogleSession *)session modelName];
    } else {
        self.backend = CDChatSessionBackendLocal;
    }
    if (model) self.modelIdentifier = model.modelIdentifier;

    if (self.title.length == 0) {
        NSString * title = [CDChatSession titleFromMessages:self.messages];
        if (title) self.title = title;
    }
}

- (void)recordPersonaText:(nullable NSString *)text atPath:(nullable NSURL *)path {
    self.personaText = text;
    self.personaPath = path;
    NSData * utf8 = [text dataUsingEncoding:NSUTF8StringEncoding];
    self.personaSHA256 = utf8.length ? apSHA256Hex(utf8) : nil;
}

/// The first non-empty user turn, first line, clipped on a character boundary.
+ (nullable NSString *)titleFromMessages:(NSArray<APMessage *> *)messages {
    for (APMessage * message in messages) {
        if (message.role != APRoleUser) continue;
        NSString * text = [message.textRepresentation
                           stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
        if (text.length == 0) continue;

        NSRange newline = [text rangeOfCharacterFromSet:NSCharacterSet.newlineCharacterSet];
        if (newline.location != NSNotFound) text = [text substringToIndex:newline.location];
        if (text.length > kDerivedTitleLength) {
            NSRange clip = [text rangeOfComposedCharacterSequencesForRange:
                            NSMakeRange(0, kDerivedTitleLength - 1)];
            text = [[text substringWithRange:clip] stringByAppendingString:@"…"];
        }
        return text;
    }
    return nil;
}

@end
