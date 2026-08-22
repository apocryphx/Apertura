//  APTranscript — see header. Moved verbatim from CDChatSession's private codec
//  (2026-08-21) so the app and the MCP server share one wire format; behavior is
//  unchanged by design.
#import "APTranscript.h"
#import "APContent.h"

const int16_t APTranscriptCurrentSchemaVersion = 1;

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

@implementation APTranscript

+ (NSData *)dataFromMessages:(NSArray<APMessage *> *)messages {
    NSMutableArray<NSDictionary *> * turns = [NSMutableArray arrayWithCapacity:messages.count];
    for (APMessage * message in messages) {
        NSString * role = apRoleName(message.role);
        if (!role) {
            NSLog(@"APTranscript: dropping a message with unknown role %ld", (long)message.role);
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

    NSDictionary * document = @{ kKeyVersion  : @(APTranscriptCurrentSchemaVersion),
                                 kKeyMessages : turns };
    // Sorted keys: identical transcripts produce identical bytes, which keeps diffs and
    // any future content hashing honest.
    NSError * error = nil;
    NSData * data = [NSJSONSerialization dataWithJSONObject:document
                                                    options:NSJSONWritingSortedKeys
                                                      error:&error];
    if (!data) NSLog(@"APTranscript: could not encode transcript — %@", error);
    return data;
}

+ (NSArray<APMessage *> *)messagesFromData:(NSData *)data version:(int16_t *)outVersion {
    *outVersion = 0;
    if (data.length == 0) return nil;

    NSError * error = nil;
    id document = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (![document isKindOfClass:NSDictionary.class]) {
        NSLog(@"APTranscript: stored bytes are not a JSON object — %@", error ?: document);
        return nil;
    }
    id rawVersion = document[kKeyVersion];
    if ([rawVersion isKindOfClass:NSNumber.class]) *outVersion = [rawVersion shortValue];
    id rawTurns = document[kKeyMessages];
    if (![rawTurns isKindOfClass:NSArray.class]) {
        NSLog(@"APTranscript: document has no messages array");
        return nil;
    }

    NSMutableArray<APMessage *> * messages = [NSMutableArray arrayWithCapacity:[rawTurns count]];
    for (id rawTurn in (NSArray *)rawTurns) {
        if (![rawTurn isKindOfClass:NSDictionary.class]) continue;
        id rawRole = rawTurn[kKeyRole];
        APRole role;
        if (![rawRole isKindOfClass:NSString.class] || !apRoleFromName(rawRole, &role)) {
            NSLog(@"APTranscript: skipping a turn with unreadable role %@", rawRole);
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
                // than guessed at — the reported version is how a caller learns the
                // document may hold more than it got back.
                if ([rawPart[kKeyKind] isEqual:kKindText] && [text isKindOfClass:NSString.class]) {
                    [content addObject:[APContent textContent:text]];
                }
            }
        }
        [messages addObject:[APMessage messageWithRole:role content:content]];
    }
    return messages;
}

@end
