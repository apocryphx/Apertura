//  APTranscript — see header. Moved verbatim from CDChatSession's private codec
//  (2026-08-21) so the app and the MCP server share one wire format. v2 (2026-10-15)
//  adds the generation record: reasoning, tool-call and attachment parts, and per-turn
//  stats + model. A v1 reader keeps the text parts of a v2 document and skips the rest.
#import "APTranscript.h"
#import "APContent.h"
#import "APResponseInternal.h"

const int16_t APTranscriptCurrentSchemaVersion = 2;

// Document keys.
static NSString * const kKeyVersion  = @"version";
static NSString * const kKeyMessages = @"messages";
static NSString * const kKeyRole     = @"role";
static NSString * const kKeyContent  = @"content";
static NSString * const kKeyKind     = @"kind";
static NSString * const kKeyText     = @"text";
// v2 turn keys.
static NSString * const kKeyModel    = @"model";
static NSString * const kKeyStats    = @"stats";
// v2 part keys (attachment / tool call).
static NSString * const kKeyName      = @"name";
static NSString * const kKeySize      = @"size";
static NSString * const kKeyType      = @"type";
static NSString * const kKeyArguments = @"arguments";
static NSString * const kKeyResult    = @"result";
static NSString * const kKeyStatus    = @"status";
static NSString * const kKeyDuration  = @"duration";
// v2 stats keys.
static NSString * const kKeyPromptTokens   = @"prompt_tokens";
static NSString * const kKeyResponseTokens = @"response_tokens";
static NSString * const kKeyTTFT           = @"ttft";
static NSString * const kKeyPrefillTPS     = @"prefill_tps";
static NSString * const kKeyDecodeTPS      = @"decode_tps";

// Content kind names. These strings are the wire format — never renumber, never rename.
// Role names are APRoleName / APRoleFromName (APMessage.h), the same wire strings.
static NSString * const kKindText       = @"text";
static NSString * const kKindImage      = @"image";
static NSString * const kKindAudio      = @"audio";
static NSString * const kKindReasoning  = @"reasoning";
static NSString * const kKindToolCall   = @"tool_call";
static NSString * const kKindAttachment = @"attachment";

// Tool-call status names.
static NSString * const kStatusSucceeded = @"ok";
static NSString * const kStatusFailed    = @"error";
static NSString * const kStatusVetoed    = @"vetoed";

#pragma mark - Enum <-> name

static NSString * apContentKindName(APContentKind kind) {
    switch (kind) {
        case APContentKindText:       return kKindText;
        case APContentKindImage:      return kKindImage;
        case APContentKindAudio:      return kKindAudio;
        case APContentKindReasoning:  return kKindReasoning;
        case APContentKindToolCall:   return kKindToolCall;
        case APContentKindAttachment: return kKindAttachment;
    }
    return nil;
}

static NSString * apToolCallStatusName(APToolCallStatus status) {
    switch (status) {
        case APToolCallStatusSucceeded: return kStatusSucceeded;
        case APToolCallStatusFailed:    return kStatusFailed;
        case APToolCallStatusVetoed:    return kStatusVetoed;
    }
    return nil;
}

static BOOL apToolCallStatusFromName(NSString * name, APToolCallStatus * outStatus) {
    if      ([name isEqualToString:kStatusSucceeded]) { *outStatus = APToolCallStatusSucceeded; return YES; }
    else if ([name isEqualToString:kStatusFailed])    { *outStatus = APToolCallStatusFailed;    return YES; }
    else if ([name isEqualToString:kStatusVetoed])    { *outStatus = APToolCallStatusVetoed;    return YES; }
    return NO;
}

static NSString * apString(id value) {
    return [value isKindOfClass:NSString.class] ? value : nil;
}

static NSNumber * apNumber(id value) {
    return [value isKindOfClass:NSNumber.class] ? value : nil;
}

#pragma mark - Encoding

/// One part as its wire dictionary; nil for a kind this build cannot encode.
static NSDictionary * apEncodePart(APContent * content) {
    NSString * kind = apContentKindName(content.kind);
    if (!kind) return nil;
    NSMutableDictionary * part = [NSMutableDictionary dictionaryWithObject:kind forKey:kKeyKind];
    switch (content.kind) {
        case APContentKindText:
        case APContentKindReasoning:
        case APContentKindImage:
        case APContentKindAudio:
            if (content.text) part[kKeyText] = content.text;
            break;
        case APContentKindAttachment: {
            APAttachment * a = content.attachment;
            if (!a) return nil;
            part[kKeyName] = a.name;
            part[kKeyText] = a.text;
            part[kKeySize] = @(a.sizeBytes);
            if (a.contentTypeIdentifier) part[kKeyType] = a.contentTypeIdentifier;
            break;
        }
        case APContentKindToolCall: {
            APToolCall * t = content.toolCall;
            NSString * status = t ? apToolCallStatusName(t.status) : nil;
            if (!status) return nil;
            part[kKeyName] = t.name;
            // Arguments are the model's parsed JSON object — already JSON-compatible; an
            // unencodable value would fail the whole document, so guard here.
            part[kKeyArguments] = [NSJSONSerialization isValidJSONObject:t.arguments] ? t.arguments : @{};
            part[kKeyStatus] = status;
            if (t.result) part[kKeyResult] = t.result;
            if (t.duration > 0) part[kKeyDuration] = @(t.duration);
            break;
        }
    }
    return part;
}

static NSDictionary * apEncodeStats(APResponseStats * stats) {
    return @{ kKeyPromptTokens   : @(stats.promptTokenCount),
              kKeyResponseTokens : @(stats.responseTokenCount),
              kKeyTTFT           : @(stats.timeToFirstToken),
              kKeyPrefillTPS     : @(stats.prefillTokensPerSecond),
              kKeyDecodeTPS      : @(stats.decodeTokensPerSecond) };
}

#pragma mark - Decoding

/// One wire part as content; nil for anything unreadable or unknown (skipped, not guessed).
static APContent * apDecodePart(NSDictionary * part) {
    NSString * kind = apString(part[kKeyKind]);
    NSString * text = apString(part[kKeyText]);
    if ([kind isEqualToString:kKindText]) {
        return text ? [APContent textContent:text] : nil;
    }
    if ([kind isEqualToString:kKindReasoning]) {
        return text ? [APContent reasoningContent:text] : nil;
    }
    if ([kind isEqualToString:kKindAttachment]) {
        NSString * name = apString(part[kKeyName]);
        if (!name || !text) return nil;
        return [APContent attachmentContent:
                [APAttachment attachmentWithName:name text:text
                                       sizeBytes:apNumber(part[kKeySize]).unsignedLongLongValue
                           contentTypeIdentifier:apString(part[kKeyType])]];
    }
    if ([kind isEqualToString:kKindToolCall]) {
        NSString * name = apString(part[kKeyName]);
        APToolCallStatus status;
        NSString * statusName = apString(part[kKeyStatus]);
        if (!name || !statusName || !apToolCallStatusFromName(statusName, &status)) return nil;
        NSDictionary * arguments = [part[kKeyArguments] isKindOfClass:NSDictionary.class]
            ? part[kKeyArguments] : @{};
        return [APContent toolCallContent:
                [APToolCall toolCallWithName:name arguments:arguments status:status
                                      result:apString(part[kKeyResult])
                                    duration:apNumber(part[kKeyDuration]).doubleValue]];
    }
    // Image and audio are reserved in APContent with no way to construct them; unknown
    // kinds come from newer writers. Both are skipped — the reported version is how a
    // caller learns the document may hold more than it got back.
    return nil;
}

static APResponseStats * apDecodeStats(id raw) {
    if (![raw isKindOfClass:NSDictionary.class]) return nil;
    NSDictionary * s = raw;
    return [[APResponseStats alloc]
        initWithPromptTokens:apNumber(s[kKeyPromptTokens]).integerValue
              responseTokens:apNumber(s[kKeyResponseTokens]).integerValue
            timeToFirstToken:apNumber(s[kKeyTTFT]).doubleValue
                  prefillTPS:apNumber(s[kKeyPrefillTPS]).doubleValue
                   decodeTPS:apNumber(s[kKeyDecodeTPS]).doubleValue];
}

@implementation APTranscript

+ (NSData *)dataFromMessages:(NSArray<APMessage *> *)messages {
    NSMutableArray<NSDictionary *> * turns = [NSMutableArray arrayWithCapacity:messages.count];
    for (APMessage * message in messages) {
        NSString * role = APRoleName(message.role);
        if (!role) {
            NSLog(@"APTranscript: dropping a message with unknown role %ld", (long)message.role);
            continue;
        }
        NSMutableArray<NSDictionary *> * parts = [NSMutableArray arrayWithCapacity:message.content.count];
        for (APContent * content in message.content) {
            NSDictionary * part = apEncodePart(content);
            if (part) [parts addObject:part];
        }
        NSMutableDictionary * turn = [NSMutableDictionary dictionaryWithDictionary:
                                      @{ kKeyRole : role, kKeyContent : parts }];
        if (message.modelIdentifier) turn[kKeyModel] = message.modelIdentifier;
        if (message.stats) turn[kKeyStats] = apEncodeStats(message.stats);
        [turns addObject:turn];
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
        if (![rawRole isKindOfClass:NSString.class] || !APRoleFromName(rawRole, &role)) {
            NSLog(@"APTranscript: skipping a turn with unreadable role %@", rawRole);
            continue;
        }

        NSMutableArray<APContent *> * content = [NSMutableArray array];
        id rawParts = rawTurn[kKeyContent];
        if ([rawParts isKindOfClass:NSArray.class]) {
            for (id rawPart in (NSArray *)rawParts) {
                if (![rawPart isKindOfClass:NSDictionary.class]) continue;
                APContent * part = apDecodePart(rawPart);
                if (part) [content addObject:part];
            }
        }
        [messages addObject:[APMessage messageWithRole:role content:content
                                                 stats:apDecodeStats(rawTurn[kKeyStats])
                                       modelIdentifier:apString(rawTurn[kKeyModel])]];
    }
    return messages;
}

@end
