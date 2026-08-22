//  APTranscript — the versioned JSON transcript codec.
//
//  One codec for every host: the app's Core Data rows and the MCP server encode and
//  decode conversations through here, so a transcript written by one is always readable
//  by the other. The document is deliberately tool-friendly (jq/sqlite3): roles and
//  content kinds are STRINGS — the wire format; never renumber, never rename —
//  and keys are sorted so identical transcripts produce identical bytes.
//
//  Shape: { "version": N, "messages": [ { "role": "user"|"assistant"|"system"|"tool",
//           "content": [ { "kind": "text"|"image"|"audio", "text": … } ] } ] }
//
//  Decoding is forward-compatible: unknown roles and kinds are skipped, not guessed at,
//  and the WRITER's version is reported so a caller can tell the row may hold more than
//  it got back.
#import <Foundation/Foundation.h>
#import <AperturaKit/APMessage.h>

NS_ASSUME_NONNULL_BEGIN

/// The envelope version this build writes.
FOUNDATION_EXPORT const int16_t APTranscriptCurrentSchemaVersion;

@interface APTranscript : NSObject

/// Encode messages as the versioned JSON document. Messages with an unknown role are
/// dropped (logged); content parts with an unknown kind are skipped. nil on JSON
/// encoding failure.
+ (nullable NSData *)dataFromMessages:(NSArray<APMessage *> *)messages;

/// Decode a transcript document. nil when `data` is empty or not a transcript.
/// `outVersion` (required) receives the version the document claims — 0 when there is
/// none to read. Text is all v1 can rebuild; image/audio parts are skipped rather than
/// guessed at.
+ (nullable NSArray<APMessage *> *)messagesFromData:(NSData *)data
                                            version:(int16_t *)outVersion;

@end

NS_ASSUME_NONNULL_END
