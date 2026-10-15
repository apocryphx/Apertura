//  APTranscript — the versioned JSON transcript codec.
//
//  One codec for every host: the app's Core Data rows and the MCP server encode and
//  decode conversations through here, so a transcript written by one is always readable
//  by the other. The document is deliberately tool-friendly (jq/sqlite3): roles and
//  content kinds are STRINGS — the wire format; never renumber, never rename —
//  and keys are sorted so identical transcripts produce identical bytes.
//
//  Shape (v2):
//    { "version": 2, "messages": [
//        { "role": "user", "content": [
//            { "kind": "attachment", "name": …, "text": …, "size": N, "type": <UTI> },
//            { "kind": "text", "text": … } ] },
//        { "role": "assistant", "model": …,
//          "stats": { "prompt_tokens": N, "response_tokens": N, "ttft": s,
//                     "prefill_tps": x, "decode_tps": x },
//          "content": [
//            { "kind": "reasoning", "text": … },
//            { "kind": "tool_call", "name": …, "arguments": {…},
//              "status": "ok"|"error"|"vetoed", "result": …, "duration": s },
//            { "kind": "text", "text": … } ] } ] }
//  v1 wrote text parts only, with no turn-level keys; a v1 document decodes unchanged.
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
/// none to read. Reserved (image/audio) and unknown parts are skipped rather than
/// guessed at.
+ (nullable NSArray<APMessage *> *)messagesFromData:(NSData *)data
                                            version:(int16_t *)outVersion;

@end

NS_ASSUME_NONNULL_END
