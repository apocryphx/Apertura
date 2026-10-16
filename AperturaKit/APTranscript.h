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
//            { "kind": "image", "id": <sha256 hex>, "type": <media type>, "size": N,
//              "width": px, "height": px, "name": … },
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
//  An image part is a REFERENCE: the bytes live in an APFileStore (sidecar files, never
//  base64 in the document — this JSON rides Core Data and CloudKit), and decoding with a
//  store resolves each reference to its file.
//
//  Decoding is forward-compatible: unknown roles and kinds are skipped, not guessed at,
//  and the WRITER's version is reported so a caller can tell the row may hold more than
//  it got back.
#import <Foundation/Foundation.h>
#import <AperturaKit/APMessage.h>

@class APFileStore;

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
/// none to read. Reserved (audio) and unknown parts are skipped rather than guessed at.
/// Image parts decode with no `fileURL` (same as passing a nil store below): displayable
/// as a placeholder, never dropped.
+ (nullable NSArray<APMessage *> *)messagesFromData:(NSData *)data
                                            version:(int16_t *)outVersion;

/// Same, resolving every image reference against `fileStore`: an image whose bytes are
/// in the store decodes with its `fileURL`, one whose bytes are missing decodes with nil
/// `fileURL` (the reference is kept so a later re-save does not lose it).
+ (nullable NSArray<APMessage *> *)messagesFromData:(NSData *)data
                                            version:(int16_t *)outVersion
                                          fileStore:(nullable APFileStore *)fileStore;

@end

NS_ASSUME_NONNULL_END
