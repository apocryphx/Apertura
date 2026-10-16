//  APLocalSession — streaming conversation backend over the gated engine paths.
//
//  The token flow mirrors es::ESSession::respond EXACTLY (same sampling order, same
//  final-token caching) and the turn deltas mirror the --chat-session construction that
//  is delta-byte-identity-verified against ESChatTemplate::build. --facade-verify gates
//  this file against the reference path token-for-token.
//
//  Threading: ALL engine work runs on the model's dedicated engine thread (MLX streams
//  are per-thread — see APEngineRunner in APModel.mm); engine-side state read by public
//  getters is guarded by @synchronized(self); transcript/tool state lives in the
//  APSession base. Callbacks are delivered on callbackQueue (base `deliver:`).
#import "APLocalSession.h"
#import "APLocalSession+Checkpointing.h"
#import "APSessionSubclass.h"
#import "APInternal.h"
#import "APResponseInternal.h"
#import "APError.h"
#import "APGemmaToolGrammar.h"
#import <CommonCrypto/CommonDigest.h>

#include "ESKVCache.h"
#include "ESSampler.h"
#include "ESImagePreprocessor.h"
#include "ESGemma4VisionModel.h"
#include "mlx/mlx.h"

#include <memory>
#include <string>
#include <vector>

namespace mx = mlx::core;

/// Length of the longest prefix of `s` that ends on a complete UTF-8 sequence.
static size_t apCompleteUTF8PrefixLength(const std::string & s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        unsigned char c = (unsigned char)s[n - back];
        if ((c & 0x80) == 0) return n;                 // ASCII: complete
        if ((c & 0xC0) == 0xC0) {                      // lead byte at n-back
            size_t need = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2;
            return (back >= need) ? n : n - back;      // complete iff all continuations present
        }
        // continuation byte: keep scanning backwards
    }
    return n;
}

/// The tool_response body for one dispatch outcome: {error:<|"|>…<|"|>} when the tool
/// failed or was vetoed, {result:<|"|>…<|"|>} otherwise — the gated CLI --tool-result
/// grammar (strings sanitized so results cannot smuggle marker text).
static NSString * apToolResponseBody(APToolOutcome * outcome) {
    if (outcome.error)
        return [NSString stringWithFormat:@"{error:<|\"|>%@<|\"|>}",
                [APGemmaToolGrammar sanitize:outcome.error.localizedDescription]];
    if (outcome.vetoed) return @"{error:<|\"|>tool unavailable<|\"|>}";
    return [NSString stringWithFormat:@"{result:<|\"|>%@<|\"|>}",
            [APGemmaToolGrammar sanitize:outcome.resultText ?: @""]];
}

/// The ESSoftTokens for a prompt: every `imageTokenId` position in `ids`, paired in order with
/// the concatenated per-image soft tokens. Empty when there are no images.
static es::ESSoftTokens apSoftTokens(const std::vector<int> & ids, int imageTokenId,
                                     const std::vector<mx::array> & imageEmbeds) {
    if (imageEmbeds.empty()) return es::ESSoftTokens();
    std::vector<int> positions;
    for (int i = 0; i < (int) ids.size(); ++i) if (ids[i] == imageTokenId) positions.push_back(i);
    mx::array all = imageEmbeds.size() == 1 ? imageEmbeds[0] : mx::concatenate(imageEmbeds, 0);
    if ((int) positions.size() != all.shape(0))
        throw std::runtime_error("image placeholder count does not match the encoded soft tokens");
    return es::ESSoftTokens(std::move(positions), all);
}

@implementation APLocalSession

#pragma mark - Images

- (BOOL)acceptsImageContent {
    return _model.hasVisionTower;
}

- (NSString *)imageContentRefusalReason {
    return [NSString stringWithFormat:
            @"the loaded model (%@) has no vision tower — an image cannot be sent to it",
            _model.modelIdentifier];
}

/// ENGINE THREAD. Preprocess (at `budget` soft tokens) + encode every image of `m`, appending
/// one [softTokens, hidden] array per image to `embeds` and its soft-token count to `counts`.
/// Fails with APErrorUnsupportedContent (no tower), APErrorFileMissing (no resolved file) or
/// APErrorFileUnreadable (not a decodable image).
- (BOOL)encodeImagesOfMessage:(APMessage *)m
                       budget:(NSInteger)budget
                       counts:(std::vector<int> &)counts
                       embeds:(std::vector<mx::array> &)embeds
                        error:(NSError **)error {
    if (m.images.count == 0) return YES;
    if (![APGenerationOptions isSupportedVisualTokenBudget:budget]) {
        if (error) *error = APMakeError(APErrorInvalidMessage, [NSString stringWithFormat:
            @"visualTokenBudget %ld is not supported (70, 140, 280, 560 or 1120)", (long) budget]);
        return NO;
    }
    es::ESGemma4VisionModel * vision = [_model internalVision];
    if (!vision) {
        if (error) *error = APMakeError(APErrorUnsupportedContent, [self imageContentRefusalReason]);
        return NO;
    }
    es::ESImagePreprocessor pre([_model internalConfig]->vision);
    for (APImage * image in m.images) {
        if (!image.fileURL) {
            if (error) *error = APMakeError(APErrorFileMissing, [NSString stringWithFormat:
                @"image %@ (%@) is no longer available", image.originalName ?: image.identifier, image.mediaType]);
            return NO;
        }
        try {
            es::ESImagePatches patches = pre.processFile(std::string(image.fileURL.path.UTF8String), (int) budget);
            counts.push_back(patches.numSoftTokens());
            embeds.push_back(vision->softTokens(patches));
        } catch (const std::exception & e) {
            if (error) *error = APMakeError(APErrorFileUnreadable, [NSString stringWithFormat:
                @"image %@ could not be decoded: %s", image.originalName ?: image.identifier, e.what()]);
            return NO;
        }
    }
    return YES;
}

- (instancetype)initWithModel:(APModel *)model {
    if ((self = [super init])) {
        _model = model;
        _cache = std::make_unique<es::ESKVCache>([model internalConfig]->numHiddenLayers);
        _pos = 0;
        _turnCount = 0;
        _openModelTurn = NO;
        _warnedNearFull = NO;
    }
    return self;
}

- (NSInteger)contextTokenCount {
    @synchronized(self) { return _pos; }
}

- (void)reset {
    [_model performOnEngineAndWait:^{
        self->_cache->reset();
        @synchronized(self) {
            self->_pos = 0;
            self->_turnCount = 0;
            self->_openModelTurn = NO;
            self->_warnedNearFull = NO;
            self->_lastIds = nil;
        }
        [self clearTranscript];
    }];
}

- (NSInteger)contextLimit {
    NSInteger limit = [_model internalConfiguration].maximumContextLength;
    return limit > 0 ? MIN(limit, _model.maximumContextLength) : _model.maximumContextLength;
}

- (NSArray<NSNumber *> *)lastResponseTokenIDsForTesting {
    @synchronized(self) { return _lastIds; }
}

#pragma mark - Prime

NSInteger APLocalSessionResponseTokenBudget(NSInteger limit, NSInteger position,
                                            NSInteger turnTokenCount, NSInteger requested) {
    const NSInteger available = limit - position - turnTokenCount - 2;
    if (available < 1) return 0;
    return requested > 0 ? MIN(requested, available) : available;
}

/// Snapshot validity key: format version, the model's identity bytes (name, weight byte
/// count, head precision — -[APModel identityData], shared with checkpoint fingerprints)
/// and the EXACT prime token ids (which transitively cover the persona text, tokenizer,
/// and chat-template layout). SHA-256, hex.
/// Images contribute their content identifiers: two pictures of one size render the SAME
/// placeholder ids, so the ids alone would let a snapshot primed on one picture serve another.
static std::string apSnapshotFingerprint(APModel * model, const std::vector<int> & ids,
                                         NSArray<NSString *> * imageIdentifiers = nil) {
    NSMutableData * blob = [NSMutableData data];
    uint32_t version = 1;
    [blob appendBytes:&version length:sizeof(version)];
    [blob appendData:[model identityData]];
    [blob appendBytes:ids.data() length:ids.size() * sizeof(int)];
    for (NSString * ident in imageIdentifiers)
        [blob appendData:[[ident stringByAppendingString:@"\n"] dataUsingEncoding:NSUTF8StringEncoding]];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(blob.bytes, (CC_LONG) blob.length, digest);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    return std::string(hex, 2 * CC_SHA256_DIGEST_LENGTH);
}

- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                             cacheURL:(NSURL *)cacheURL
                           completion:(void (^)(NSError *_Nullable))completion {
    NSProgress * progress = [NSProgress progressWithTotalUnitCount:-1];
    APResponseTask * task = [[APResponseTask alloc] initWithProgress:progress];
    [_model performOnEngine:^{
        NSError * invalid = [self validationErrorForPrimeMessages:messages];
        if (invalid) { [self deliver:^{ completion(invalid); }]; return; }
        try {
            std::vector<es::ESChatMessage> msgs;
            std::vector<mx::array> imageEmbeds;          // soft tokens, prompt order
            NSMutableArray<NSString *> * imageIds = [NSMutableArray array];
            for (APMessage * m in messages) {
                es::ESChatMessage cm{std::string((APRoleName(m.role) ?: @"user").UTF8String),
                                     std::string(m.textRepresentation.UTF8String)};
                NSError * imgErr = nil;
                // Prime replays history at the default budget (the budget is a per-turn generation choice).
                if (![self encodeImagesOfMessage:m budget:APVisualTokenBudgetDefault
                                          counts:cm.imageSoftTokenCounts embeds:imageEmbeds error:&imgErr]) {
                    [self deliver:^{ completion(imgErr); }]; return;
                }
                for (APImage * im in m.images) [imageIds addObject:im.identifier];
                msgs.push_back(std::move(cm));
            }
            es::ESChatTemplate * chat = [self->_model internalTemplate];
            // Registered tools are advertised in the system turn (register BEFORE prime).
            // Sorted by name so the prime ids — and the snapshot fingerprint — are stable.
            std::vector<std::string> decls;
            for (id<APTool> tool in [self toolsSortedByName])
                decls.push_back(std::string([APGemmaToolGrammar declarationForTool:tool].UTF8String));
            std::vector<int> ids = chat->build(msgs, /*think=*/self.reasoningEnabled, /*addGen=*/false, decls);
            if ((NSInteger)ids.size() + 64 > [self contextLimit]) {
                [self deliver:^{ completion(APMakeError(APErrorContextOverflow,
                    @"prime messages exceed the context limit")); }];
                return;
            }

            // Snapshot fast path: valid only on a fresh session; restored content is
            // byte-identical to a fresh prefill (--persist-verify), so continuation matches.
            BOOL restored = NO;
            std::string fingerprint;
            if (cacheURL && self->_pos == 0) {
                fingerprint = apSnapshotFingerprint(self->_model, ids, imageIds);
                if ([NSFileManager.defaultManager fileExistsAtPath:cacheURL.path]) {
                    int pos = self->_cache->restoreSnapshot(std::string(cacheURL.path.UTF8String),
                                                            fingerprint, APLocalSessionRawMode(self->_model));
                    if (pos == (int) ids.size()) {
                        restored = YES;
                    } else if (pos >= 0) {
                        self->_cache->reset();   // valid file, unexpected pos — refill cleanly
                    }
                }
            }

            if (!restored) {
                es::ESSoftTokens soft = apSoftTokens(ids, [self->_model internalConfig]->vision.imageTokenId, imageEmbeds);
                mx::array ll = [self->_model internalLM]->lastLogits(ids, self->_cache.get(), self->_pos, &soft);
                mx::eval(ll);
                if (cacheURL && self->_pos == 0) {   // best-effort write; priming already succeeded
                    self->_cache->saveSnapshot(std::string(cacheURL.path.UTF8String),
                                               fingerprint, (int) ids.size());
                }
            }
            @synchronized(self) { self->_pos += (int)ids.size(); }
            [self noteLastPrimeRestoredFromSnapshot:restored];
            [self appendTranscriptMessages:messages];
            [self deliver:^{ completion(nil); }];
        } catch (const std::exception & e) {
            NSError * err = APMakeError(APErrorEngineFailure, @(e.what()));
            [self deliver:^{ completion(err); }];
        }
    }];
    return task;
}

#pragma mark - Respond

- (APResponseTask *)respondToMessage:(APMessage *)message
                             options:(APGenerationOptions *)options
                        deltaHandler:(void (^)(APResponseDelta *))deltaHandler
                          completion:(void (^)(APResponse *, NSError *))completion {
    APGenerationOptions * opts = [options copy] ?: [APGenerationOptions defaultOptions];
    NSInteger maxTokens = opts.maximumResponseTokens;
    NSProgress * progress = [NSProgress progressWithTotalUnitCount:(maxTokens > 0 ? maxTokens : -1)];
    APResponseTask * task = [[APResponseTask alloc] initWithProgress:progress];

    [_model performOnEngine:^{
        NSError * invalid = [self validationErrorForRespondMessage:message];
        if (invalid) { [self deliver:^{ completion(nil, invalid); }]; return; }
        try {
            [self runTurnWithMessage:message options:opts task:task progress:progress
                        deltaHandler:deltaHandler completion:completion];
        } catch (const std::exception & e) {
            NSError * err = APMakeError(APErrorEngineFailure, @(e.what()));
            [self deliver:^{ completion(nil, err); }];
        }
    }];
    return task;
}

// Runs on the ENGINE thread. Token flow mirrors es::ESSession::respond; turn delta
// mirrors the --chat-session construction (delta-byte-identity-verified vs build()).
- (void)runTurnWithMessage:(APMessage *)message
                   options:(APGenerationOptions *)opts
                      task:(APResponseTask *)task
                  progress:(NSProgress *)progress
              deltaHandler:(void (^)(APResponseDelta *))deltaHandler
                completion:(void (^)(APResponse *, NSError *))completion {
    es::ESGemma4TextForCausalLM * lm = [_model internalLM];
    es::ESTokenizer * tok = [_model internalTokenizer];
    es::ESChatTemplate * chat = [_model internalTemplate];
    const es::ESChatTokens & T = chat->tokens();

    auto enc = [&](const char * s) { return tok->encode(s, /*addSpecial=*/false); };
    auto push = [](std::vector<int> & a, const std::vector<int> & b) {
        a.insert(a.end(), b.begin(), b.end());
    };

    // Unprimed session: ingest the empty prefix once (BOS + the rendered-empty system
    // turn, matching build()'s layout) so the first turn delta composes identically.
    if (_pos == 0) {
        std::vector<int> prefix = chat->build({}, /*think=*/self.reasoningEnabled, /*addGen=*/false);
        mx::array pl = lm->lastLogits(prefix, _cache.get(), _pos);
        mx::eval(pl);
        @synchronized(self) { _pos += (int)prefix.size(); }
    }

    // ---- turn delta (user turn + open model turn, thinking pre-closed). Mirrors the
    // --chat-session construction: no separator before the FIRST turn after the prefix.
    std::vector<int> d;
    if (_openModelTurn) { d.push_back(T.turnClose); push(d, enc("\n")); _openModelTurn = NO; }
    else if (_turnCount > 0) push(d, enc("\n"));
    d.push_back(T.turnOpen); push(d, enc("user\n"));
    // Images first (the template renders content parts [image..., text]): each becomes
    // <|image> N×<|image|> <image|>; the encoder's soft tokens replace the placeholders at prefill.
    std::vector<int> imageCounts;
    std::vector<mx::array> imageEmbeds;
    {
        NSError * imgErr = nil;
        if (![self encodeImagesOfMessage:message budget:opts.visualTokenBudget
                                  counts:imageCounts embeds:imageEmbeds error:&imgErr]) {
            [self deliver:^{ completion(nil, imgErr); }];
            return;
        }
        for (int n : imageCounts) push(d, chat->imageTokens(n));
    }
    push(d, tok->encode(std::string(message.textRepresentation.UTF8String), false));
    d.push_back(T.turnClose); push(d, enc("\n"));
    d.push_back(T.turnOpen); push(d, enc("model\n"));
    // Reasoning off: pre-close an EMPTY thought channel where this checkpoint's template does
    // (suppresses reasoning; E2B/E4B templates don't, and the model emits a stray <channel|> if
    // we do). Reasoning on: leave the model turn open — the model writes its own thought channel.
    const BOOL think = self.reasoningEnabled;
    if (!think && chat->preclosesEmptyThought()) {
        d.push_back(T.channelOpen); push(d, enc("thought\n")); d.push_back(T.channelClose);
    }

    // ---- context pre-flight ----
    NSInteger limit = [self contextLimit];
    NSInteger maxNew = APLocalSessionResponseTokenBudget(
        limit, _pos, (NSInteger)d.size(), opts.maximumResponseTokens);
    if (maxNew < 1) {
        [self deliver:^{ completion(nil, APMakeError(APErrorContextOverflow,
            @"context limit reached; reset the session or raise maximumContextLength")); }];
        return;
    }

    es::ESSamplingConfig sc;
    sc.greedy = (opts.temperature <= 0);
    sc.temperature = MAX(opts.temperature, 1e-6f);
    sc.topK = (int)opts.topK;
    sc.topP = opts.topP;
    sc.seed = opts.seed;
    sc.maxNewTokens = (int)maxNew;
    sc.eosTokenId = chat->stopToken();
    es::ESSampler sampler(sc);

    // ---- prefill the turn delta (mirrors ESSession::respond; image soft tokens scattered in) ----
    NSDate * t0 = [NSDate date];
    es::ESSoftTokens soft = apSoftTokens(d, [_model internalConfig]->vision.imageTokenId, imageEmbeds);
    mx::array ll = lm->lastLogits(d, _cache.get(), _pos, &soft);
    mx::eval(ll);
    NSTimeInterval prefillS = -[t0 timeIntervalSinceNow];
    @synchronized(self) { _pos += (int)d.size(); }

    // Reasoning excision (Kolja's design, gated by --excise-verify): mark the turn start
    // now; after the turn completes, rewind the cache here and re-prefill only the answer,
    // so the thought channel never stays in context — the cache then matches the reference
    // template's stripped-history rendering. The mark deep-copies the sliding layers' live
    // windows (a later compaction would drop their pre-window rows); the copies are freed
    // at rewind/clear below.
    const BOOL exciseArmed = (think && self.excludesReasoningFromContext)
                                 ? _cache->markRewindPoint(_pos) : NO;

    std::vector<int> out;
    int next = sampler.sample(ll);
    out.push_back(next);
    NSTimeInterval ttft = -[t0 timeIntervalSinceNow];

    // ---- streaming decode: channel- and tool-aware ----
    // Stream shape: [<|channel>thought\n <reasoning> <channel|>] [<|tool_call>call:… <tool_call|>]*
    // <answer>. Marker tokens contribute no text (skip-special decode), so region flips
    // happen exactly at byte boundaries of the running diff. Thought text streams with
    // isThought; tool-call text is machinery and never streams; the answer's leading
    // newlines are trimmed. When the model closes a tool call and a matching tool is
    // registered, the turn PAUSES: the call dispatches (delegate may veto), and
    // <|tool_response>response:NAME{…}<tool_response|> is spliced with the model turn
    // still open — exactly the gated CLI --tool-result pattern — then decoding resumes
    // within this same respond lifetime.
    NSArray<id<APTool>> * activeTools = [self toolsSortedByName];
    APTurnRecorder * recorder = [[APTurnRecorder alloc] initWithSession:self deltaHandler:deltaHandler];

    std::string decoded;
    size_t emitted = 0;
    bool inThought = false, thoughtLabelPending = false, answerLeadPending = false;
    bool inToolCall = false, sawToolCallClose = false;
    APFinishReason reason = APFinishReasonMaxTokens;
    auto emitDeltas = [&](NSInteger tokens) {
        decoded = tok->decode(out, /*skipSpecial=*/true);
        size_t safe = apCompleteUTF8PrefixLength(decoded);
        if (safe <= emitted) return;
        std::string chunk(decoded.data() + emitted, safe - emitted);
        emitted = safe;
        if (inToolCall) return;                          // call text is machinery, not chat
        if (inThought && thoughtLabelPending) {          // drop through "thought\n"
            size_t nl = chunk.find('\n');
            if (nl == std::string::npos) return;         // label not complete yet
            chunk.erase(0, nl + 1);
            thoughtLabelPending = false;
        }
        if (!inThought && answerLeadPending) {           // trim the answer's leading newlines
            size_t start = chunk.find_first_not_of('\n');
            if (start == std::string::npos) return;
            chunk.erase(0, start);
            answerLeadPending = false;
        }
        if (chunk.empty()) return;
        NSString * text = [[NSString alloc] initWithBytes:chunk.data() length:chunk.size()
                                                 encoding:NSUTF8StringEncoding];
        if (text.length == 0) return;
        [recorder emit:[[APResponseDelta alloc] initWithText:text tokenCount:tokens
                                                   isThought:inThought]];
    };
    auto noteToken = [&](int tokId) {
        if (tokId == T.channelOpen)    { emitDeltas(0); inThought = true;  thoughtLabelPending = true; }
        if (tokId == T.channelClose)   { emitDeltas(0); inThought = false; answerLeadPending = true; }
        if (tokId == T.toolCallOpen)   { emitDeltas(0); inToolCall = true; }
        if (tokId == T.toolCallClose)  { emitDeltas(0); inToolCall = false; sawToolCallClose = true; }
    };
    noteToken(next);
    emitDeltas(1);
    progress.completedUnitCount = 1;

    NSDate * tDecode = [NSDate date];
    int toolRounds = 0;
    while (true) {
        // ---- sample until end-of-turn / budget / cancel / a completed tool call ----
        while ((int) out.size() < sc.maxNewTokens) {
            if (next == sc.eosTokenId) { reason = APFinishReasonEndOfTurn; break; }
            if ([task isCancelled])    { reason = APFinishReasonCancelled;  break; }
            if (sawToolCallClose && activeTools.count && toolRounds < APSessionMaxToolRounds) break;
            ll = lm->lastLogits({next}, _cache.get(), _pos);
            mx::eval(ll);
            @synchronized(self) { _pos += 1; }
            next = sampler.sample(ll);
            out.push_back(next);
            progress.completedUnitCount = (int64_t)out.size();
            noteToken(next);
            emitDeltas(1);
        }
        if (reason != APFinishReasonCancelled && next == sc.eosTokenId)
            reason = APFinishReasonEndOfTurn;   // eos sampled on the final permitted step

        if (!(sawToolCallClose && activeTools.count && toolRounds < APSessionMaxToolRounds) ||
            reason == APFinishReasonEndOfTurn || reason == APFinishReasonCancelled) break;
        sawToolCallClose = false;
        toolRounds += 1;

        // ---- dispatch the call (the LAST parsed call is this round's) ----
        es::ESParsedResponse midParse = chat->parse(out);
        if (midParse.toolCalls.empty()) continue;        // defensive: nothing to dispatch
        const es::ESToolCall & call = midParse.toolCalls.back();
        NSString * callName = @(call.name.c_str()) ?: @"";
        NSDictionary * args = [APGemmaToolGrammar parseArguments:@(call.args.c_str()) ?: @""];
        // Veto → invoke → notify in the base; the invoke blocks THIS engine thread until
        // the tool's completion fires (other sessions on the model wait — documented).
        APToolOutcome * outcome = [self dispatchToolNamed:callName arguments:args
                                                    among:activeTools recorder:recorder];
        NSString * body = apToolResponseBody(outcome);

        // ---- splice: pending <tool_call|> + <|tool_response>response:NAME{…}<tool_response|>,
        // model turn stays OPEN; the returned logits continue the turn.
        std::vector<int> splice;
        splice.push_back(next);                          // the un-cached <tool_call|>
        splice.push_back(T.toolRespOpen);
        std::vector<int> respIds = chat->encodeQuoted("response:" + call.name +
                                                      "{" + std::string(body.UTF8String) + "}");
        splice.insert(splice.end(), respIds.begin(), respIds.end());
        splice.push_back(T.toolRespClose);
        ll = lm->lastLogits(splice, _cache.get(), _pos);
        mx::eval(ll);
        @synchronized(self) { _pos += (int) splice.size(); }
        next = sampler.sample(ll);
        out.push_back(next);
        answerLeadPending = true;                        // trim newline after the response
        noteToken(next);
        emitDeltas(1);
    }
    NSTimeInterval decodeS = -[tDecode timeIntervalSinceNow];

    // Reasoning excision: rewind to the pre-turn mark and re-prefill the kept tokens —
    // everything after the thought channel closes, including the final sampled token (which
    // the non-excised path below caches separately). Kept-as-generated: turns with tool
    // rounds (the reference template retains thinking for those), turns cancelled or
    // budget-capped mid-thought (no channel close), and thought-less turns. A refused
    // rewind (rewindToMark -1) leaves the cache exactly as generated — never worse.
    BOOL excised = NO;
    if (exciseArmed) {
        int ansStart = -1;
        if (toolRounds == 0 && out.size() > 1 && out.front() == T.channelOpen)
            for (size_t k = 1; k < out.size(); ++k)
                if (out[k] == T.channelClose) { ansStart = (int)k + 1; break; }
        if (ansStart > 0 && ansStart < (int)out.size()) {
            const int mark = _cache->rewindToMark();
            if (mark >= 0) {
                std::vector<int> kept(out.begin() + ansStart, out.end());
                mx::array kl = lm->lastLogits(kept, _cache.get(), mark);
                mx::eval(kl);
                @synchronized(self) { _pos = mark + (int)kept.size(); }
                excised = YES;
            }
        }
        if (!excised) _cache->clearRewindMark();
    }

    // Cache the final sampled token so the next turn attends the complete reply
    // (identical to ESSession::respond). The excised path already prefilled it above.
    if (!excised && !out.empty()) {
        mx::array t = lm->lastLogits({out.back()}, _cache.get(), _pos);
        mx::eval(t);
        @synchronized(self) { _pos += 1; }
    }
    _openModelTurn = (reason != APFinishReasonEndOfTurn);
    _turnCount += 1;

    // ---- near-full signal (once) ----
    if (!_warnedNearFull && _pos > (limit * 4) / 5) {
        _warnedNearFull = YES;
        __weak APSession * weakSelf = self;
        id<APSessionDelegate> delegate = self.delegate;
        if ([delegate respondsToSelector:@selector(sessionContextIsNearlyFull:)])
            [self deliver:^{ APSession * s = weakSelf; if (s) [delegate sessionContextIsNearlyFull:s]; }];
    }

    // ---- finalize: the recorded turn (what streamed, in order), stats, transcript ----
    APResponseStats * stats = [[APResponseStats alloc]
        initWithPromptTokens:(NSInteger)d.size()
              responseTokens:(NSInteger)out.size()
            timeToFirstToken:ttft
                   prefillTPS:(prefillS > 0 ? d.size() / prefillS : 0)
                    decodeTPS:(decodeS > 0 ? (out.size() > 1 ? (out.size() - 1) / decodeS : 0) : 0)];
    APMessage * reply = [recorder assistantMessageWithStats:stats
                                            modelIdentifier:_model.modelIdentifier];
    NSMutableArray<NSNumber *> * ids = [NSMutableArray arrayWithCapacity:out.size()];
    for (int t : out) [ids addObject:@(t)];
    [self appendTranscriptMessages:@[ message, reply ]];
    @synchronized(self) { _lastIds = ids; }

    APResponse * response = [[APResponse alloc] initWithMessage:reply finishReason:reason
                                                          stats:stats];
    [self deliver:^{ completion(response, nil); }];
}

@end
