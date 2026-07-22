//  APSession — streaming conversation facade over the gated engine paths.
//
//  The token flow mirrors es::ESSession::respond EXACTLY (same sampling order, same
//  final-token caching) and the turn deltas mirror the --chat-session construction that
//  is delta-byte-identity-verified against ESChatTemplate::build. --facade-verify gates
//  this file against the reference path token-for-token.
//
//  Threading: ALL engine work runs on the model's dedicated engine thread (MLX streams
//  are per-thread — see APEngineRunner in APModel.mm); shared state read by public
//  getters is guarded by @synchronized(self). Callbacks are delivered on callbackQueue.
#import "APSession.h"
#import "APInternal.h"
#import "APError.h"
#import <CommonCrypto/CommonDigest.h>

#include "ESKVCache.h"
#include "ESSampler.h"
#include "mlx/mlx.h"

#include <memory>
#include <string>
#include <vector>

namespace mx = mlx::core;

static NSError * apSessionError(APErrorCode code, NSString * message) {
    return [NSError errorWithDomain:APErrorDomain code:code
                           userInfo:@{ NSLocalizedDescriptionKey : message }];
}

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

static std::string apRoleString(APRole role) {
    switch (role) {
        case APRoleSystem:    return "system";
        case APRoleUser:      return "user";
        case APRoleAssistant: return "assistant";
        case APRoleTool:      return "tool";
    }
    return "user";
}

static BOOL apTextOnly(APMessage * m) {
    for (APContent * c in m.content)
        if (c.kind != APContentKindText) return NO;
    return YES;
}

@implementation APSession {
    APModel * _model;
    std::unique_ptr<es::ESKVCache> _cache;   // touched ONLY on the engine thread
    int  _pos;
    int  _turnCount;
    BOOL _openModelTurn;     // last response did not close its turn (cancel/max-tokens)
    BOOL _warnedNearFull;
    NSMutableArray<APMessage *> * _transcript;
    NSMutableDictionary<NSString *, id<APTool>> * _tools;
    NSArray<NSNumber *> * _lastIds;
    BOOL _lastPrimeRestoredFromSnapshot;
}

- (BOOL)lastPrimeRestoredFromSnapshot {
    @synchronized(self) { return _lastPrimeRestoredFromSnapshot; }
}

- (instancetype)initWithModel:(APModel *)model {
    if ((self = [super init])) {
        _model = model;
        _cache = std::make_unique<es::ESKVCache>([model internalConfig]->numHiddenLayers);
        _pos = 0;
        _turnCount = 0;
        _openModelTurn = NO;
        _warnedNearFull = NO;
        _callbackQueue = dispatch_get_main_queue();
        _transcript = [NSMutableArray array];
        _tools = [NSMutableDictionary dictionary];
    }
    return self;
}

- (NSArray<APMessage *> *)transcript {
    @synchronized(self) { return [_transcript copy]; }
}

- (NSInteger)contextTokenCount {
    @synchronized(self) { return _pos; }
}

- (void)reset {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [_model performOnEngine:^{
        self->_cache->reset();
        @synchronized(self) {
            self->_pos = 0;
            self->_turnCount = 0;
            self->_openModelTurn = NO;
            self->_warnedNearFull = NO;
            [self->_transcript removeAllObjects];
            self->_lastIds = nil;
        }
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
}

- (NSInteger)contextLimit {
    NSInteger limit = [_model internalConfiguration].maximumContextLength;
    return limit > 0 ? MIN(limit, _model.maximumContextLength) : _model.maximumContextLength;
}

- (NSArray<NSNumber *> *)lastResponseTokenIDsForTesting {
    @synchronized(self) { return _lastIds; }
}

#pragma mark - Tools (registration only in v1; see APTool.h)

- (void)registerTool:(id<APTool>)tool {
    @synchronized(self) { _tools[tool.name] = tool; }
    if ([tool respondsToSelector:@selector(willAttachToSession:)])
        [tool willAttachToSession:self];
}

- (void)unregisterToolNamed:(NSString *)name {
    id<APTool> tool;
    @synchronized(self) { tool = _tools[name]; [_tools removeObjectForKey:name]; }
    if (tool && [tool respondsToSelector:@selector(didDetachFromSession:)])
        [tool didDetachFromSession:self];
}

#pragma mark - Helpers

- (void)deliver:(dispatch_block_t)block {
    dispatch_async(_callbackQueue ?: dispatch_get_main_queue(), block);
}

#pragma mark - Prime

/// Snapshot validity key: format version, model identity (name + weight byte count),
/// head precision, and the EXACT prime token ids (which transitively cover the persona
/// text, tokenizer, and chat-template layout). SHA-256, hex.
static std::string apSnapshotFingerprint(APModel * model, const std::vector<int> & ids) {
    NSMutableData * blob = [NSMutableData data];
    uint32_t version = 1;
    [blob appendBytes:&version length:sizeof(version)];
    NSData * name = [model.modelIdentifier dataUsingEncoding:NSUTF8StringEncoding];
    [blob appendData:name];
    unsigned long long wb = [model internalWeightBytes];
    [blob appendBytes:&wb length:sizeof(wb)];
    int64_t head = [model internalConfiguration].headBits;
    [blob appendBytes:&head length:sizeof(head)];
    [blob appendBytes:ids.data() length:ids.size() * sizeof(int)];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(blob.bytes, (CC_LONG) blob.length, digest);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    return std::string(hex, 2 * CC_SHA256_DIGEST_LENGTH);
}

- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                           completion:(void (^)(NSError *_Nullable))completion {
    return [self primeWithMessages:messages cacheURL:nil completion:completion];
}

- (APResponseTask *)primeWithMessages:(NSArray<APMessage *> *)messages
                             cacheURL:(NSURL *)cacheURL
                           completion:(void (^)(NSError *_Nullable))completion {
    NSProgress * progress = [NSProgress progressWithTotalUnitCount:-1];
    APResponseTask * task = [[APResponseTask alloc] initWithProgress:progress];
    [_model performOnEngine:^{
        for (APMessage * m in messages) {
            if (m.role == APRoleTool) {
                [self deliver:^{ completion(apSessionError(APErrorInvalidMessage,
                    @"tool messages are not supported in prime (v1)")); }];
                return;
            }
            if (!apTextOnly(m)) {
                [self deliver:^{ completion(apSessionError(APErrorUnsupportedContent,
                    @"this model accepts text content only")); }];
                return;
            }
        }
        try {
            std::vector<es::ESChatMessage> msgs;
            for (APMessage * m in messages)
                msgs.push_back({apRoleString(m.role), std::string(m.textRepresentation.UTF8String)});
            es::ESChatTemplate * chat = [self->_model internalTemplate];
            std::vector<int> ids = chat->build(msgs, /*think=*/false, /*addGen=*/false);
            if ((NSInteger)ids.size() + 64 > [self contextLimit]) {
                [self deliver:^{ completion(apSessionError(APErrorContextOverflow,
                    @"prime messages exceed the context limit")); }];
                return;
            }

            // Snapshot fast path: valid only on a fresh session; restored content is
            // byte-identical to a fresh prefill (--persist-verify), so continuation matches.
            BOOL restored = NO;
            std::string fingerprint;
            if (cacheURL && self->_pos == 0) {
                fingerprint = apSnapshotFingerprint(self->_model, ids);
                if ([NSFileManager.defaultManager fileExistsAtPath:cacheURL.path]) {
                    int pos = self->_cache->restoreSnapshot(std::string(cacheURL.path.UTF8String),
                                                            fingerprint);
                    if (pos == (int) ids.size()) {
                        restored = YES;
                    } else if (pos >= 0) {
                        self->_cache->reset();   // valid file, unexpected pos — refill cleanly
                    }
                }
            }

            if (!restored) {
                mx::array ll = [self->_model internalLM]->lastLogits(ids, self->_cache.get(), self->_pos);
                mx::eval(ll);
                if (cacheURL && self->_pos == 0) {   // best-effort write; priming already succeeded
                    self->_cache->saveSnapshot(std::string(cacheURL.path.UTF8String),
                                               fingerprint, (int) ids.size());
                }
            }
            @synchronized(self) {
                self->_pos += (int)ids.size();
                self->_lastPrimeRestoredFromSnapshot = restored;
                [self->_transcript addObjectsFromArray:messages];
            }
            [self deliver:^{ completion(nil); }];
        } catch (const std::exception & e) {
            NSError * err = apSessionError(APErrorEngineFailure, @(e.what()));
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
        if (message.role != APRoleUser) {
            [self deliver:^{ completion(nil, apSessionError(APErrorInvalidMessage,
                @"respond requires a user message (v1)")); }];
            return;
        }
        if (!apTextOnly(message)) {
            [self deliver:^{ completion(nil, apSessionError(APErrorUnsupportedContent,
                @"this model accepts text content only")); }];
            return;
        }
        try {
            [self runTurnWithMessage:message options:opts task:task progress:progress
                        deltaHandler:deltaHandler completion:completion];
        } catch (const std::exception & e) {
            NSError * err = apSessionError(APErrorEngineFailure, @(e.what()));
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
        std::vector<int> prefix = chat->build({}, /*think=*/false, /*addGen=*/false);
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
    push(d, tok->encode(std::string(message.textRepresentation.UTF8String), false));
    d.push_back(T.turnClose); push(d, enc("\n"));
    d.push_back(T.turnOpen); push(d, enc("model\n"));
    d.push_back(T.channelOpen); push(d, enc("thought\n")); d.push_back(T.channelClose);

    // ---- context pre-flight ----
    NSInteger limit = [self contextLimit];
    NSInteger maxNew = opts.maximumResponseTokens > 0 ? opts.maximumResponseTokens
                                                      : (limit - _pos - (NSInteger)d.size() - 2);
    if (_pos + (NSInteger)d.size() + 2 > limit || maxNew < 1) {
        [self deliver:^{ completion(nil, apSessionError(APErrorContextOverflow,
            @"context limit reached; reset the session or raise maximumContextLength")); }];
        return;
    }

    es::ESSamplingConfig sc;
    sc.greedy = (opts.temperature <= 0);
    sc.temperature = MAX(opts.temperature, 1e-6f);
    sc.topK = (int)opts.topK;
    sc.topP = opts.topP;
    sc.maxNewTokens = (int)maxNew;
    sc.eosTokenId = chat->stopToken();
    es::ESSampler sampler(sc);

    // ---- prefill the turn delta (mirrors ESSession::respond) ----
    NSDate * t0 = [NSDate date];
    mx::array ll = lm->lastLogits(d, _cache.get(), _pos);
    mx::eval(ll);
    NSTimeInterval prefillS = -[t0 timeIntervalSinceNow];
    @synchronized(self) { _pos += (int)d.size(); }

    std::vector<int> out;
    int next = sampler.sample(ll);
    out.push_back(next);
    NSTimeInterval ttft = -[t0 timeIntervalSinceNow];

    // ---- streaming decode ----
    std::string decoded;
    size_t emitted = 0;
    APFinishReason reason = APFinishReasonMaxTokens;
    auto emitDeltas = [&](NSInteger tokens) {
        decoded = tok->decode(out, /*skipSpecial=*/true);
        size_t safe = apCompleteUTF8PrefixLength(decoded);
        if (safe > emitted) {
            NSString * text = [[NSString alloc] initWithBytes:decoded.data() + emitted
                                                       length:safe - emitted
                                                     encoding:NSUTF8StringEncoding];
            emitted = safe;
            if (text.length > 0 && deltaHandler) {
                APResponseDelta * delta = [[APResponseDelta alloc] initWithText:text
                                                                     tokenCount:tokens];
                [self deliver:^{ deltaHandler(delta); }];
            }
        }
    };
    emitDeltas(1);
    progress.completedUnitCount = 1;

    NSDate * tDecode = [NSDate date];
    for (int s = 1; s < sc.maxNewTokens; ++s) {
        if (next == sc.eosTokenId) { reason = APFinishReasonEndOfTurn; break; }
        if ([task isCancelled])    { reason = APFinishReasonCancelled;  break; }
        ll = lm->lastLogits({next}, _cache.get(), _pos);
        mx::eval(ll);
        @synchronized(self) { _pos += 1; }
        next = sampler.sample(ll);
        out.push_back(next);
        progress.completedUnitCount = (int64_t)out.size();
        emitDeltas(1);
    }
    if (reason != APFinishReasonCancelled && next == sc.eosTokenId)
        reason = APFinishReasonEndOfTurn;   // eos sampled on the final permitted step
    NSTimeInterval decodeS = -[tDecode timeIntervalSinceNow];

    // Cache the final sampled token so the next turn attends the complete reply
    // (identical to ESSession::respond).
    if (!out.empty()) {
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

    // ---- finalize: parsed answer, stats, transcript ----
    es::ESParsedResponse parsed = chat->parse(out);
    NSString * answer = @(parsed.answer.c_str()) ?: @"";
    APMessage * reply = [APMessage assistantMessageWithText:answer];
    NSMutableArray<NSNumber *> * ids = [NSMutableArray arrayWithCapacity:out.size()];
    for (int t : out) [ids addObject:@(t)];
    @synchronized(self) {
        [_transcript addObject:message];
        [_transcript addObject:reply];
        _lastIds = ids;
    }

    APResponseStats * stats = [[APResponseStats alloc]
        initWithPromptTokens:(NSInteger)d.size()
              responseTokens:(NSInteger)out.size()
            timeToFirstToken:ttft
                   prefillTPS:(prefillS > 0 ? d.size() / prefillS : 0)
                    decodeTPS:(decodeS > 0 ? (out.size() > 1 ? (out.size() - 1) / decodeS : 0) : 0)];
    APResponse * response = [[APResponse alloc] initWithMessage:reply finishReason:reason stats:stats];
    [self deliver:^{ completion(response, nil); }];
}

@end
