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
#import "APSessionSubclass.h"
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

#pragma mark - Tool grammar rendering

/// Strings that reach the tool grammar must not smuggle marker text.
static NSString * apSanitizeForGrammar(NSString * s) {
    return [[s stringByReplacingOccurrencesOfString:@"<|" withString:@"< |"]
                stringByReplacingOccurrencesOfString:@"|>" withString:@"| >"];
}

/// Render a JSON-ish value into Gemma-4's compact tool grammar: bare keys, strings in
/// <|"|> quote markers (spliced to the single quote token by encodeQuoted), arrays and
/// objects in their plain bracketed forms.
static NSString * apCompactValue(id v) {
    if ([v isKindOfClass:NSString.class]) {
        return [NSString stringWithFormat:@"<|\"|>%@<|\"|>", apSanitizeForGrammar(v)];
    }
    if ([v isKindOfClass:NSNumber.class]) {
        NSNumber * n = v;
        if (n == (id) kCFBooleanTrue)  return @"true";
        if (n == (id) kCFBooleanFalse) return @"false";
        return n.stringValue;
    }
    if ([v isKindOfClass:NSArray.class]) {
        NSMutableArray * parts = [NSMutableArray array];
        for (id e in (NSArray *) v) [parts addObject:apCompactValue(e)];
        return [NSString stringWithFormat:@"[%@]", [parts componentsJoinedByString:@","]];
    }
    if ([v isKindOfClass:NSDictionary.class]) {
        NSDictionary * d = v;
        NSArray * keys = [d.allKeys sortedArrayUsingSelector:@selector(compare:)];
        NSMutableArray * parts = [NSMutableArray array];
        for (NSString * k in keys)
            [parts addObject:[NSString stringWithFormat:@"%@:%@", k, apCompactValue(d[k])]];
        return [NSString stringWithFormat:@"{%@}", [parts componentsJoinedByString:@","]];
    }
    return @"null";
}

/// The reference chat template renders JSON-Schema `type` VALUES uppercase in tool
/// declarations (type:<|"|>OBJECT<|"|>, STRING, ...) — `value['type'] | upper` in the
/// official Jinja. Apps hand us lowercase JSON-Schema; uppercase the type values (and
/// arrays of type names) here so the primed declaration matches what the model saw in
/// training. Everything else (descriptions, enums, property names) keeps its case.
static id apUppercaseSchemaTypes(id node) {
    if ([node isKindOfClass:NSDictionary.class]) {
        NSMutableDictionary * out = [NSMutableDictionary dictionaryWithCapacity:
                                     [(NSDictionary *)node count]];
        [(NSDictionary *)node enumerateKeysAndObjectsUsingBlock:^(NSString * key, id v, BOOL * stop) {
            if ([key isEqualToString:@"type"]) {
                if ([v isKindOfClass:NSString.class]) { out[key] = [v uppercaseString]; return; }
                if ([v isKindOfClass:NSArray.class]) {
                    NSMutableArray * types = [NSMutableArray array];
                    for (id t in (NSArray *)v)
                        [types addObject:[t isKindOfClass:NSString.class] ? [t uppercaseString] : t];
                    out[key] = types; return;
                }
            }
            out[key] = apUppercaseSchemaTypes(v);
        }];
        return out;
    }
    if ([node isKindOfClass:NSArray.class]) {
        NSMutableArray * out = [NSMutableArray arrayWithCapacity:[(NSArray *)node count]];
        for (id v in (NSArray *)node) [out addObject:apUppercaseSchemaTypes(v)];
        return out;
    }
    return node;
}

/// declaration:NAME{description:<|"|>...<|"|>,parameters:{...}} — the reference grammar
/// line the system turn advertises (the CLI --tools file format, generated).
static std::string apToolDeclaration(id<APTool> tool) {
    NSString * decl = [NSString stringWithFormat:@"declaration:%@{description:%@,parameters:%@}",
                       tool.name, apCompactValue(tool.toolDescription),
                       apCompactValue(apUppercaseSchemaTypes(tool.parameterSchema ?: @{}))];
    return std::string(decl.UTF8String);
}

/// JSON-escape a <|"|>-span's contents so multi-line prose survives NSJSONSerialization.
static NSString * apJSONEscape(NSString * s) {
    NSMutableString * out = [NSMutableString stringWithCapacity:s.length + 16];
    for (NSUInteger i = 0; i < s.length; i++) {
        unichar c = [s characterAtIndex:i];
        switch (c) {
            case '\\': [out appendString:@"\\\\"]; break;
            case '"':  [out appendString:@"\\\""]; break;
            case '\n': [out appendString:@"\\n"];  break;
            case '\r': [out appendString:@"\\r"];  break;
            case '\t': [out appendString:@"\\t"];  break;
            default:
                if (c < 0x20) [out appendFormat:@"\\u%04x", c];
                else [out appendFormat:@"%C", c];
        }
    }
    return out;
}

/// Best-effort parse of the model's compact argument body `key:val,...` into a
/// dictionary. The body alternates structure and <|"|>-quoted string spans; span
/// contents are JSON-escaped verbatim (newlines, quotes, braces and all), bare keys in
/// the structure parts get quoted, then JSON parses. Falls back to {"_raw": body} so
/// tools always receive something actionable.
static NSDictionary<NSString *, id> * apParseToolArguments(NSString * rawBody) {
    NSArray<NSString *> * parts =
        [(rawBody ?: @"") componentsSeparatedByString:@"<|\"|>"];
    BOOL needsBraces = ![[parts[0] stringByTrimmingCharactersInSet:
                          NSCharacterSet.whitespaceAndNewlineCharacterSet] hasPrefix:@"{"];
    NSRegularExpression * bareKey =
        [NSRegularExpression regularExpressionWithPattern:@"([{,]\\s*)([A-Za-z_][A-Za-z0-9_]*)\\s*:"
                                                  options:0 error:nil];
    NSMutableString * jsonish = [NSMutableString string];
    if (needsBraces) [jsonish appendString:@"{"];
    for (NSUInteger i = 0; i < parts.count; i++) {
        if (i % 2) {   // inside a quoted span: verbatim string content
            [jsonish appendFormat:@"\"%@\"", apJSONEscape(parts[i])];
        } else {       // structure: quote bare keys, leave everything else alone
            NSString * st = [bareKey stringByReplacingMatchesInString:parts[i] options:0
                                                                range:NSMakeRange(0, parts[i].length)
                                                         withTemplate:@"$1\"$2\":"];
            if (i == 0 && needsBraces) {   // key at the very start has no [{,] anchor
                NSRange first = [st rangeOfString:@"^\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*:"
                                          options:NSRegularExpressionSearch];
                if (first.location != NSNotFound) {
                    NSString * head = [st substringWithRange:first];
                    NSString * name = [[head stringByReplacingOccurrencesOfString:@":" withString:@""]
                        stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
                    st = [NSString stringWithFormat:@"\"%@\":%@", name,
                          [st substringFromIndex:NSMaxRange(first)]];
                }
            }
            [jsonish appendString:st];
        }
    }
    if (needsBraces) [jsonish appendString:@"}"];
    NSData * data = [jsonish dataUsingEncoding:NSUTF8StringEncoding];
    id parsed = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    if ([parsed isKindOfClass:NSDictionary.class]) return parsed;
    return @{ @"_raw" : rawBody ?: @"" };
}

@implementation APLocalSession {
    APModel * _model;
    std::unique_ptr<es::ESKVCache> _cache;   // touched ONLY on the engine thread
    int  _pos;
    int  _turnCount;
    BOOL _openModelTurn;     // last response did not close its turn (cancel/max-tokens)
    BOOL _warnedNearFull;
    NSArray<NSNumber *> * _lastIds;
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
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [_model performOnEngine:^{
        self->_cache->reset();
        @synchronized(self) {
            self->_pos = 0;
            self->_turnCount = 0;
            self->_openModelTurn = NO;
            self->_warnedNearFull = NO;
            self->_lastIds = nil;
        }
        [self clearTranscript];
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
            // Registered tools are advertised in the system turn (register BEFORE prime).
            // Sorted by name so the prime ids — and the snapshot fingerprint — are stable.
            std::vector<std::string> decls;
            for (id<APTool> tool in [self toolsSortedByName])
                decls.push_back(apToolDeclaration(tool));
            std::vector<int> ids = chat->build(msgs, /*think=*/self.reasoningEnabled, /*addGen=*/false, decls);
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
            @synchronized(self) { self->_pos += (int)ids.size(); }
            [self noteLastPrimeRestoredFromSnapshot:restored];
            [self appendTranscriptMessages:messages];
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
    push(d, tok->encode(std::string(message.textRepresentation.UTF8String), false));
    d.push_back(T.turnClose); push(d, enc("\n"));
    d.push_back(T.turnOpen); push(d, enc("model\n"));
    // Reasoning off: pre-close an EMPTY thought channel (suppresses reasoning).
    // Reasoning on: leave the model turn open — the model writes its own thought channel.
    const BOOL think = self.reasoningEnabled;
    if (!think) { d.push_back(T.channelOpen); push(d, enc("thought\n")); d.push_back(T.channelClose); }

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
    NSMutableArray<NSString *> * executedTools = [NSMutableArray array];
    const int kMaxToolRounds = 4;

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
        if (chunk.empty() || !deltaHandler) return;
        NSString * text = [[NSString alloc] initWithBytes:chunk.data() length:chunk.size()
                                                 encoding:NSUTF8StringEncoding];
        if (text.length == 0) return;
        const BOOL thought = inThought;
        APResponseDelta * delta = [[APResponseDelta alloc] initWithText:text
                                                             tokenCount:tokens
                                                              isThought:thought];
        [self deliver:^{ deltaHandler(delta); }];
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
            if (sawToolCallClose && activeTools.count && toolRounds < kMaxToolRounds) break;
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

        if (!(sawToolCallClose && activeTools.count && toolRounds < kMaxToolRounds) ||
            reason == APFinishReasonEndOfTurn || reason == APFinishReasonCancelled) break;
        sawToolCallClose = false;
        toolRounds += 1;

        // ---- dispatch the call (the LAST parsed call is this round's) ----
        es::ESParsedResponse midParse = chat->parse(out);
        if (midParse.toolCalls.empty()) continue;        // defensive: nothing to dispatch
        const es::ESToolCall & call = midParse.toolCalls.back();
        NSString * callName = @(call.name.c_str()) ?: @"";
        NSDictionary * args = apParseToolArguments(@(call.args.c_str()) ?: @"");
        id<APTool> tool = nil;
        for (id<APTool> t in activeTools)
            if ([t.name isEqualToString:callName]) { tool = t; break; }

        __block NSString * resultText = nil;
        __block NSError * toolError = nil;
        id<APSessionDelegate> delegate = self.delegate;
        BOOL allowed = tool != nil;
        if (allowed && [delegate respondsToSelector:@selector(session:shouldInvokeTool:arguments:)])
            allowed = [delegate session:self shouldInvokeTool:tool arguments:args];
        if (allowed) {
            // Tools do IO: block THIS engine thread until the completion fires (other
            // sessions on the same model wait — documented serialization).
            dispatch_semaphore_t toolSem = dispatch_semaphore_create(0);
            [tool invokeWithArguments:args completion:^(APContent * result, NSError * error) {
                resultText = result.text;
                toolError = error;
                dispatch_semaphore_signal(toolSem);
            }];
            dispatch_semaphore_wait(toolSem, DISPATCH_TIME_FOREVER);
            [executedTools addObject:callName];
        }
        NSString * body = toolError ? [NSString stringWithFormat:@"{error:<|\"|>%@<|\"|>}",
                                       apSanitizeForGrammar(toolError.localizedDescription)]
                        : !allowed  ? @"{error:<|\"|>tool unavailable<|\"|>}"
                                    : [NSString stringWithFormat:@"{result:<|\"|>%@<|\"|>}",
                                       apSanitizeForGrammar(resultText ?: @"")];
        if ([delegate respondsToSelector:@selector(session:didInvokeTool:arguments:result:)]) {
            NSString * info = allowed ? (resultText ?: toolError.localizedDescription ?: @"") : @"(vetoed)";
            [self deliver:^{ [delegate session:self didInvokeTool:callName arguments:args result:info]; }];
        }

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

    // ---- finalize: parsed answer + reasoning, stats, transcript ----
    es::ESParsedResponse parsed = chat->parse(out);
    NSString * answer = @(parsed.answer.c_str()) ?: @"";
    NSString * reasoning = parsed.thought.empty() ? nil : (@(parsed.thought.c_str()) ?: nil);
    APMessage * reply = [APMessage assistantMessageWithText:answer];
    NSMutableArray<NSNumber *> * ids = [NSMutableArray arrayWithCapacity:out.size()];
    for (int t : out) [ids addObject:@(t)];
    [self appendTranscriptMessages:@[ message, reply ]];
    @synchronized(self) { _lastIds = ids; }

    APResponseStats * stats = [[APResponseStats alloc]
        initWithPromptTokens:(NSInteger)d.size()
              responseTokens:(NSInteger)out.size()
            timeToFirstToken:ttft
                   prefillTPS:(prefillS > 0 ? d.size() / prefillS : 0)
                    decodeTPS:(decodeS > 0 ? (out.size() > 1 ? (out.size() - 1) / decodeS : 0) : 0)];
    APResponse * response = [[APResponse alloc] initWithMessage:reply finishReason:reason
                                                          stats:stats reasoning:reasoning
                                              executedToolNames:executedTools];
    [self deliver:^{ completion(response, nil); }];
}

@end
