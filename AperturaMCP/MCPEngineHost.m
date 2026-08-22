//
//  MCPEngineHost.m
//  apertura-mcp
//

#import "MCPEngineHost.h"
#import <AperturaKit/AperturaKit.h>
#import "APModelRegistry.h"
#import "APPersistence.h"
#import "CDChatSession.h"
#import "CDPersona.h"

static NSError * apHostError(NSString * text) {
    return [NSError errorWithDomain:@"com.elarity.apertura-mcp" code:1
                           userInfo:@{ NSLocalizedDescriptionKey : text }];
}

/// Compact JSON for tool results (sorted keys: stable output for tests and diffs).
static NSString * apJSON(id object) {
    NSData * data = [NSJSONSerialization dataWithJSONObject:object
                                                    options:NSJSONWritingSortedKeys
                                                      error:nil];
    return data ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"{}";
}

static NSString * apISO(NSDate * date) {
    if (!date) return nil;
    static NSISO8601DateFormatter * f;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ f = [[NSISO8601DateFormatter alloc] init]; });
    return [f stringFromDate:date];
}

@interface MCPLiveSession : NSObject
@property (nonatomic) APLocalSession * session;
@property (nonatomic) CDChatSession * row;          // background-context object
@property (nonatomic, copy) NSString * personaName;
@end
@implementation MCPLiveSession
@end

@implementation MCPEngineHost {
    APModel * _model;
    NSMutableDictionary<NSString *, MCPLiveSession *> * _sessions;
    NSManagedObjectContext * _moc;                  // background context; stdio thread waits
    dispatch_queue_t _callbackQueue;                // session callbacks (never main: no runloop)
}

- (instancetype)init {
    if ((self = [super init])) {
        _sessions = [NSMutableDictionary dictionary];
        _callbackQueue = dispatch_queue_create("apertura-mcp.session-callbacks",
                                               DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (NSManagedObjectContext *)moc {
    if (!_moc) {
        _moc = [APPersistence.sharedContainer newBackgroundContext];
        _moc.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy;
    }
    return _moc;
}

- (void)shutdown {
    NSManagedObjectContext * moc = _moc;
    [moc performBlockAndWait:^{ [moc save:nil]; }];
}

#pragma mark - Model

- (APModel *)modelOrError:(NSError **)error {
    if (_model) return _model;
    NSURL * url = [APModelRegistry resolvedModelURL];
    if (!url) {
        if (error) *error = apHostError(@"no model registered — use load_model with a path, "
                                        @"or register one in the app");
        return nil;
    }
    return [self loadModelAtURL:url configuration:[APModelRegistry configurationForResolvedModel]
                          error:error];
}

- (APModel *)loadModelAtURL:(NSURL *)url configuration:(APModelConfiguration *)config
                      error:(NSError **)error {
    fprintf(stderr, "[apertura-mcp] loading %s…\n", url.lastPathComponent.UTF8String);
    NSDate * t0 = [NSDate date];
    APModel * model = [APModel modelWithContentsOfURL:url configuration:config error:error];
    if (model) {
        fprintf(stderr, "[apertura-mcp] loaded in %.1fs (context %ld)\n",
                -[t0 timeIntervalSinceNow], (long)model.maximumContextLength);
        _model = model;
        // Sessions from a previous model are meaningless against new weights.
        [_sessions removeAllObjects];
    }
    return model;
}

#pragma mark - Persona helpers (background context)

- (CDPersona *)personaForArguments:(NSDictionary *)args error:(NSError **)error {
    __block CDPersona * persona = nil;
    NSManagedObjectContext * moc = [self moc];
    [moc performBlockAndWait:^{
        NSString * pid = [args[@"persona_id"] isKindOfClass:NSString.class] ? args[@"persona_id"] : nil;
        NSString * pname = [args[@"persona_name"] isKindOfClass:NSString.class] ? args[@"persona_name"] : nil;
        if (pid) {
            NSUUID * uuid = [[NSUUID alloc] initWithUUIDString:pid];
            persona = uuid ? [CDPersona personaWithIdentifier:uuid inContext:moc error:nil] : nil;
            return;
        }
        NSArray<CDPersona *> * heads =
            [moc executeFetchRequest:CDPersona.currentPersonasFetchRequest error:nil];
        if (pname) {
            for (CDPersona * head in heads)
                if ([head.name localizedCaseInsensitiveCompare:pname] == NSOrderedSame) {
                    persona = head; return;
                }
            return;
        }
        for (CDPersona * head in heads)
            if (head.body.length) { persona = head; return; }
    }];
    if (!persona && error) *error = apHostError(@"persona not found");
    return persona;
}

#pragma mark - Blocking bridges (semaphores wait on the stdio thread)

- (NSError *)primeSession:(APLocalSession *)session messages:(NSArray<APMessage *> *)messages
                 cacheURL:(NSURL *)cacheURL {
    __block NSError * result = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [session primeWithMessages:messages cacheURL:cacheURL completion:^(NSError * error) {
        result = error;
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    return result;
}

- (APResponse *)respondIn:(APLocalSession *)session message:(APMessage *)message
                  options:(APGenerationOptions *)options error:(NSError **)error {
    __block APResponse * response = nil;
    __block NSError * responseError = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [session respondToMessage:message options:options deltaHandler:nil
                   completion:^(APResponse * r, NSError * e) {
        response = r;
        responseError = e;
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    if (!response && error) *error = responseError ?: apHostError(@"generation failed");
    return response;
}

#pragma mark - Tools

- (void)registerToolsInto:(MCPToolRegistry *)registry {
    __weak typeof(self) weakSelf = self;

    [registry registerToolNamed:@"list_models"
        description:@"List registered model bundles (name, size, active flag) and which one resolves for loading."
        schema:@{ @"type" : @"object", @"properties" : @{} }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            NSMutableArray * models = [NSMutableArray array];
            NSString * selected = [APModelRegistry selectedModelName];
            for (APInstalledModel * m in [APModelRegistry installedModels])
                [models addObject:@{ @"name" : m.name,
                                     @"size_bytes" : @(m.sizeBytes),
                                     @"symlink" : @(m.isSymlink),
                                     @"active" : @([m.name isEqualToString:selected]) }];
            return apJSON(@{ @"models" : models,
                             @"resolved" : [APModelRegistry resolvedModelURL].path ?: @"" });
        }];

    [registry registerToolNamed:@"load_model"
        description:@"Load a model (blocking, tens of seconds). Default: the registry's resolved model. Optional path overrides; head_bits (8|4), cache_mode (0 standard | 1 raw | 2 raw-q8), prefill_chunk, max_context override the stored configuration."
        schema:@{ @"type" : @"object", @"properties" : @{
            @"path" : @{ @"type" : @"string" },
            @"head_bits" : @{ @"type" : @"integer" },
            @"cache_mode" : @{ @"type" : @"integer" },
            @"prefill_chunk" : @{ @"type" : @"integer" },
            @"max_context" : @{ @"type" : @"integer" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            NSString * path = [args[@"path"] isKindOfClass:NSString.class] ? args[@"path"] : nil;
            NSURL * url = path ? [NSURL fileURLWithPath:path] : [APModelRegistry resolvedModelURL];
            if (!url) { if (error) *error = apHostError(@"no model to load"); return nil; }
            APModelConfiguration * config =
                [APModelRegistry configurationForModelNamed:url.lastPathComponent];
            if (args[@"head_bits"])     config.headBits = [args[@"head_bits"] integerValue];
            if (args[@"cache_mode"])    config.globalKVCacheMode = [args[@"cache_mode"] integerValue];
            if (args[@"prefill_chunk"]) config.prefillChunkLength = [args[@"prefill_chunk"] integerValue];
            if (args[@"max_context"])   config.maximumContextLength = [args[@"max_context"] integerValue];
            NSDate * t0 = [NSDate date];
            APModel * model = [self loadModelAtURL:url configuration:config error:error];
            if (!model) return nil;
            return apJSON(@{ @"model" : model.modelIdentifier,
                             @"max_context" : @(model.maximumContextLength),
                             @"load_seconds" : @(-[t0 timeIntervalSinceNow]) });
        }];

    [registry registerToolNamed:@"list_personas"
        description:@"List the stored custom GPTs (current versions): id, name, body length, version count."
        schema:@{ @"type" : @"object", @"properties" : @{} }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            __block NSArray * payload = nil;
            NSManagedObjectContext * moc = [self moc];
            [moc performBlockAndWait:^{
                NSMutableArray * list = [NSMutableArray array];
                for (CDPersona * head in
                     [moc executeFetchRequest:CDPersona.currentPersonasFetchRequest error:nil])
                    [list addObject:@{ @"id" : head.identifier.UUIDString ?: @"",
                                       @"name" : head.name ?: @"",
                                       @"body_length" : @(head.body.length),
                                       @"versions" : @([head versionChain].count) }];
                payload = list;
            }];
            return apJSON(@{ @"personas" : payload });
        }];

    [registry registerToolNamed:@"get_persona"
        description:@"Read a persona's full body (by persona_id or persona_name; default = the first with a body)."
        schema:@{ @"type" : @"object", @"properties" : @{
            @"persona_id" : @{ @"type" : @"string" },
            @"persona_name" : @{ @"type" : @"string" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            CDPersona * persona = [self personaForArguments:args error:error];
            if (!persona) return nil;
            __block NSString * result = nil;
            [[self moc] performBlockAndWait:^{
                result = apJSON(@{ @"id" : persona.identifier.UUIDString ?: @"",
                                   @"name" : persona.name ?: @"",
                                   @"sha256" : persona.sha256 ?: @"",
                                   @"body" : persona.body ?: @"" });
            }];
            return result;
        }];

    [registry registerToolNamed:@"create_persona"
        description:@"Create a custom GPT with a name and body (the standing prefix; join sections with \\n\\n---\\n\\n)."
        schema:@{ @"type" : @"object",
                  @"required" : @[ @"name", @"body" ],
                  @"properties" : @{
            @"name" : @{ @"type" : @"string" },
            @"body" : @{ @"type" : @"string" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            NSString * name = [args[@"name"] isKindOfClass:NSString.class] ? args[@"name"] : nil;
            NSString * body = [args[@"body"] isKindOfClass:NSString.class] ? args[@"body"] : nil;
            if (!name.length || !body.length) {
                if (error) *error = apHostError(@"create_persona needs name and body");
                return nil;
            }
            __block NSString * result = nil;
            NSManagedObjectContext * moc = [self moc];
            [moc performBlockAndWait:^{
                CDPersona * persona = [CDPersona insertInContext:moc];
                persona.name = name;
                [persona updateBody:body];
                [moc save:nil];
                result = apJSON(@{ @"id" : persona.identifier.UUIDString ?: @"",
                                   @"name" : name, @"body_length" : @(body.length) });
            }];
            return result;
        }];

    [registry registerToolNamed:@"update_persona"
        description:@"Replace a persona's body (archive-first: the old version is kept in its history chain)."
        schema:@{ @"type" : @"object",
                  @"required" : @[ @"body" ],
                  @"properties" : @{
            @"persona_id" : @{ @"type" : @"string" },
            @"persona_name" : @{ @"type" : @"string" },
            @"body" : @{ @"type" : @"string" },
            @"note" : @{ @"type" : @"string" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            NSString * body = [args[@"body"] isKindOfClass:NSString.class] ? args[@"body"] : nil;
            if (!body.length) { if (error) *error = apHostError(@"update_persona needs body"); return nil; }
            CDPersona * persona = [self personaForArguments:args error:error];
            if (!persona) return nil;
            __block NSString * result = nil;
            NSManagedObjectContext * moc = [self moc];
            [moc performBlockAndWait:^{
                [persona snapshotBeforeEditWithNote:
                    ([args[@"note"] isKindOfClass:NSString.class] ? args[@"note"] : @"updated")
                                             author:@"mcp"];
                [persona updateBody:body];
                [moc save:nil];
                result = apJSON(@{ @"id" : persona.identifier.UUIDString ?: @"",
                                   @"versions" : @([persona versionChain].count) });
            }];
            return result;
        }];

    [registry registerToolNamed:@"list_sessions"
        description:@"List stored conversations, newest first: id, title, turns, context tokens, checkpoint state."
        schema:@{ @"type" : @"object", @"properties" : @{
            @"limit" : @{ @"type" : @"integer" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            __block NSArray * payload = nil;
            NSManagedObjectContext * moc = [self moc];
            [moc performBlockAndWait:^{
                NSFetchRequest<CDChatSession *> * request = CDChatSession.recentSessionsFetchRequest;
                NSInteger limit = [args[@"limit"] integerValue];
                request.fetchLimit = limit > 0 ? (NSUInteger)limit : 50;
                NSMutableArray * list = [NSMutableArray array];
                for (CDChatSession * row in [moc executeFetchRequest:request error:nil])
                    [list addObject:@{ @"id" : row.identifier.UUIDString ?: @"",
                                       @"title" : row.title ?: @"",
                                       @"turns" : @(row.messageCount),
                                       @"context_tokens" : @(row.contextTokenCount),
                                       @"backend" : row.backend ?: @"",
                                       @"modified" : apISO(row.dateModified) ?: @"",
                                       @"checkpoint" : row.checkpointDate ? @YES : @NO,
                                       @"persona" : row.persona.name ?: @"" }];
                payload = list;
            }];
            return apJSON(@{ @"sessions" : payload });
        }];

    [registry registerToolNamed:@"create_session"
        description:@"Open a live local session primed with a persona (persona_id/persona_name; default = first with a body). Returns session_id. Uses the shared persona KV snapshot when one exists — sub-second start."
        schema:@{ @"type" : @"object", @"properties" : @{
            @"persona_id" : @{ @"type" : @"string" },
            @"persona_name" : @{ @"type" : @"string" },
            @"reasoning" : @{ @"type" : @"boolean" },
            @"excludes_reasoning_from_context" : @{ @"type" : @"boolean" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            APModel * model = [self modelOrError:error];
            if (!model) return nil;
            CDPersona * persona = [self personaForArguments:args error:error];
            if (!persona) return nil;
            __block NSString * personaBody = nil, * personaName = nil;
            __block NSUUID * personaID = nil;
            [[self moc] performBlockAndWait:^{
                personaBody = persona.body;
                personaName = persona.name;
                personaID = persona.identifier;
            }];
            if (!personaBody.length) { if (error) *error = apHostError(@"persona body empty"); return nil; }

            BOOL reasoning = args[@"reasoning"] ? [args[@"reasoning"] boolValue] : YES;
            APLocalSession * session = [[APLocalSession alloc] initWithModel:model];
            session.callbackQueue = self->_callbackQueue;
            session.reasoningEnabled = reasoning;
            session.excludesReasoningFromContext =
                args[@"excludes_reasoning_from_context"]
                    ? [args[@"excludes_reasoning_from_context"] boolValue] : YES;

            // The SAME persona snapshot convention the app uses — a persona the app
            // already primed starts here in under a second (and vice versa).
            NSString * snapName = [NSString stringWithFormat:@"persona-%@-%@.safetensors",
                                   personaID.UUIDString, reasoning ? @"think" : @"plain"];
            NSURL * cacheURL = [[APModelRegistry checkpointsDirectory]
                                   URLByAppendingPathComponent:snapName];
            NSDate * t0 = [NSDate date];
            NSError * primeError = [self primeSession:session
                messages:@[ [APMessage systemMessageWithText:personaBody] ] cacheURL:cacheURL];
            if (primeError) { if (error) *error = primeError; return nil; }

            MCPLiveSession * live = [[MCPLiveSession alloc] init];
            live.session = session;
            live.personaName = personaName;
            __block NSString * rowID = nil;
            NSManagedObjectContext * moc = [self moc];
            [moc performBlockAndWait:^{
                CDChatSession * row = [CDChatSession insertInContext:moc];
                [row recordPersonaText:personaBody atPath:nil];
                row.persona = persona;
                row.reasoningEnabled = reasoning;
                row.backend = CDChatSessionBackendLocal;
                row.modelIdentifier = model.modelIdentifier;
                [moc save:nil];
                live.row = row;
                rowID = row.identifier.UUIDString;
            }];
            self->_sessions[rowID] = live;
            return apJSON(@{ @"session_id" : rowID,
                             @"persona" : personaName ?: @"",
                             @"context_tokens" : @(session.contextTokenCount),
                             @"restored_from_snapshot" : @(session.lastPrimeRestoredFromSnapshot),
                             @"prime_seconds" : @(-[t0 timeIntervalSinceNow]) });
        }];

    [registry registerToolNamed:@"send_message"
        description:@"Send a user message to a live session (blocking; a long reply takes its decode time). Returns answer, reasoning, and stats. Options override the defaults per call."
        schema:@{ @"type" : @"object",
                  @"required" : @[ @"session_id", @"text" ],
                  @"properties" : @{
            @"session_id" : @{ @"type" : @"string" },
            @"text" : @{ @"type" : @"string" },
            @"temperature" : @{ @"type" : @"number" },
            @"top_k" : @{ @"type" : @"integer" },
            @"top_p" : @{ @"type" : @"number" },
            @"seed" : @{ @"type" : @"integer" },
            @"max_tokens" : @{ @"type" : @"integer" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            NSString * sid = [args[@"session_id"] isKindOfClass:NSString.class] ? args[@"session_id"] : nil;
            MCPLiveSession * live = sid ? self->_sessions[sid] : nil;
            if (!live) { if (error) *error = apHostError(@"unknown session_id (create_session first)"); return nil; }
            NSString * text = [args[@"text"] isKindOfClass:NSString.class] ? args[@"text"] : nil;
            if (!text.length) { if (error) *error = apHostError(@"empty text"); return nil; }

            APGenerationOptions * options = [APGenerationOptions defaultOptions];
            if (args[@"temperature"]) options.temperature = [args[@"temperature"] floatValue];
            if (args[@"top_k"])      options.topK = [args[@"top_k"] integerValue];
            if (args[@"top_p"])      options.topP = [args[@"top_p"] floatValue];
            if (args[@"seed"])       options.seed = (unsigned long long)[args[@"seed"] longLongValue];
            options.maximumResponseTokens = args[@"max_tokens"]
                ? [args[@"max_tokens"] integerValue]
                : (live.session.reasoningEnabled ? 2048 : 1024);

            APResponse * response = [self respondIn:live.session
                                            message:[APMessage userMessageWithText:text]
                                            options:options error:error];
            if (!response) return nil;

            NSManagedObjectContext * moc = [self moc];
            [moc performBlockAndWait:^{
                [live.row captureSession:live.session model:self->_model];
                [moc save:nil];
            }];
            return apJSON(@{ @"answer" : response.message.textRepresentation ?: @"",
                             @"reasoning" : response.reasoning ?: @"",
                             @"finish" : @(response.finishReason),
                             @"stats" : @{
                                 @"prompt_tokens" : @(response.stats.promptTokenCount),
                                 @"response_tokens" : @(response.stats.responseTokenCount),
                                 @"ttft_seconds" : @(response.stats.timeToFirstToken),
                                 @"prefill_tps" : @(response.stats.prefillTokensPerSecond),
                                 @"decode_tps" : @(response.stats.decodeTokensPerSecond) },
                             @"context_tokens" : @(live.session.contextTokenCount) });
        }];

    [registry registerToolNamed:@"save_checkpoint"
        description:@"Save a live session's KV cache to its conversation's checkpoint file (multi-GB at depth; blocking)."
        schema:@{ @"type" : @"object",
                  @"required" : @[ @"session_id" ],
                  @"properties" : @{ @"session_id" : @{ @"type" : @"string" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            NSString * sid = [args[@"session_id"] isKindOfClass:NSString.class] ? args[@"session_id"] : nil;
            MCPLiveSession * live = sid ? self->_sessions[sid] : nil;
            if (!live) { if (error) *error = apHostError(@"unknown session_id"); return nil; }
            __block NSURL * url = nil;
            __block NSUUID * rowUUID = nil;
            NSManagedObjectContext * moc = [self moc];
            [moc performBlockAndWait:^{ url = live.row.checkpointURL; rowUUID = live.row.identifier; }];
            BOOL saved = [live.session saveCheckpointToURL:url sessionID:rowUUID];
            if (!saved) { if (error) *error = apHostError(@"checkpoint save failed"); return nil; }
            NSDictionary * attrs = [NSFileManager.defaultManager attributesOfItemAtPath:url.path
                                                                                  error:nil];
            [moc performBlockAndWait:^{
                live.row.checkpointDate = NSDate.date;
                live.row.checkpointBytes = (int64_t)[attrs[NSFileSize] unsignedLongLongValue];
                [moc save:nil];
            }];
            return apJSON(@{ @"saved" : @YES, @"bytes" : attrs[NSFileSize] ?: @0,
                             @"path" : url.path });
        }];

    [registry registerToolNamed:@"restore_checkpoint"
        description:@"Reopen a stored conversation from its checkpoint into a fresh live session (falls back with an error when none matches; then create_session + replay is the slow path). Returns a session_id."
        schema:@{ @"type" : @"object",
                  @"required" : @[ @"session_id" ],
                  @"properties" : @{ @"session_id" : @{ @"type" : @"string" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            APModel * model = [self modelOrError:error];
            if (!model) return nil;
            NSString * sid = [args[@"session_id"] isKindOfClass:NSString.class] ? args[@"session_id"] : nil;
            NSUUID * uuid = sid ? [[NSUUID alloc] initWithUUIDString:sid] : nil;
            if (!uuid) { if (error) *error = apHostError(@"bad session_id"); return nil; }

            __block CDChatSession * row = nil;
            __block NSArray<APMessage *> * prime = nil;
            __block BOOL reasoning = NO;
            __block NSURL * url = nil;
            __block NSString * personaName = nil;
            NSManagedObjectContext * moc = [self moc];
            [moc performBlockAndWait:^{
                row = [CDChatSession sessionWithIdentifier:uuid inContext:moc error:nil];
                if (!row) return;
                NSMutableArray<APMessage *> * messages = [NSMutableArray array];
                if (row.personaText.length)
                    [messages addObject:[APMessage systemMessageWithText:row.personaText]];
                [messages addObjectsFromArray:row.messages];
                prime = messages;
                reasoning = row.reasoningEnabled;
                url = row.checkpointURL;
                personaName = row.persona.name;
            }];
            if (!row) { if (error) *error = apHostError(@"no such conversation"); return nil; }

            APLocalSession * session = [[APLocalSession alloc] initWithModel:model];
            session.callbackQueue = self->_callbackQueue;
            session.reasoningEnabled = reasoning;
            __block NSError * restoreError = nil;
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [session restoreCheckpointFromURL:url sessionID:uuid messages:prime
                                   completion:^(NSError * e) {
                restoreError = e;
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            if (restoreError) { if (error) *error = restoreError; return nil; }

            MCPLiveSession * live = [[MCPLiveSession alloc] init];
            live.session = session;
            live.row = row;
            live.personaName = personaName;
            self->_sessions[uuid.UUIDString] = live;
            return apJSON(@{ @"session_id" : uuid.UUIDString,
                             @"context_tokens" : @(session.contextTokenCount) });
        }];

    [registry registerToolNamed:@"tokenize"
        description:@"Tokenize text with the loaded model's tokenizer. Returns the count, and the ids for short inputs."
        schema:@{ @"type" : @"object",
                  @"required" : @[ @"text" ],
                  @"properties" : @{ @"text" : @{ @"type" : @"string" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            APModel * model = [self modelOrError:error];
            if (!model) return nil;
            NSString * text = [args[@"text"] isKindOfClass:NSString.class] ? args[@"text"] : @"";
            NSArray<NSNumber *> * ids = [model tokenizeText:text];
            return apJSON(ids.count <= 512 ? @{ @"count" : @(ids.count), @"ids" : ids }
                                           : @{ @"count" : @(ids.count) });
        }];

    [registry registerToolNamed:@"bench_decode"
        description:@"Decode benchmark: greedy-generate N tokens (default 64) in a scratch session and report tok/s."
        schema:@{ @"type" : @"object", @"properties" : @{
            @"tokens" : @{ @"type" : @"integer" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            APModel * model = [self modelOrError:error];
            if (!model) return nil;
            NSInteger tokens = MAX([args[@"tokens"] integerValue], 16);
            APLocalSession * scratch = [[APLocalSession alloc] initWithModel:model];
            scratch.callbackQueue = self->_callbackQueue;
            NSError * primeError = [self primeSession:scratch
                messages:@[ [APMessage systemMessageWithText:@"You are a benchmark. Repeat the word token forever."] ]
                cacheURL:nil];
            if (primeError) { if (error) *error = primeError; return nil; }
            APGenerationOptions * options = [APGenerationOptions deterministicOptions];
            options.maximumResponseTokens = tokens;
            APResponse * response = [self respondIn:scratch
                message:[APMessage userMessageWithText:@"Begin."] options:options error:error];
            if (!response) return nil;
            return apJSON(@{ @"decode_tokens" : @(response.stats.responseTokenCount),
                             @"decode_tps" : @(response.stats.decodeTokensPerSecond),
                             @"ttft_seconds" : @(response.stats.timeToFirstToken) });
        }];

    [registry registerToolNamed:@"bench_prefill"
        description:@"Prefill benchmark: prime a scratch session with ~N tokens of filler (default 2048) and report tok/s."
        schema:@{ @"type" : @"object", @"properties" : @{
            @"tokens" : @{ @"type" : @"integer" } } }
        handler:^NSString *(NSDictionary * args, NSError ** error) {
            typeof(self) self = weakSelf;
            APModel * model = [self modelOrError:error];
            if (!model) return nil;
            NSInteger target = MAX([args[@"tokens"] integerValue], 256);
            // "lorem " is stable filler; ~1.3 tokens a word on this tokenizer.
            NSMutableString * filler = [NSMutableString string];
            while ([model tokenCountForText:filler] < target) {
                NSInteger deficit = target - [model tokenCountForText:filler];
                for (NSInteger i = 0; i < MAX(deficit, 64); i++) [filler appendString:@"lorem "];
            }
            APLocalSession * scratch = [[APLocalSession alloc] initWithModel:model];
            scratch.callbackQueue = self->_callbackQueue;
            NSDate * t0 = [NSDate date];
            NSError * primeError = [self primeSession:scratch
                messages:@[ [APMessage systemMessageWithText:filler] ] cacheURL:nil];
            if (primeError) { if (error) *error = primeError; return nil; }
            NSTimeInterval seconds = -[t0 timeIntervalSinceNow];
            NSInteger primed = scratch.contextTokenCount;
            return apJSON(@{ @"prefill_tokens" : @(primed),
                             @"prefill_seconds" : @(seconds),
                             @"prefill_tps" : @(seconds > 0 ? primed / seconds : 0) });
        }];
}

@end
