//  APModel — facade over the es:: engine. Objective-C++: C++ ivars live directly in the
//  implementation (constructed/destructed with the object); no C++ crosses the header.
#import "APModel.h"
#import "APInternal.h"
#import "APError.h"

#include "ESWeightLoader.h"
#include "mlx/mlx.h"
#include "mlx/backend/metal/metal.h"

#include <memory>
#include <string>

static NSError * apError(APErrorCode code, NSString * message) {
    return [NSError errorWithDomain:APErrorDomain code:code
                           userInfo:@{ NSLocalizedDescriptionKey : message }];
}

//  APEngineRunner — one dedicated thread for ALL engine work of one model.
//
//  MLX streams are registered per thread (thread_local encoder registries), and GCD
//  queues do not pin threads — so engine calls must run on a stable thread that owns
//  its own default streams. One runner per model also provides the documented
//  serialization of generation across sessions sharing a model.
@interface APEngineRunner : NSObject
- (void)perform:(dispatch_block_t)block;
@end

@implementation APEngineRunner {
    NSThread * _thread;
    NSCondition * _cond;
    NSMutableArray<dispatch_block_t> * _blocks;
    BOOL _stop;
}
- (instancetype)init {
    if ((self = [super init])) {
        _cond = [[NSCondition alloc] init];
        _blocks = [NSMutableArray array];
        _stop = NO;
        _thread = [[NSThread alloc] initWithTarget:self selector:@selector(threadMain) object:nil];
        _thread.name = @"com.apertura.engine";
        _thread.qualityOfService = NSQualityOfServiceUserInitiated;
        [_thread start];
    }
    return self;
}
- (void)threadMain {
    // This thread owns its MLX streams (default_stream creates them lazily per thread —
    // everything engine-related, INCLUDING the load, must run here and only here).
    try {
        mlx::core::set_default_device(mlx::core::Device::gpu);
    } catch (const std::exception &) { /* engine calls will surface real errors */ }
    while (true) {
        dispatch_block_t block = nil;
        [_cond lock];
        while (_blocks.count == 0 && !_stop) [_cond wait];
        if (_stop && _blocks.count == 0) { [_cond unlock]; break; }
        block = _blocks.firstObject;
        [_blocks removeObjectAtIndex:0];
        [_cond unlock];
        @autoreleasepool { block(); }
    }
}
- (void)perform:(dispatch_block_t)block {
    [_cond lock];
    [_blocks addObject:[block copy]];
    [_cond signal];
    [_cond unlock];
}
- (void)dealloc {
    [_cond lock]; _stop = YES; [_cond broadcast]; [_cond unlock];
}
@end

/// Recursive size of the model's weight payload (safetensors files).
static unsigned long long apWeightBytesAtURL(NSURL * url) {
    NSFileManager * fm = NSFileManager.defaultManager;
    unsigned long long total = 0;
    NSDirectoryEnumerator * e = [fm enumeratorAtURL:url
                         includingPropertiesForKeys:@[ NSURLFileSizeKey ]
                                            options:0 errorHandler:nil];
    for (NSURL * f in e) {
        if (![f.pathExtension isEqualToString:@"safetensors"]) continue;
        NSNumber * size = nil;
        [f getResourceValue:&size forKey:NSURLFileSizeKey error:nil];
        total += size.unsignedLongLongValue;
    }
    return total;
}

@implementation APModel {
    std::unique_ptr<es::ESWeightLoader>          _weights;
    std::unique_ptr<es::ESGemma4TextForCausalLM> _lm;
    std::unique_ptr<es::ESTokenizer>             _tokenizer;
    std::unique_ptr<es::ESChatTemplate>          _template;
    es::ESModelConfig                            _config;
    APModelConfiguration *                       _apConfiguration;
    APEngineRunner *                             _runner;
    unsigned long long                           _weightBytes;
    BOOL                                         _prewarmed;
}

+ (APModelAvailability)availabilityOfModelAtURL:(NSURL *)url
                                  configuration:(APModelConfiguration *)configuration {
    NSFileManager * fm = NSFileManager.defaultManager;
    BOOL isDir = NO;
    if (![fm fileExistsAtPath:url.path isDirectory:&isDir] || !isDir) return APModelNotFound;
    NSString * cfg = [url.path stringByAppendingPathComponent:@"config.json"];
    if (![fm fileExistsAtPath:cfg]) return APModelIncompatible;
    // RAM pre-flight: weight bytes + working headroom vs physical memory. Conservative:
    // bf16 snapshots load at file size; bundles are already quantized. 2 GB fixed
    // overhead (KV cache, transients, JIT) and an 0.8 utilization ceiling.
    unsigned long long weightBytes = apWeightBytesAtURL(url);
    if (weightBytes == 0) return APModelIncompatible;
    unsigned long long need = weightBytes + (2ull << 30);
    unsigned long long have = (unsigned long long)(NSProcessInfo.processInfo.physicalMemory * 0.8);
    if (need > have) return APModelInsufficientMemory;
    return APModelAvailable;
}

- (nullable instancetype)initWithContentsOfURL:(NSURL *)url
                                 configuration:(APModelConfiguration *)configuration
                                         error:(NSError **)error {
    if (!(self = [super init])) return nil;
    APModelAvailability avail = [APModel availabilityOfModelAtURL:url configuration:configuration];
    if (avail != APModelAvailable) {
        if (error) {
            APErrorCode code = (avail == APModelNotFound) ? APErrorModelNotFound
                             : (avail == APModelInsufficientMemory) ? APErrorInsufficientMemory
                                                                    : APErrorIncompatibleModel;
            *error = apError(code, [NSString stringWithFormat:@"model unavailable at %@", url.path]);
        }
        return nil;
    }
    APModelConfiguration * conf = configuration ?: [APModelConfiguration defaultConfiguration];
    std::string dir([url.path UTF8String]);
    // The ENTIRE load runs on the model's dedicated engine thread: MLX streams are
    // per-thread, and the model builds lazy constant graphs at construction (embed
    // scales, re-quantized head, norm views) that must carry the engine thread's
    // streams — a load on the caller's thread would poison later evals with foreign
    // stream indices. init blocks until the load completes.
    _runner = [[APEngineRunner alloc] init];
    __block NSError * loadError = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [_runner perform:^{
        // When AperturaKit is a signed framework, the MLX metallib ships in its
        // Resources (codesign forbids it next to the binary); point MLX there before
        // the first Metal use. In non-bundle contexts (the research CLI) the resource
        // is absent and MLX's colocated-with-binary fallback applies unchanged.
        static dispatch_once_t once;
        dispatch_once(&once, ^{
            NSString * lib = [[NSBundle bundleForClass:APModel.class] pathForResource:@"mlx"
                                                                               ofType:@"metallib"];
            if (lib) mlx::core::metal::set_metallib_path(std::string(lib.UTF8String));
        });
        try {
            self->_config = es::ESModelConfig::fromConfigJSON(dir + "/config.json");
            self->_config.computeDtype = mlx::core::bfloat16;
            self->_config.fused = true;                        // the production path
            self->_config.prefillChunk = (int)conf.prefillChunkLength;
            if (conf.headBits > 0 && conf.headBits != 8)       // Q4-head opt-in (P4)
                self->_config.quantEmbedBits = (int)conf.headBits;
            self->_weights = std::make_unique<es::ESWeightLoader>(dir, self->_config);
            self->_lm = std::make_unique<es::ESGemma4TextForCausalLM>(self->_config, *self->_weights);
            self->_tokenizer = std::make_unique<es::ESTokenizer>(dir + "/tokenizer.json");
            self->_template = std::make_unique<es::ESChatTemplate>(*self->_tokenizer);
        } catch (const std::exception & e) {
            loadError = apError(APErrorEngineFailure, @(e.what()));
        }
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    if (loadError) {
        if (error) *error = loadError;
        return nil;
    }
    _apConfiguration = [conf copy];
    _modelURL = url;
    _modelIdentifier = url.lastPathComponent;
    _weightBytes = apWeightBytesAtURL(url);
    _prewarmed = NO;
    return self;
}

+ (nullable instancetype)modelWithContentsOfURL:(NSURL *)url
                                  configuration:(APModelConfiguration *)configuration
                                          error:(NSError **)error {
    return [[self alloc] initWithContentsOfURL:url configuration:configuration error:error];
}

+ (void)loadModelAtURL:(NSURL *)url
         configuration:(APModelConfiguration *)configuration
            completion:(void (^)(APModel *, NSError *))completion {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSError * err = nil;
        APModel * m = [self modelWithContentsOfURL:url configuration:configuration error:&err];
        dispatch_async(dispatch_get_main_queue(), ^{ completion(m, err); });
    });
}

- (void)prewarmWithCompletion:(void (^)(void))completion {
    [_runner perform:^{
        if (!self->_prewarmed) {
            try {   // tiny forward through the whole stack triggers the Metal JIT
                mlx::core::array ll = self->_lm->lastLogits({2}, nullptr, 0);
                mlx::core::eval(ll);
                self->_prewarmed = YES;
            } catch (const std::exception &) { /* prewarm is best-effort */ }
        }
        if (completion) dispatch_async(dispatch_get_main_queue(), ^{ completion(); });
    }];
}

- (void)reclaimMemory {
    [_runner perform:^{ mlx::core::clear_cache(); }];
}

- (void)performOnEngine:(dispatch_block_t)block {
    [_runner perform:block];
}

- (NSInteger)maximumContextLength { return _config.maxPositionEmbeddings; }
- (BOOL)supportsImageInput { return NO; }
- (BOOL)supportsAudioInput { return NO; }

#pragma mark - Internal (APInternal.h)

- (APModelConfiguration *)internalConfiguration { return _apConfiguration; }
- (unsigned long long)internalWeightBytes { return _weightBytes; }
- (es::ESGemma4TextForCausalLM *)internalLM { return _lm.get(); }
- (es::ESTokenizer *)internalTokenizer { return _tokenizer.get(); }
- (es::ESChatTemplate *)internalTemplate { return _template.get(); }
- (const es::ESModelConfig *)internalConfig { return &_config; }

@end
