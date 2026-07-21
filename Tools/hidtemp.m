// hidtemp — read Apple Silicon die temperature sensors WITHOUT root.
//
// Build:  clang -fobjc-arc -O2 Tools/hidtemp.m -framework Foundation -framework IOKit -o hidtemp
// Usage:  ./hidtemp          -> "max die temp: 46.0 C  (45 sensors)"
//         ./hidtemp all      -> every sensor, sorted by temperature
//
// Why it exists (see PERFORMANCE_ROADMAP.md §6): thermals are a first-order benchmark
// confound on Apple Silicon — one AperturaResearch --bench run swings the M4 Max die
// 47->84 C and compresses decode throughput ~24% (21.1 cold vs 16.2 hot tok/s @512 ctx).
// Gate every benchmark arm on a cold start (max die <= ~48 C) and annotate before/after:
//
//   while [ "$(./hidtemp | sed 's/.*: \([0-9]*\).*/\1/')" -gt 48 ]; do sleep 20; done
//
// Mechanism: AppleSMC vends temperature sensors as HID services (usage page 0xff00,
// usage 5); each service's current reading is a HID temperature event. Same approach as
// the Stats/macmon apps. Uses private-but-stable IOKit HID symbols — no sudo needed,
// unlike `powermetrics` (which additionally reports GPU frequency + thermal pressure and
// is the better instrument when a NOPASSWD sudoers grant is acceptable).
//
// Sensor naming (M-series): "PMU tdie*" are the SoC die sensors (the throttle-relevant
// ones — CPU/GPU clusters), "PMU tdev*" package sensors, plus battery/NAND. "max die
// temp" aggregates tdie/SOC/GPU-named sensors only.
#import <Foundation/Foundation.h>

typedef struct __IOHIDEventSystemClient * IOHIDEventSystemClientRef;
typedef struct __IOHIDServiceClient * IOHIDServiceClientRef;
typedef struct __IOHIDEvent * IOHIDEventRef;

extern IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);
extern int IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef, CFDictionaryRef);
extern CFArrayRef IOHIDEventSystemClientCopyServices(IOHIDEventSystemClientRef);
extern CFTypeRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef, CFStringRef);
extern IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef, int64_t, int32_t, int64_t);
extern double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);

#define kIOHIDEventTypeTemperature 15
#define IOHIDEventFieldBase(type) ((type) << 16)

int main(int argc, char ** argv) {
    @autoreleasepool {
        bool all = (argc > 1 && strcmp(argv[1], "all") == 0);
        IOHIDEventSystemClientRef client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
        if (!client) { fprintf(stderr, "no HID event system client\n"); return 1; }
        NSDictionary * match = @{ @"PrimaryUsagePage": @(0xff00), @"PrimaryUsage": @(5) };
        IOHIDEventSystemClientSetMatching(client, (__bridge CFDictionaryRef) match);
        NSArray * services = CFBridgingRelease(IOHIDEventSystemClientCopyServices(client));

        double maxDie = 0;
        NSMutableArray * rows = [NSMutableArray array];
        for (id s in services) {
            IOHIDServiceClientRef svc = (__bridge IOHIDServiceClientRef) s;
            NSString * name = CFBridgingRelease(IOHIDServiceClientCopyProperty(svc, CFSTR("Product")));
            if (!name) continue;
            IOHIDEventRef ev = IOHIDServiceClientCopyEvent(svc, kIOHIDEventTypeTemperature, 0, 0);
            if (!ev) continue;
            double t = IOHIDEventGetFloatValue(ev, IOHIDEventFieldBase(kIOHIDEventTypeTemperature));
            CFRelease(ev);
            if (t <= 0 || t > 150) continue;  // absent/implausible readings
            [rows addObject:[NSString stringWithFormat:@"%7.2f C  %@", t, name]];
            if ([name containsString:@"tdie"] || [name hasPrefix:@"SOC"] || [name containsString:@"GPU"])
                if (t > maxDie) maxDie = t;
        }
        [rows sortUsingSelector:@selector(compare:)];
        if (all) for (NSString * r in rows) printf("%s\n", r.UTF8String);
        printf("max die temp: %.1f C  (%lu sensors)\n", maxDie, (unsigned long) rows.count);
    }
    return 0;
}
