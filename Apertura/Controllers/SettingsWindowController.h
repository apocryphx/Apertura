//
//  SettingsWindowController.h
//  Apertura
//
//  Settings (Cmd-,): two tabs. Models — the folder registry: import (copy or link in
//  place), select the active model, per-model load configuration (applies at next
//  load). Generation — the app-default sampling parameters copied into new sessions
//  (per-session values win) plus the reasoning defaults.

#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

/// Generation-default keys (registered in +initialize; the coordinator reads them).
FOUNDATION_EXPORT NSString * const AperturaGenTemperatureKey;   // double, 0.7
FOUNDATION_EXPORT NSString * const AperturaGenTopKKey;          // integer, 64
FOUNDATION_EXPORT NSString * const AperturaGenTopPKey;          // double, 0.95
FOUNDATION_EXPORT NSString * const AperturaGenSeedKey;          // integer, 0 = engine default
FOUNDATION_EXPORT NSString * const AperturaGenMaxTokensKey;     // integer, 0 = auto (2048/1024)
FOUNDATION_EXPORT NSString * const AperturaExcludesReasoningKey;// bool, YES

@interface SettingsWindowController : NSWindowController

+ (instancetype)sharedController;
- (void)showSettings;

@end

NS_ASSUME_NONNULL_END
