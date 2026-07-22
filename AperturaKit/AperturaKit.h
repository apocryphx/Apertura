//
//  AperturaKit.h
//  AperturaKit — on-device Gemma-4 inference for applications.
//
//  Created by Kolja Wawrowsky on 7/21/26.
//
//  Pure Objective-C surface over the conformance-gated Apertura engine (Objective-C++
//  and MLX underneath; no C++ types cross this boundary). Design + rationale:
//  aptransformer/API_PROPOSAL.md. Measured performance: PERFORMANCE_ROADMAP.md.

#import <Foundation/Foundation.h>

//! Project version number for AperturaKit.
FOUNDATION_EXPORT double AperturaKitVersionNumber;

//! Project version string for AperturaKit.
FOUNDATION_EXPORT const unsigned char AperturaKitVersionString[];

#import <AperturaKit/APError.h>
#import <AperturaKit/APModelConfiguration.h>
#import <AperturaKit/APModel.h>
#import <AperturaKit/APContent.h>
#import <AperturaKit/APMessage.h>
#import <AperturaKit/APGenerationOptions.h>
#import <AperturaKit/APResponse.h>
#import <AperturaKit/APTool.h>
#import <AperturaKit/APSession.h>
