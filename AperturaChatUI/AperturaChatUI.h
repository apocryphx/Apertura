//  AperturaChatUI — the SwiftUI chat transcript for Apertura.
//
//  Renders AperturaChatModel's blocks (Markdown, code, tables, reasoning, tool calls,
//  attachments) as one scrolling column of turns. Swift only; this header exists so the
//  framework has an umbrella for Objective-C clients, which reach the Swift types through
//  the generated AperturaChatUI-Swift.h via @import AperturaChatUI.
#import <Foundation/Foundation.h>

//! Project version number for AperturaChatUI.
FOUNDATION_EXPORT double AperturaChatUIVersionNumber;

//! Project version string for AperturaChatUI.
FOUNDATION_EXPORT const unsigned char AperturaChatUIVersionString[];
