//  AperturaChatModel — the display model for Apertura chats.
//
//  Foundation only: turns AperturaKit messages and streamed deltas into renderable blocks
//  (Markdown structure, highlighted code, readable tool calls, attachments) with semantic
//  attributes a SwiftUI or AppKit view resolves into fonts and colors. Nothing here draws.
#import <Foundation/Foundation.h>

//! Project version number for AperturaChatModel.
FOUNDATION_EXPORT double AperturaChatModelVersionNumber;

//! Project version string for AperturaChatModel.
FOUNDATION_EXPORT const unsigned char AperturaChatModelVersionString[];

#import <AperturaChatModel/APSemanticText.h>
#import <AperturaChatModel/APCodeHighlighter.h>
#import <AperturaChatModel/APReadableJSON.h>
#import <AperturaChatModel/APMarkdownDocument.h>
#import <AperturaChatModel/APMarkdownStream.h>
#import <AperturaChatModel/APChatBlock.h>
#import <AperturaChatModel/APDisplayMessage.h>
#import <AperturaChatModel/APLiveMessage.h>
