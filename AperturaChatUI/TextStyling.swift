//  TextStyling — semantic text (APSemanticText.h) resolved into SwiftUI attributes.
//
//  The display model records intent only: "bold", "code", "a keyword". Fonts and colors
//  are chosen here, at display time, from system text styles and system colors, so Dynamic
//  Type and Dark Mode need nothing from the parser.

import SwiftUI
import AperturaChatModel

enum TextStyling {

    // MARK: Inline text

    /// Semantic inline text (bold, italic, code, strikethrough, links) in `font`.
    static func inlineText(_ semantic: NSAttributedString, font: Font) -> AttributedString {
        var result = AttributedString()
        let whole = NSRange(location: 0, length: semantic.length)
        semantic.enumerateAttributes(in: whole) { attributes, range, _ in
            var run = AttributedString(semantic.attributedSubstring(from: range).string)
            let weight = (attributes[.APSemanticFontWeightAttributeName] as? NSNumber)?.doubleValue
                ?? APFontWeightRegular
            let bold = weight >= APFontWeightSemibold
            let italic = (attributes[.APSemanticItalicAttributeName] as? NSNumber)?.boolValue ?? false
            let code = (attributes[.APSemanticCodeAttributeName] as? NSNumber)?.boolValue ?? false
            let strike = (attributes[.APSemanticStrikethroughAttributeName] as? NSNumber)?.boolValue ?? false

            var runFont = code ? font.monospaced() : font
            if bold { runFont = runFont.bold() }
            if italic { runFont = runFont.italic() }
            run.font = runFont
            if code { run.backgroundColor = Color(nsColor: .quaternarySystemFill) }
            if strike { run.strikethroughStyle = .single }
            if let link = attributes[.APSemanticLinkAttributeName] as? String, let url = URL(string: link) {
                run.link = url
            }
            result.append(run)
        }
        return result
    }

    // MARK: Code

    /// System colors for token kinds; they adapt to light and dark appearance.
    static func color(for kind: APCodeTokenKind) -> Color {
        switch kind {
        case .keyword:   return .pink
        case .type:      return .teal
        case .string:    return .red
        case .comment:   return .gray
        case .number:    return .blue
        case .attribute: return .orange
        @unknown default: return .primary
        }
    }

    /// Code tagged with APCodeTokenAttributeName, colored, in `font`.
    static func code(_ highlighted: NSAttributedString, font: Font) -> AttributedString {
        var result = AttributedString()
        let whole = NSRange(location: 0, length: highlighted.length)
        highlighted.enumerateAttribute(.APCodeTokenAttributeName, in: whole) { value, range, _ in
            var run = AttributedString(highlighted.attributedSubstring(from: range).string)
            run.font = font
            if let raw = (value as? NSNumber)?.intValue, let kind = APCodeTokenKind(rawValue: raw) {
                run.foregroundColor = color(for: kind)
            }
            result.append(run)
        }
        return result
    }

    // MARK: Headings

    static func headingFont(level: Int) -> Font {
        switch level {
        case 1:  return .title.bold()
        case 2:  return .title2.bold()
        case 3:  return .title3.bold()
        default: return .headline
        }
    }
}

enum Formatting {

    /// "0.02 s" under a tenth, "0.3 s" under ten seconds, "57 s" from there on.
    static func seconds(_ interval: TimeInterval) -> String {
        if interval < 0.095 { return String(format: "%.2f s", interval) }
        return interval < 10 ? String(format: "%.1f s", interval) : String(format: "%.0f s", interval)
    }

    /// The parts of a generation footer: "15.7 tok/s · 1104 tokens · 70.8 s · model".
    static func statsLine(_ stats: APResponseStats?, modelIdentifier: String?) -> String? {
        var parts: [String] = []
        if let stats {
            if stats.decodeTokensPerSecond > 0 {
                parts.append(String(format: "%.1f tok/s", stats.decodeTokensPerSecond))
            }
            if stats.responseTokenCount > 0 {
                parts.append("\(stats.responseTokenCount) tokens")
            }
            // The engine measures time to first token and the decode rate; the turn's
            // wall-clock is their sum.
            var duration = stats.timeToFirstToken
            if stats.decodeTokensPerSecond > 0 {
                duration += Double(stats.responseTokenCount) / stats.decodeTokensPerSecond
            }
            if duration > 0 { parts.append(String(format: "%.1f s", duration)) }
        }
        if let modelIdentifier, !modelIdentifier.isEmpty { parts.append(modelIdentifier) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    static func byteCount(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes), countStyle: .file)
    }
}
