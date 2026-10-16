//  BlockViews — one view per APChatBlock kind.
//
//  Markdown text flows through MarkdownDocumentView; reasoning and tool calls are
//  disclosures whose open/closed state lives in ExpansionState under the block's key;
//  an attachment is a chip; an image is a thumbnail (or a placeholder when its file is
//  gone) that opens in the default app on click.

import SwiftUI
import AperturaKit
import AperturaChatModel

struct BlockView: View {
    let block: APChatBlock
    let key: BlockKey
    let expansion: ExpansionState
    /// Seconds the model spent in this reasoning block, when measured.
    let reasoningDuration: TimeInterval?
    /// YES for the last block of a turn still streaming.
    let isStreaming: Bool
    let alignment: HorizontalAlignment

    var body: some View {
        if let text = block as? APTextBlock {
            MarkdownDocumentView(document: text.document)
        } else if let reasoning = block as? APReasoningBlock {
            ReasoningDisclosure(block: reasoning, key: key, expansion: expansion,
                                duration: reasoningDuration, isStreaming: isStreaming)
        } else if let call = block as? APToolCallBlock {
            ToolCallDisclosure(block: call, key: key, expansion: expansion)
        } else if let attachment = block as? APAttachmentBlock {
            AttachmentChipView(chip: AttachmentChip(name: attachment.attachment.name,
                                                    detail: Formatting.byteCount(attachment.attachment.sizeBytes)))
                .frame(maxWidth: .infinity, alignment: Alignment(horizontal: alignment, vertical: .center))
        } else if let image = block as? APImageBlock {
            ImageBlockView(block: image)
                .frame(maxWidth: .infinity, alignment: Alignment(horizontal: alignment, vertical: .center))
        }
    }
}

// MARK: - Disclosure

/// A clickable title row with a chevron; the content sits below it when open.
struct DisclosureRow<Label: View, Content: View>: View {
    @Binding var isExpanded: Bool
    @ViewBuilder let label: Label
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                isExpanded.toggle()
            } label: {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    label
                    Image(systemName: "chevron.right")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if isExpanded {
                content
                    .padding(.leading, 4)
            }
        }
    }
}

// MARK: - Reasoning

struct ReasoningDisclosure: View {
    let block: APReasoningBlock
    let key: BlockKey
    let expansion: ExpansionState
    let duration: TimeInterval?
    let isStreaming: Bool

    private var title: String {
        if let duration { return "Thought for \(Formatting.seconds(duration))" }
        return isStreaming ? "Thinking…" : "Thoughts"
    }

    var body: some View {
        DisclosureRow(isExpanded: expansion.expansionBinding(key)) {
            Label(title, systemImage: "brain")
                .font(.callout)
                .foregroundStyle(.secondary)
        } content: {
            Text(block.text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Tool call

struct ToolCallDisclosure: View {
    let block: APToolCallBlock
    let key: BlockKey
    let expansion: ExpansionState

    private var outcome: String {
        switch block.toolCall.status {
        case .succeeded: return "ok"
        case .failed:    return "failed"
        case .vetoed:    return "vetoed"
        @unknown default: return "?"
        }
    }

    private var statusColor: Color {
        block.toolCall.status == .succeeded ? .green : .red
    }

    var body: some View {
        DisclosureRow(isExpanded: expansion.expansionBinding(key)) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "wrench.and.screwdriver")
                Text(block.toolCall.name)
                    .fontWeight(.medium)
                Text("→ \(outcome)")
                    .foregroundStyle(statusColor)
                if block.toolCall.duration > 0 {
                    Text("· \(Formatting.seconds(block.toolCall.duration))")
                        .foregroundStyle(.tertiary)
                }
            }
            .font(.callout)
            .foregroundStyle(.secondary)
        } content: {
            VStack(alignment: .leading, spacing: 8) {
                Caption("Arguments")
                CodeBlockView(snippet: block.argumentsSnippet, font: .caption.monospaced(), showsLanguage: false)
                if let preview = block.resultPreview, let result = block.toolCall.result {
                    Caption("Result")
                    if expansion.isRevealed(key) {
                        Text(result)
                            .font(.caption.monospaced())
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .background(Color(nsColor: .quaternarySystemFill),
                                        in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    } else {
                        CodeBlockView(snippet: preview, font: .caption.monospaced(), showsLanguage: false)
                    }
                    if block.isResultTruncated {
                        Button(expansion.isRevealed(key) ? "Show less" : "Show more") {
                            expansion.setRevealed(key, !expansion.isRevealed(key))
                        }
                        .buttonStyle(.link)
                        .font(.caption)
                    }
                }
            }
        }
    }

    private struct Caption: View {
        let text: String
        init(_ text: String) { self.text = text }
        var body: some View {
            Text(text)
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }
}

// MARK: - Image

/// A thumbnail of an attached image, aspect-fit within `maxEdge` points, decoded off the
/// main thread from the block's file; a placeholder chip when the file is not there.
/// Click opens the file in its default app.
struct ImageBlockView: View {
    let block: APImageBlock

    /// Longest edge of the thumbnail, in points.
    static let maxEdge: CGFloat = 320

    @State private var loaded: NSImage?
    @State private var failed = false

    private var fitSize: CGSize {
        let px = block.image.pixelSize
        guard px.width > 0, px.height > 0 else { return CGSize(width: Self.maxEdge, height: Self.maxEdge * 0.75) }
        let scale = min(1, Self.maxEdge / max(px.width, px.height))
        return CGSize(width: px.width * scale, height: px.height * scale)
    }

    var body: some View {
        if let url = block.image.fileURL, !failed {
            Button {
                NSWorkspace.shared.open(url)
            } label: {
                ZStack {
                    if let loaded {
                        Image(nsImage: loaded)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                    } else {
                        Color(nsColor: .quaternarySystemFill)
                        ProgressView().controlSize(.small)
                    }
                }
                .frame(width: fitSize.width, height: fitSize.height)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Color(nsColor: .separatorColor), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .help("\(block.displayName) — \(block.detail)")
            .accessibilityLabel(block.displayName)
            .task(id: url) { await load(url) }
        } else {
            AttachmentChipView(chip: AttachmentChip(name: block.displayName,
                                                    detail: failed ? "\(block.detail) · unreadable" : "\(block.detail) · file missing"),
                               systemImage: "photo")
        }
    }

    private func load(_ url: URL) async {
        let edge = Self.maxEdge * 2   // enough pixels for a Retina thumbnail
        let image = await Task.detached(priority: .utility) { () -> NSImage? in
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: edge,
                kCGImageSourceCreateThumbnailWithTransform: true,
            ]
            guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
            return NSImage(cgImage: cg, size: .zero)
        }.value
        if let image { loaded = image } else { failed = true }
    }
}

// MARK: - Attachment

struct AttachmentChipView: View {
    let chip: AttachmentChip
    var systemImage = "doc.text"

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
            Text(chip.name)
            if let detail = chip.detail {
                Text(detail)
                    .foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.6), in: Capsule())
        .fixedSize()
    }
}
