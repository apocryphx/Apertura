//  BlockViews — one view per APChatBlock kind.
//
//  Markdown text flows through MarkdownDocumentView; reasoning and tool calls are
//  disclosures whose open/closed state lives in ExpansionState under the block's key;
//  an attachment is a chip.

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

// MARK: - Attachment

struct AttachmentChipView: View {
    let chip: AttachmentChip

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "doc.text")
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
