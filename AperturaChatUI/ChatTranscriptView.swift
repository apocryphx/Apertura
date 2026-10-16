//  ChatTranscriptView — the conversation as one scrolling column of turns.
//
//  Finished turns are laid out lazily and never re-rendered by streaming: the live turn
//  is the only view that observes the LiveTurn, so a delta re-evaluates that turn alone.
//  User turns sit in a trailing bubble sized to their content; assistant turns run full
//  width with the sender above and the generation record below.

import SwiftUI
import AperturaKit
import AperturaChatModel

public struct ChatTranscriptView: View {

    private enum Source {
        case model(ChatTranscriptModel)
        case fixed(entries: [TranscriptEntry], live: TranscriptTurn?)
    }

    private let source: Source
    @State private var localExpansion = ExpansionState()

    /// The transcript the coordinator drives (ChatTranscriptModel owns the expansion state).
    public init(model: ChatTranscriptModel) {
        source = .model(model)
    }

    /// Recorded messages plus, optionally, the turn in flight. `live` is read when the view
    /// is evaluated; re-create the view after each delta.
    public init(messages: [APMessage], live: APLiveMessage?, assistantName: String = "Assistant") {
        let entries = messages.map { TranscriptEntry.turn(TranscriptTurn(message: $0, sender: assistantName)) }
        let liveTurn = live.map {
            TranscriptTurn(role: .assistant, sender: assistantName, blocks: $0.blocks, isStreaming: true)
        }
        source = .fixed(entries: entries, live: liveTurn)
    }

    public var body: some View {
        switch source {
        case .model(let model):
            TranscriptList(entries: model.entries, live: model.live, fixedLive: nil, expansion: model.expansion)
        case .fixed(let entries, let live):
            TranscriptList(entries: entries, live: nil, fixedLive: live, expansion: localExpansion)
        }
    }
}

/// Maximum width of the reading column.
let columnWidth: CGFloat = 820

/// YES inside a user bubble: flow text sizes to its content instead of filling the column,
/// so the bubble hugs what it holds.
extension EnvironmentValues {
    @Entry var hugsContent: Bool = false
}

private struct TranscriptList: View {
    let entries: [TranscriptEntry]
    let live: LiveTurn?
    let fixedLive: TranscriptTurn?
    let expansion: ExpansionState

    private static let bottomID = "transcript.bottom"

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                LazyVStack(alignment: .leading, spacing: 20) {
                    ForEach(entries) { entry in
                        EntryView(entry: entry, expansion: expansion)
                    }
                    if let live {
                        LiveTurnView(live: live, expansion: expansion)
                    } else if let fixedLive {
                        TurnView(turn: fixedLive, expansion: expansion)
                    }
                    Color.clear
                        .frame(height: 1)
                        .id(Self.bottomID)
                }
                .frame(maxWidth: columnWidth)
                .frame(maxWidth: .infinity)
                .padding(16)
            }
            // Growth while streaming keeps the end in view; a new turn jumps to it.
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .defaultScrollAnchor(.bottom, for: .sizeChanges)
            .onAppear { proxy.scrollTo(Self.bottomID, anchor: .bottom) }
            .onChange(of: entries.count) { proxy.scrollTo(Self.bottomID, anchor: .bottom) }
            .onChange(of: live?.id) { proxy.scrollTo(Self.bottomID, anchor: .bottom) }
        }
        .textSelection(.enabled)
    }
}

private struct EntryView: View {
    let entry: TranscriptEntry
    let expansion: ExpansionState

    var body: some View {
        switch entry {
        case .turn(let turn):
            TurnView(turn: turn, expansion: expansion)
        case .note(_, let text):
            NoteView(text: text)
        }
    }
}

/// Reads the live turn, so only this view re-evaluates per delta.
private struct LiveTurnView: View {
    let live: LiveTurn
    let expansion: ExpansionState

    var body: some View {
        TurnView(turn: live.turn, expansion: expansion)
    }
}

/// Small tertiary machinery between turns.
struct NoteView: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Turns

struct TurnView: View {
    let turn: TranscriptTurn
    let expansion: ExpansionState

    var body: some View {
        switch turn.role {
        case .user:
            HStack(spacing: 0) {
                Spacer(minLength: 80)
                UserBubble(turn: turn, expansion: expansion)
            }
        case .assistant:
            AssistantTurnView(turn: turn, expansion: expansion)
        }
    }
}

private struct UserBubble: View {
    let turn: TranscriptTurn
    let expansion: ExpansionState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(turn.chips, id: \.self) { chip in
                AttachmentChipView(chip: chip)
            }
            TurnBlocks(turn: turn, expansion: expansion)
        }
        .padding(12)
        .background(Color.accentColor.opacity(0.12), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .environment(\.hugsContent, true)
    }
}

private struct AssistantTurnView: View {
    let turn: TranscriptTurn
    let expansion: ExpansionState

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !turn.sender.isEmpty {
                Label(turn.sender, systemImage: "sparkles")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if turn.isStreaming && turn.blocks.isEmpty {
                ProgressView()
                    .controlSize(.small)
            }
            TurnBlocks(turn: turn, expansion: expansion)
            if let line = Formatting.statsLine(turn.stats, modelIdentifier: turn.modelIdentifier) {
                Text(line)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A turn's blocks in order, each keyed by its index within the turn.
private struct TurnBlocks: View {
    let turn: TranscriptTurn
    let expansion: ExpansionState

    var body: some View {
        ForEach(Array(turn.blocks.enumerated()), id: \.offset) { index, block in
            BlockView(block: block,
                      key: BlockKey(turn: turn.id, index: index),
                      expansion: expansion,
                      reasoningDuration: turn.reasoningDurations[index],
                      isStreaming: turn.isStreaming && index == turn.blocks.count - 1,
                      alignment: turn.role == .user ? .trailing : .leading)
        }
    }
}
