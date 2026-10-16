//  ChatTranscriptModel — the transcript as the chat coordinator drives it.
//
//  The coordinator narrates a conversation through APChatCoordinatorDelegate: a speaker
//  header opens a turn, streamed text fills a user turn, deltas fill an assistant turn
//  (live or replayed from the archive, the same way), a note is machinery between turns.
//  This model folds that narration into turns a SwiftUI view renders: finished turns are
//  immutable values, the open assistant turn is an observable LiveTurn that updates per
//  delta, so a delta never touches a finished turn's view.

import AppKit
import Observation
import SwiftUI
import AperturaKit
import AperturaChatModel

// MARK: - Keys and values

/// Identifies one block of one turn across re-parses. The live message re-wraps its open
/// part on every delta and a finished turn re-derives its blocks from the recorded
/// message, so object identity does not survive streaming — but a turn's blocks only
/// ever append, so (turn, index) does.
public struct BlockKey: Hashable, Sendable {
    public let turn: UUID
    public let index: Int
    public init(turn: UUID, index: Int) {
        self.turn = turn
        self.index = index
    }
}

public enum TurnRole: Sendable {
    case user
    case assistant
}

/// An attachment shown as a chip: the file's name and a detail ("4 KB").
public struct AttachmentChip: Hashable, Sendable {
    public let name: String
    public let detail: String?
    public init(name: String, detail: String?) {
        self.name = name
        self.detail = detail
    }
}

/// One rendered turn. A value: the blocks it holds are immutable, and a finished turn
/// never changes again.
public struct TranscriptTurn: Identifiable {
    public let id: UUID
    public let role: TurnRole
    /// The speaker name the assistant turn shows ("Isolde").
    public let sender: String
    public let blocks: [APChatBlock]
    public let stats: APResponseStats?
    public let modelIdentifier: String?
    /// Chips the coordinator described in notes rather than as attachment parts (a user
    /// turn's files); attachment parts render from their own APAttachmentBlock.
    public let chips: [AttachmentChip]
    /// Seconds the model spent in each reasoning block, by block index, where measured.
    public let reasoningDurations: [Int: TimeInterval]
    /// YES while deltas are still arriving.
    public let isStreaming: Bool

    public init(id: UUID = UUID(), role: TurnRole, sender: String, blocks: [APChatBlock],
                stats: APResponseStats? = nil, modelIdentifier: String? = nil,
                chips: [AttachmentChip] = [], reasoningDurations: [Int: TimeInterval] = [:],
                isStreaming: Bool = false) {
        self.id = id
        self.role = role
        self.sender = sender
        self.blocks = blocks
        self.stats = stats
        self.modelIdentifier = modelIdentifier
        self.chips = chips
        self.reasoningDurations = reasoningDurations
        self.isStreaming = isStreaming
    }

    /// A recorded message as a turn.
    public init(message: APMessage, sender: String) {
        self.init(role: message.role == .user ? .user : .assistant,
                  sender: sender,
                  blocks: APDisplayMessage(message: message).blocks,
                  stats: message.stats,
                  modelIdentifier: message.modelIdentifier)
    }
}

/// What the transcript lists: turns, and the small notes between them.
public enum TranscriptEntry: Identifiable {
    case turn(TranscriptTurn)
    case note(id: UUID, text: String)

    public var id: UUID {
        switch self {
        case .turn(let turn): return turn.id
        case .note(let id, _): return id
        }
    }
}

// MARK: - Expansion

/// Which disclosures are open, keyed by block so the state survives scrolling (lazy
/// views come and go) and streaming (blocks are re-created).
@Observable
public final class ExpansionState {
    public private(set) var expanded: Set<BlockKey> = []
    /// Tool results shown in full rather than as their preview.
    public private(set) var revealed: Set<BlockKey> = []

    public init() {}

    public func isExpanded(_ key: BlockKey) -> Bool { expanded.contains(key) }
    public func isRevealed(_ key: BlockKey) -> Bool { revealed.contains(key) }

    public func setExpanded(_ key: BlockKey, _ value: Bool) {
        if value { expanded.insert(key) } else { expanded.remove(key) }
    }

    public func setRevealed(_ key: BlockKey, _ value: Bool) {
        if value { revealed.insert(key) } else { revealed.remove(key) }
    }

    func expansionBinding(_ key: BlockKey) -> Binding<Bool> {
        Binding { self.isExpanded(key) } set: { self.setExpanded(key, $0) }
    }
}

// MARK: - Live turn

/// The assistant turn in flight: an APLiveMessage fed delta by delta, re-published as a
/// TranscriptTurn after each one. Observed only by the view of this turn.
@Observable
public final class LiveTurn {
    public let id: UUID
    public let sender: String
    public private(set) var turn: TranscriptTurn

    @ObservationIgnored private let message = APLiveMessage()
    @ObservationIgnored private var reasoningStart: Date?
    @ObservationIgnored private var reasoningIndex: Int?
    @ObservationIgnored private var reasoningDurations: [Int: TimeInterval] = [:]

    /// A replayed turn arrives in one run-loop pass; shorter "thoughts" than this are
    /// not measurements and get no duration.
    private static let measurableReasoning: TimeInterval = 0.1

    public init(sender: String) {
        id = UUID()
        self.sender = sender
        turn = TranscriptTurn(id: id, role: .assistant, sender: sender, blocks: [], isStreaming: true)
    }

    public func append(_ delta: APResponseDelta) {
        message.append(delta)
        if delta.kind == .reasoning {
            if reasoningStart == nil {
                reasoningStart = Date()
                reasoningIndex = message.blocks.count - 1
            }
        } else {
            closeReasoning()
        }
        publish(streaming: true)
    }

    /// The finished turn, with the message the session records for it.
    public func finish(stats: APResponseStats?, modelIdentifier: String?) -> TranscriptTurn {
        closeReasoning()
        let recorded = message.message(with: stats, modelIdentifier: modelIdentifier)
        return TranscriptTurn(id: id, role: .assistant, sender: sender,
                              blocks: APDisplayMessage(message: recorded).blocks,
                              stats: stats, modelIdentifier: modelIdentifier,
                              reasoningDurations: reasoningDurations, isStreaming: false)
    }

    private func closeReasoning() {
        guard let start = reasoningStart, let index = reasoningIndex else { return }
        let elapsed = Date().timeIntervalSince(start)
        if elapsed >= Self.measurableReasoning { reasoningDurations[index] = elapsed }
        reasoningStart = nil
        reasoningIndex = nil
    }

    private func publish(streaming: Bool) {
        turn = TranscriptTurn(id: id, role: .assistant, sender: sender, blocks: message.blocks,
                              reasoningDurations: reasoningDurations, isStreaming: streaming)
    }
}

// MARK: - Model

/// The transcript, driven by APChatCoordinator through its delegate's rendering calls
/// (forwarded one to one by the chat view controller) or filled directly from messages.
@objc(APChatTranscriptModel)
@Observable
public final class ChatTranscriptModel: NSObject {

    public private(set) var entries: [TranscriptEntry] = []
    public private(set) var live: LiveTurn?
    public let expansion = ExpansionState()

    /// The speaker header that opens a user turn; every other name opens an assistant turn.
    @objc public var userName: String = "You"

    @ObservationIgnored private var pendingUser: PendingUserTurn?

    /// A user turn being narrated: its text arrives after its attachment notes — or, from
    /// a coordinator that hands the message over, the message itself.
    private struct PendingUserTurn {
        var text = ""
        var chips: [AttachmentChip] = []
        var message: APMessage?
    }

    public override init() {
        super.init()
    }

    // MARK: Messages

    /// Appends a recorded message as a finished turn.
    public func append(message: APMessage, sender: String) {
        closeOpenTurn()
        entries.append(.turn(TranscriptTurn(message: message, sender: sender)))
    }

    // MARK: Coordinator narration

    @objc public func clearTranscript() {
        pendingUser = nil
        live = nil
        entries.removeAll()
    }

    /// The user turn as the session receives it: images, attachments and text become the
    /// turn's blocks (APDisplayMessage). Inside an open user turn it is that turn's body;
    /// outside one it opens and closes a user turn of its own.
    @objc public func appendUserMessage(_ message: APMessage) {
        if pendingUser == nil {
            closeOpenTurn()
            pendingUser = PendingUserTurn()
        }
        pendingUser?.message = message
    }

    @objc public func appendSpeakerHeader(_ name: String) {
        closeOpenTurn()
        if name == userName {
            pendingUser = PendingUserTurn()
        } else {
            live = LiveTurn(sender: name)
        }
    }

    /// A user turn's body; on an assistant turn the coordinator's trailing newline, which
    /// closes the turn when no generation record came first.
    @objc public func appendStreamedText(_ text: String) {
        if pendingUser != nil {
            pendingUser?.text += text
        } else if let live {
            if text.allSatisfy(\.isWhitespace) {
                finishLive(live, stats: nil, modelIdentifier: nil)
            } else {
                live.append(APResponseDelta.textDelta(text))
            }
        } else if !text.allSatisfy(\.isWhitespace) {
            appendNote(text)
        }
    }

    @objc public func appendDelta(_ delta: APResponseDelta) {
        if live == nil {
            // Deltas without a header: open an anonymous assistant turn rather than lose them.
            closeOpenTurn()
            live = LiveTurn(sender: "")
        }
        live?.append(delta)
    }

    /// Machinery between turns ("— resumed … —", staging notices). Inside an open user
    /// turn, a "📎 name (size)" note is that turn's attachment chip.
    @objc public func appendNote(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if pendingUser != nil, let chip = Self.attachmentChip(fromNote: trimmed) {
            pendingUser?.chips.append(chip)
            return
        }
        entries.append(.note(id: UUID(), text: trimmed))
    }

    /// The generation record for the turn in flight; closes it.
    @objc public func finishTurn(stats: APResponseStats?, modelIdentifier: String?) {
        if let live {
            finishLive(live, stats: stats, modelIdentifier: modelIdentifier)
        } else {
            closeOpenTurn()
        }
    }

    // MARK: Hosting

    /// The transcript view, for an AppKit hierarchy.
    @objc public func makeTranscriptView() -> NSView {
        let hosting = NSHostingView(rootView: ChatTranscriptView(model: self))
        hosting.sizingOptions = []
        return hosting
    }

    // MARK: Internals

    private func closeOpenTurn() {
        if let pending = pendingUser {
            pendingUser = nil
            let message: APMessage
            if let handed = pending.message {
                message = handed
            } else {
                let text = pending.text.trimmingCharacters(in: .newlines)
                let content: [APContent] = text.isEmpty ? [] : [APContent.textContent(text)]
                message = APMessage(role: .user, content: content)
            }
            entries.append(.turn(TranscriptTurn(role: .user, sender: userName,
                                                blocks: APDisplayMessage(message: message).blocks,
                                                chips: pending.chips)))
        }
        if let live {
            finishLive(live, stats: nil, modelIdentifier: nil)
        }
    }

    private func finishLive(_ live: LiveTurn, stats: APResponseStats?, modelIdentifier: String?) {
        entries.append(.turn(live.finish(stats: stats, modelIdentifier: modelIdentifier)))
        self.live = nil
    }

    /// "📎 report.md (4 KB)" → name "report.md", detail "4 KB".
    static func attachmentChip(fromNote note: String) -> AttachmentChip? {
        guard note.hasPrefix("📎") else { return nil }
        let body = note.dropFirst().trimmingCharacters(in: .whitespaces)
        guard !body.isEmpty else { return nil }
        if body.hasSuffix(")"), let open = body.range(of: " (", options: .backwards) {
            let name = String(body[..<open.lowerBound])
            let detail = String(body[open.upperBound...].dropLast())
            return AttachmentChip(name: name, detail: detail.isEmpty ? nil : detail)
        }
        return AttachmentChip(name: body, detail: nil)
    }
}
