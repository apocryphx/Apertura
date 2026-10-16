//  Previews — the transcript on the fixture, so the look can be iterated without a model.

import SwiftUI
import AperturaKit
import AperturaChatModel

#Preview("Conversation") {
    ChatTranscriptView(messages: ChatFixture.conversation, live: nil, assistantName: ChatFixture.assistantName)
        .frame(width: 760, height: 900)
}

#Preview("Streaming") {
    ChatTranscriptView(model: ChatFixture.streamingModel())
        .frame(width: 760, height: 900)
}

#Preview("Narrow") {
    ChatTranscriptView(messages: ChatFixture.conversation, live: nil, assistantName: ChatFixture.assistantName)
        .frame(width: 520, height: 900)
}

#Preview("Disclosures open") {
    let model = ChatFixture.streamingModel()
    if case .turn(let turn) = model.entries.first(where: { if case .turn(let t) = $0 { return t.role == .assistant } else { return false } })! {
        for index in 0..<3 { model.expansion.setExpanded(BlockKey(turn: turn.id, index: index), true) }
    }
    return ChatTranscriptView(model: model)
        .frame(width: 760, height: 900)
}
