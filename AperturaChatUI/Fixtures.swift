//  Fixtures — a rich conversation for previews, decoded through the kit's own v2
//  transcript codec (the only public way to a message with a generation record).

import AppKit
import AperturaKit

enum ChatFixture {

    static let assistantName = "Isolde"

    /// A user turn with an attachment and an image, then an assistant turn with reasoning,
    /// two tool calls and Markdown covering headings, lists, quotes, code, a table and a
    /// rule. Image references resolve against `fileStore`, so the thumbnail renders
    /// without any asset.
    static var conversation: [APMessage] {
        var version: Int16 = 0
        return APTranscript.messages(from: transcript, version: &version, fileStore: fileStore) ?? []
    }

    /// A store in a temporary directory holding one generated picture.
    static let fileStore: APFileStore = {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AperturaChatUI-fixture-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
        return APFileStore(directoryURL: dir)
    }()

    /// The generated picture, stored: a 640×400 gradient with a circle, as PNG.
    static let picture: APImage = {
        let width = 640, height = 400
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let colors = [CGColor(srgbRed: 0.18, green: 0.32, blue: 0.62, alpha: 1),
                      CGColor(srgbRed: 0.90, green: 0.58, blue: 0.25, alpha: 1)] as CFArray
        let gradient = CGGradient(colorsSpace: space, colors: colors, locations: [0, 1])!
        context.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: width, y: height), options: [])
        context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.85))
        context.fillEllipse(in: CGRect(x: 220, y: 100, width: 200, height: 200))
        let cg = context.makeImage()!
        let rep = NSBitmapImageRep(cgImage: cg)
        let png = rep.representation(using: .png, properties: [:])!
        return try! fileStore.storeImageData(png, originalName: "sketch.png")
    }()

    private static let transcript: Data = {
        let markdown = """
        ## Roadmap summary

        The notes describe **three milestones** for the *display model*: parse once, \
        re-parse only the tail, and keep `APChatBlock` free of AppKit. See the \
        [design note](https://example.com/design) for the full argument.

        1. Settle the source at block boundaries
        2. Re-parse the open tail per delta
           - a paragraph, list or table still being written
           - a fenced code block until its closing fence
        3. Hand immutable blocks to the view

        > Rendering resolves appearance at display time, so a theme change needs no re-parse.

        ```swift
        let stream = APMarkdownStream()
        for delta in deltas where delta.kind == .text {
            stream.appendString(delta.text)   // O(tail), not O(document)
        }
        ```

        | Phase | Cost per delta | Notes |
        | --- | --- | --- |
        | Full re-parse | ~1.6 ms | grows with the document |
        | Settled prefix | 0.02 ms | p50 on M4 Max |
        | Tail only | 0.03 ms | p95 |

        ---

        That is ~~ten~~ eighty times cheaper at the end of a long answer.
        """
        let reasoning = """
        The user attached meeting notes and wants the roadmap compared with the design \
        note. I should search the notes first, then read the roadmap file, and present \
        the milestones as a numbered list with the measured costs in a table.
        """
        let searchResult = """
        {"hits": [{"path": "notes/roadmap.md", "score": 0.91}, {"path": "notes/2026-10-13.md", "score": 0.44}], \
        "summary": "Two notes mention the roadmap:\\n  - roadmap.md lists the milestones\\n  - 2026-10-13.md records the benchmark numbers"}
        """
        let document: [String: Any] = [
            "version": 2,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["kind": "image", "id": picture.identifier, "type": picture.mediaType,
                         "size": picture.byteCount, "width": Int(picture.pixelSize.width),
                         "height": Int(picture.pixelSize.height), "name": "sketch.png"],
                        ["kind": "attachment", "name": "meeting-notes.md",
                         "text": "# Meeting\n\nRoadmap discussed.", "size": 4821,
                         "type": "net.daringfireball.markdown"],
                        ["kind": "text", "text": "Compare the roadmap in these notes (and the sketch) with the design note and summarise the milestones."],
                    ],
                ],
                [
                    "role": "assistant",
                    "model": "google/gemma-4-12b-it-qat",
                    "stats": ["prompt_tokens": 2310, "response_tokens": 1104, "ttft": 1.6,
                              "prefill_tps": 1443.0, "decode_tps": 15.7],
                    "content": [
                        ["kind": "reasoning", "text": reasoning],
                        ["kind": "tool_call", "name": "search_notes",
                         "arguments": ["query": "roadmap", "limit": 5],
                         "status": "ok", "result": searchResult, "duration": 0.31],
                        ["kind": "tool_call", "name": "read_note",
                         "arguments": ["path": "notes/design.md"],
                         "status": "error", "result": "ENOENT: notes/design.md does not exist", "duration": 0.02],
                        ["kind": "text", "text": markdown],
                    ],
                ],
                [
                    "role": "user",
                    "content": [
                        ["kind": "image", "id": String(repeating: "0", count: 64), "type": "image/jpeg",
                         "size": 2_480_113, "width": 4032, "height": 3024, "name": "IMG_0421.jpeg"],
                        ["kind": "text", "text": "Thanks — and the *tail* re-parse is bounded how?"],
                    ],
                ],
            ],
        ]
        return try! JSONSerialization.data(withJSONObject: document)
    }()

    /// A model mid-stream: the fixture conversation, then an assistant turn that has
    /// thought, called a tool and begun its answer.
    static func streamingModel() -> ChatTranscriptModel {
        let model = ChatTranscriptModel()
        model.appendNote("— resumed \"Roadmap\" —")
        for message in conversation {
            model.append(message: message, sender: assistantName)
        }
        model.appendSpeakerHeader(assistantName)
        model.appendDelta(.reasoningDelta("By the settled prefix: only the text after the last trusted block boundary is parsed again, "))
        model.appendDelta(.reasoningDelta("so the cost is proportional to the open paragraph, list, table or code block."))
        model.appendDelta(.toolCall(APToolCall(name: "read_note",
                                                     arguments: ["path": "notes/2026-10-13.md"],
                                                     status: .succeeded,
                                                     result: "p50 0.02 ms, p95 0.03 ms, max 0.17 ms per delta",
                                                     duration: 0.05)))
        model.appendDelta(.textDelta("It is bounded by the **open tail**: the stream settles everything up to the last blank line followed by an unindented, non-list line outside a fence, and only what follows is re-parsed.\n\n- Settled blocks never change\n- The tail is at most one paragraph, list, table or code bl"))
        return model
    }
}
