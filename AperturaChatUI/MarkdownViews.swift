//  MarkdownViews — an APMarkdownDocument's blocks, stacked.
//
//  Headings, paragraphs and list items are selectable Text; code blocks scroll sideways
//  under a language caption; tables are a Grid with per-column widths; a rule is a
//  divider. Consecutive blocks at one quote level share a quote bar.

import SwiftUI
import AperturaChatModel

struct MarkdownDocumentView: View {
    let document: APMarkdownDocument

    /// Blocks grouped by quote level, in order.
    private var runs: [(level: Int, blocks: [APMarkdownBlock])] {
        var runs: [(level: Int, blocks: [APMarkdownBlock])] = []
        for block in document.blocks {
            if let last = runs.indices.last, runs[last].level == block.quoteLevel {
                runs[last].blocks.append(block)
            } else {
                runs.append((block.quoteLevel, [block]))
            }
        }
        return runs
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(runs.enumerated()), id: \.offset) { _, run in
                QuoteRun(level: run.level, blocks: run.blocks)
            }
        }
    }
}

/// Blocks at one quote level; quoted ones are indented behind a bar.
private struct QuoteRun: View {
    let level: Int
    let blocks: [APMarkdownBlock]

    var body: some View {
        if level == 0 {
            stack
        } else {
            HStack(alignment: .top, spacing: 0) {
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(.tertiary)
                    .frame(width: 3)
                stack
                    .foregroundStyle(.secondary)
                    .padding(.leading, CGFloat(level) * 14 - 3)
            }
        }
    }

    private var stack: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { index, block in
                MarkdownBlockView(block: block, first: index == 0)
            }
        }
    }
}

private struct MarkdownBlockView: View {
    let block: APMarkdownBlock
    let first: Bool
    @Environment(\.hugsContent) private var hugsContent

    private var fillWidth: CGFloat? { hugsContent ? nil : .infinity }

    var body: some View {
        if let heading = block as? APMarkdownHeadingBlock {
            Text(TextStyling.inlineText(heading.text, font: TextStyling.headingFont(level: heading.level)))
                .padding(.top, first ? 0 : (heading.level <= 2 ? 8 : 4))
                .frame(maxWidth: fillWidth, alignment: .leading)
        } else if let item = block as? APMarkdownListItemBlock {
            ListItemView(item: item)
        } else if let paragraph = block as? APMarkdownParagraphBlock {
            Text(TextStyling.inlineText(paragraph.text, font: .body))
                .frame(maxWidth: fillWidth, alignment: .leading)
        } else if let code = block as? APMarkdownCodeBlock {
            CodeBlockView(snippet: code.snippet, font: .callout.monospaced(), showsLanguage: true)
        } else if let table = block as? APMarkdownTableBlock {
            TableView(table: table.table)
        } else if block is APMarkdownRuleBlock {
            Divider()
                .padding(.vertical, 4)
        } else if let flow = block as? APMarkdownFlowBlock {
            Text(TextStyling.inlineText(flow.text, font: .body))
                .frame(maxWidth: fillWidth, alignment: .leading)
        }
    }
}

/// Marker right-aligned in an 18 pt column, text 6 pt after it; 20 pt per nesting level.
private struct ListItemView: View {
    let item: APMarkdownListItemBlock
    @Environment(\.hugsContent) private var hugsContent

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(item.marker)
                .font(.body.monospacedDigit())
                .frame(width: 18, alignment: .trailing)
            Text(TextStyling.inlineText(item.text, font: .body))
                .frame(maxWidth: hugsContent ? nil : .infinity, alignment: .leading)
        }
        .padding(.leading, CGFloat(max(item.depth - 1, 0)) * 20)
    }
}

// MARK: - Code

/// Optional language caption over highlighted code that scrolls sideways, in a filled box.
struct CodeBlockView: View {
    let snippet: APCodeSnippet
    let font: Font
    let showsLanguage: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if showsLanguage, let language = snippet.language, !language.isEmpty {
                Text(language)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 8)
                    .padding(.horizontal, 12)
            }
            ScrollView(.horizontal) {
                Text(TextStyling.code(snippet.highlighted, font: font))
                    .fixedSize(horizontal: true, vertical: true)
                    .padding(12)
            }
        }
        .background(Color(nsColor: .quaternarySystemFill), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

// MARK: - Table

/// Columns as wide as their widest cell (up to a cap, where text wraps); the whole grid
/// scrolls sideways when the row is wider than the column.
private struct TableView: View {
    let table: APMarkdownTable

    private var columns: Int { max(Int(table.columnCount), 1) }

    var body: some View {
        ScrollView(.horizontal) {
            Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
                if !table.header.isEmpty {
                    GridRow {
                        cells(table.header, font: .callout.weight(.semibold))
                    }
                    .background(Color(nsColor: .quaternarySystemFill))
                    Divider()
                }
                ForEach(Array(table.rows.enumerated()), id: \.offset) { index, row in
                    GridRow {
                        cells(row, font: .callout)
                    }
                    if index < table.rows.count - 1 {
                        Divider()
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.quaternary))
        }
    }

    @ViewBuilder
    private func cells(_ row: [NSAttributedString], font: Font) -> some View {
        ForEach(0..<columns, id: \.self) { column in
            let text = column < row.count ? row[column] : NSAttributedString()
            Text(TextStyling.inlineText(text, font: font))
                .frame(minWidth: 44, maxWidth: 320, alignment: .leading)
                .padding(.vertical, 6)
                .padding(.horizontal, 10)
        }
    }
}
