import AppKit
import SwiftUI

/// Lightweight Python syntax highlighter using regex.
enum SyntaxHighlighter {
    static let backgroundColor = NSColor(red: 0.12, green: 0.12, blue: 0.14, alpha: 1.0)

    static let baseFont = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)

    private static let baseAttributes: [NSAttributedString.Key: Any] = [
        .font: baseFont,
        .foregroundColor: NSColor(red: 0.85, green: 0.85, blue: 0.85, alpha: 1.0),
    ]

    private static let rules: [(pattern: String, color: NSColor, options: NSRegularExpression.Options)] = [
        // Triple-quoted strings (must come first)
        (#"\"\"\"[\s\S]*?\"\"\""#, NSColor(red: 0.56, green: 0.74, blue: 0.56, alpha: 1.0), [.dotMatchesLineSeparators]),
        (#"'''[\s\S]*?'''"#, NSColor(red: 0.56, green: 0.74, blue: 0.56, alpha: 1.0), [.dotMatchesLineSeparators]),
        // Single-line strings
        (#""[^"\n]*""#, NSColor(red: 0.56, green: 0.74, blue: 0.56, alpha: 1.0), []),
        (#"'[^'\n]*'"#, NSColor(red: 0.56, green: 0.74, blue: 0.56, alpha: 1.0), []),
        // Comments
        (#"#.*$"#, NSColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0), [.anchorsMatchLines]),
        // Keywords
        (#"\b(def|class|import|from|return|if|elif|else|for|while|with|as|in|not|and|or|is|True|False|None|self|yield|lambda|try|except|raise|finally|pass|break|continue|assert|global|nonlocal|del|async|await)\b"#,
         NSColor(red: 0.78, green: 0.46, blue: 0.83, alpha: 1.0), []),
        // Decorators
        (#"@\w+"#, NSColor(red: 0.9, green: 0.72, blue: 0.42, alpha: 1.0), []),
        // Numbers
        (#"\b\d+\.?\d*([eE][+-]?\d+)?\b"#, NSColor(red: 0.73, green: 0.83, blue: 0.55, alpha: 1.0), []),
        // Built-in functions
        (#"\b(print|len|range|zip|enumerate|type|isinstance|super|int|float|str|list|dict|set|tuple|bool|min|max|sum|abs|sorted|reversed|map|filter|any|all|open|hasattr|getattr|setattr)\b"#,
         NSColor(red: 0.40, green: 0.73, blue: 0.87, alpha: 1.0), []),
    ]

    static func highlight(_ code: String) -> NSAttributedString {
        let result = NSMutableAttributedString(string: code, attributes: baseAttributes)
        guard !code.isEmpty else { return result }
        let fullRange = NSRange(code.startIndex..., in: code)

        var coloredRanges: [NSRange] = []

        for (pattern, color, options) in rules {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { continue }
            let matches = regex.matches(in: code, range: fullRange)

            for match in matches {
                let range = match.range
                let overlaps = coloredRanges.contains { existing in
                    NSIntersectionRange(existing, range).length > 0
                }
                if overlaps { continue }
                result.addAttribute(.foregroundColor, value: color, range: range)
                coloredRanges.append(range)
            }
        }

        return result
    }

    /// Apply highlighting to existing NSTextStorage (preserves cursor position).
    static func applyHighlighting(to textStorage: NSTextStorage) {
        let code = textStorage.string
        guard !code.isEmpty else { return }
        let fullRange = NSRange(location: 0, length: textStorage.length)

        textStorage.beginEditing()
        textStorage.setAttributes(baseAttributes, range: fullRange)

        var coloredRanges: [NSRange] = []
        for (pattern, color, options) in rules {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { continue }
            let matches = regex.matches(in: code, range: fullRange)
            for match in matches {
                let range = match.range
                let overlaps = coloredRanges.contains { NSIntersectionRange($0, range).length > 0 }
                if overlaps { continue }
                textStorage.addAttribute(.foregroundColor, value: color, range: range)
                coloredRanges.append(range)
            }
        }
        textStorage.endEditing()
    }
}


// MARK: - Editable Code Editor

/// Editable code editor with live syntax highlighting.
struct CodeEditorView: NSViewRepresentable {
    @Binding var code: String
    @Binding var selectedText: String
    var isEditable: Bool = true

    class Coordinator: NSObject, NSTextStorageDelegate, NSTextViewDelegate {
        var parent: CodeEditorView
        var isUpdating = false

        init(_ parent: CodeEditorView) {
            self.parent = parent
        }

        func textStorage(_ textStorage: NSTextStorage,
                         didProcessEditing editedMask: NSTextStorageEditActions,
                         range editedRange: NSRange,
                         changeInLength delta: Int) {
            guard editedMask.contains(.editedCharacters), !isUpdating else { return }

            // Update binding
            parent.code = textStorage.string

            // Re-highlight (defer to avoid re-entrant editing)
            DispatchQueue.main.async {
                self.isUpdating = true
                SyntaxHighlighter.applyHighlighting(to: textStorage)
                self.isUpdating = false
            }
        }

        // Track selection changes
        func textViewDidChangeSelection(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            let range = textView.selectedRange()
            if range.length > 0,
               let str = textView.string as NSString? {
                let selected = str.substring(with: range)
                DispatchQueue.main.async {
                    self.parent.selectedText = selected
                }
            } else {
                DispatchQueue.main.async {
                    self.parent.selectedText = ""
                }
            }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        let textView = scrollView.documentView as! NSTextView

        textView.isEditable = isEditable
        textView.isSelectable = true
        textView.allowsUndo = true
        textView.backgroundColor = SyntaxHighlighter.backgroundColor
        textView.insertionPointColor = .white
        textView.textContainerInset = NSSize(width: 12, height: 12)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.font = SyntaxHighlighter.baseFont
        textView.isRichText = false
        textView.usesFindBar = true

        textView.textStorage?.delegate = context.coordinator
        textView.delegate = context.coordinator

        let highlighted = SyntaxHighlighter.highlight(code)
        textView.textStorage?.setAttributedString(highlighted)

        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        let textView = nsView.documentView as! NSTextView
        // Only update if the source changed externally (e.g. file switch)
        if textView.string != code && !context.coordinator.isUpdating {
            context.coordinator.isUpdating = true
            let highlighted = SyntaxHighlighter.highlight(code)
            textView.textStorage?.setAttributedString(highlighted)
            context.coordinator.isUpdating = false
        }
    }
}


// MARK: - Read-only Code View (for output panels)

struct CodeView: NSViewRepresentable {
    let code: String

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        let textView = scrollView.documentView as! NSTextView

        textView.isEditable = false
        textView.isSelectable = true
        textView.backgroundColor = SyntaxHighlighter.backgroundColor
        textView.textContainerInset = NSSize(width: 12, height: 12)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false

        let highlighted = SyntaxHighlighter.highlight(code)
        textView.textStorage?.setAttributedString(highlighted)

        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        let textView = nsView.documentView as! NSTextView
        let highlighted = SyntaxHighlighter.highlight(code)
        textView.textStorage?.setAttributedString(highlighted)
    }
}
