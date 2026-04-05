import AppKit
import SwiftUI

// MARK: - Code Theme

enum CodeTheme: String, CaseIterable, Identifiable {
    case dark = "Dark"
    case monokai = "Monokai"
    case solarized = "Solarized Dark"
    case github = "GitHub Dark"
    case light = "Light"

    var id: String { rawValue }

    var backgroundColor: NSColor {
        switch self {
        case .dark:      return NSColor(red: 0.12, green: 0.12, blue: 0.14, alpha: 1.0)
        case .monokai:   return NSColor(red: 0.15, green: 0.16, blue: 0.13, alpha: 1.0)
        case .solarized: return NSColor(red: 0.00, green: 0.17, blue: 0.21, alpha: 1.0)
        case .github:    return NSColor(red: 0.09, green: 0.11, blue: 0.15, alpha: 1.0)
        case .light:     return NSColor(red: 0.98, green: 0.98, blue: 0.96, alpha: 1.0)
        }
    }

    var cursorColor: NSColor {
        self == .light ? .black : .white
    }

    var baseTextColor: NSColor {
        switch self {
        case .dark:      return NSColor(red: 0.85, green: 0.85, blue: 0.85, alpha: 1.0)
        case .monokai:   return NSColor(red: 0.97, green: 0.97, blue: 0.95, alpha: 1.0)
        case .solarized: return NSColor(red: 0.51, green: 0.58, blue: 0.59, alpha: 1.0)
        case .github:    return NSColor(red: 0.90, green: 0.91, blue: 0.93, alpha: 1.0)
        case .light:     return NSColor(red: 0.15, green: 0.15, blue: 0.15, alpha: 1.0)
        }
    }

    // (strings, comments, keywords, decorators, numbers, builtins)
    var colors: (string: NSColor, comment: NSColor, keyword: NSColor, decorator: NSColor, number: NSColor, builtin: NSColor) {
        switch self {
        case .dark:
            return (
                string:    NSColor(red: 0.56, green: 0.74, blue: 0.56, alpha: 1.0),
                comment:   NSColor(red: 0.50, green: 0.50, blue: 0.50, alpha: 1.0),
                keyword:   NSColor(red: 0.78, green: 0.46, blue: 0.83, alpha: 1.0),
                decorator: NSColor(red: 0.90, green: 0.72, blue: 0.42, alpha: 1.0),
                number:    NSColor(red: 0.73, green: 0.83, blue: 0.55, alpha: 1.0),
                builtin:   NSColor(red: 0.40, green: 0.73, blue: 0.87, alpha: 1.0)
            )
        case .monokai:
            return (
                string:    NSColor(red: 0.90, green: 0.86, blue: 0.45, alpha: 1.0),
                comment:   NSColor(red: 0.46, green: 0.44, blue: 0.37, alpha: 1.0),
                keyword:   NSColor(red: 0.98, green: 0.15, blue: 0.45, alpha: 1.0),
                decorator: NSColor(red: 0.65, green: 0.89, blue: 0.18, alpha: 1.0),
                number:    NSColor(red: 0.68, green: 0.51, blue: 1.00, alpha: 1.0),
                builtin:   NSColor(red: 0.40, green: 0.85, blue: 0.94, alpha: 1.0)
            )
        case .solarized:
            return (
                string:    NSColor(red: 0.16, green: 0.63, blue: 0.60, alpha: 1.0),
                comment:   NSColor(red: 0.40, green: 0.48, blue: 0.51, alpha: 1.0),
                keyword:   NSColor(red: 0.52, green: 0.60, blue: 0.00, alpha: 1.0),
                decorator: NSColor(red: 0.80, green: 0.29, blue: 0.09, alpha: 1.0),
                number:    NSColor(red: 0.82, green: 0.43, blue: 0.12, alpha: 1.0),
                builtin:   NSColor(red: 0.15, green: 0.55, blue: 0.82, alpha: 1.0)
            )
        case .github:
            return (
                string:    NSColor(red: 0.63, green: 0.83, blue: 0.65, alpha: 1.0),
                comment:   NSColor(red: 0.53, green: 0.58, blue: 0.63, alpha: 1.0),
                keyword:   NSColor(red: 1.00, green: 0.48, blue: 0.52, alpha: 1.0),
                decorator: NSColor(red: 0.85, green: 0.69, blue: 0.33, alpha: 1.0),
                number:    NSColor(red: 0.47, green: 0.64, blue: 0.99, alpha: 1.0),
                builtin:   NSColor(red: 0.85, green: 0.69, blue: 0.33, alpha: 1.0)
            )
        case .light:
            return (
                string:    NSColor(red: 0.08, green: 0.47, blue: 0.06, alpha: 1.0),
                comment:   NSColor(red: 0.50, green: 0.50, blue: 0.50, alpha: 1.0),
                keyword:   NSColor(red: 0.67, green: 0.05, blue: 0.57, alpha: 1.0),
                decorator: NSColor(red: 0.55, green: 0.33, blue: 0.00, alpha: 1.0),
                number:    NSColor(red: 0.06, green: 0.33, blue: 0.72, alpha: 1.0),
                builtin:   NSColor(red: 0.20, green: 0.40, blue: 0.64, alpha: 1.0)
            )
        }
    }
}

// MARK: - Syntax Highlighter

/// Lightweight Python syntax highlighter using regex.
enum SyntaxHighlighter {
    static let baseFont = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)

    private static func baseAttributes(for theme: CodeTheme) -> [NSAttributedString.Key: Any] {
        [
            .font: baseFont,
            .foregroundColor: theme.baseTextColor,
        ]
    }

    private static func rules(for theme: CodeTheme) -> [(pattern: String, color: NSColor, options: NSRegularExpression.Options)] {
        let c = theme.colors
        return [
            // Triple-quoted strings (must come first)
            (#"\"\"\"[\s\S]*?\"\"\""#, c.string, [.dotMatchesLineSeparators]),
            (#"'''[\s\S]*?'''"#, c.string, [.dotMatchesLineSeparators]),
            // Single-line strings
            (#""[^"\n]*""#, c.string, []),
            (#"'[^'\n]*'"#, c.string, []),
            // Comments
            (#"#.*$"#, c.comment, [.anchorsMatchLines]),
            // Keywords
            (#"\b(def|class|import|from|return|if|elif|else|for|while|with|as|in|not|and|or|is|True|False|None|self|yield|lambda|try|except|raise|finally|pass|break|continue|assert|global|nonlocal|del|async|await)\b"#,
             c.keyword, []),
            // Decorators
            (#"@\w+"#, c.decorator, []),
            // Numbers
            (#"\b\d+\.?\d*([eE][+-]?\d+)?\b"#, c.number, []),
            // Built-in functions
            (#"\b(print|len|range|zip|enumerate|type|isinstance|super|int|float|str|list|dict|set|tuple|bool|min|max|sum|abs|sorted|reversed|map|filter|any|all|open|hasattr|getattr|setattr)\b"#,
             c.builtin, []),
        ]
    }

    static func highlight(_ code: String, theme: CodeTheme = .dark) -> NSAttributedString {
        let attrs = baseAttributes(for: theme)
        let result = NSMutableAttributedString(string: code, attributes: attrs)
        guard !code.isEmpty else { return result }
        applyRules(to: result, code: code, theme: theme)
        return result
    }

    /// Apply highlighting to existing NSTextStorage (preserves cursor position).
    static func applyHighlighting(to textStorage: NSTextStorage, theme: CodeTheme = .dark) {
        let code = textStorage.string
        guard !code.isEmpty else { return }
        let fullRange = NSRange(location: 0, length: textStorage.length)

        textStorage.beginEditing()
        textStorage.setAttributes(baseAttributes(for: theme), range: fullRange)
        applyRules(to: textStorage, code: code, theme: theme)
        textStorage.endEditing()
    }

    private static func applyRules(to attrStr: NSMutableAttributedString, code: String, theme: CodeTheme) {
        let fullRange = NSRange(code.startIndex..., in: code)
        var coloredRanges: [NSRange] = []

        for (pattern, color, options) in rules(for: theme) {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { continue }
            let matches = regex.matches(in: code, range: fullRange)
            for match in matches {
                let range = match.range
                let overlaps = coloredRanges.contains { NSIntersectionRange($0, range).length > 0 }
                if overlaps { continue }
                attrStr.addAttribute(.foregroundColor, value: color, range: range)
                coloredRanges.append(range)
            }
        }
    }
}


// MARK: - Editable Code Editor

/// Editable code editor with live syntax highlighting.
struct CodeEditorView: NSViewRepresentable {
    @Binding var code: String
    @Binding var selectedText: String
    var theme: CodeTheme = .dark
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
            let theme = parent.theme
            DispatchQueue.main.async {
                self.isUpdating = true
                SyntaxHighlighter.applyHighlighting(to: textStorage, theme: theme)
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
        textView.backgroundColor = theme.backgroundColor
        textView.insertionPointColor = theme.cursorColor
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

        let highlighted = SyntaxHighlighter.highlight(code, theme: theme)
        textView.textStorage?.setAttributedString(highlighted)

        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        let textView = nsView.documentView as! NSTextView

        // Update theme colors
        textView.backgroundColor = theme.backgroundColor
        textView.insertionPointColor = theme.cursorColor

        // Only update if the source changed externally (e.g. file switch) or theme changed
        if textView.string != code && !context.coordinator.isUpdating {
            context.coordinator.isUpdating = true
            let highlighted = SyntaxHighlighter.highlight(code, theme: theme)
            textView.textStorage?.setAttributedString(highlighted)
            context.coordinator.isUpdating = false
        } else if context.coordinator.parent.theme != theme {
            // Theme changed — re-highlight
            context.coordinator.parent = CodeEditorView(code: $code, selectedText: $selectedText, theme: theme, isEditable: isEditable)
            context.coordinator.isUpdating = true
            SyntaxHighlighter.applyHighlighting(to: textView.textStorage!, theme: theme)
            context.coordinator.isUpdating = false
        }
    }
}


// MARK: - Read-only Code View (for output panels)

struct CodeView: NSViewRepresentable {
    let code: String
    var theme: CodeTheme = .dark

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        let textView = scrollView.documentView as! NSTextView

        textView.isEditable = false
        textView.isSelectable = true
        textView.backgroundColor = theme.backgroundColor
        textView.textContainerInset = NSSize(width: 12, height: 12)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false

        let highlighted = SyntaxHighlighter.highlight(code, theme: theme)
        textView.textStorage?.setAttributedString(highlighted)

        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        let textView = nsView.documentView as! NSTextView
        textView.backgroundColor = theme.backgroundColor
        let highlighted = SyntaxHighlighter.highlight(code, theme: theme)
        textView.textStorage?.setAttributedString(highlighted)
    }
}
