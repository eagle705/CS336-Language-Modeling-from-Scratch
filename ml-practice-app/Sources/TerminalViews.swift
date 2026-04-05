import AppKit
import SwiftUI

/// Full interactive terminal view: output + keyboard input in a single NSTextView via PTY.
struct TerminalView: NSViewRepresentable {
    let text: String
    var onKeyPress: ((String) -> Void)?

    private static let bgColor = NSColor(red: 0.10, green: 0.10, blue: 0.12, alpha: 1.0)
    private static let fgColor = NSColor(red: 0.85, green: 0.92, blue: 0.85, alpha: 1.0)
    private static let placeholderColor = NSColor(red: 0.45, green: 0.50, blue: 0.45, alpha: 1.0)
    private static let font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)

    class Coordinator {
        var parent: TerminalView
        var lastLength = 0

        init(_ parent: TerminalView) {
            self.parent = parent
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = TerminalViewFactory.makeScrollView()
        let textView = scrollView.documentView as! TerminalNSTextView

        textView.isEditable = true
        textView.isSelectable = true
        textView.backgroundColor = Self.bgColor
        textView.insertionPointColor = Self.fgColor
        textView.textContainerInset = NSSize(width: 10, height: 10)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.font = Self.font
        textView.isRichText = false
        textView.allowsUndo = false
        textView.onKeyPress = context.coordinator.parent.onKeyPress

        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        let textView = nsView.documentView as! TerminalNSTextView
        textView.onKeyPress = onKeyPress

        // Update text content
        let displayText = text.isEmpty ? "Terminal session not started." : text
        let color = text.isEmpty ? Self.placeholderColor : Self.fgColor
        let attrs: [NSAttributedString.Key: Any] = [
            .font: Self.font,
            .foregroundColor: color,
        ]

        // Only update if text actually changed
        if textView.string != displayText {
            textView.textStorage?.setAttributedString(NSAttributedString(string: displayText, attributes: attrs))
            // Auto-scroll to bottom
            textView.scrollToEndOfDocument(nil)
        }
    }
}

// MARK: - Custom NSTextView that forwards all key events to PTY

class TerminalNSTextView: NSTextView {
    var onKeyPress: ((String) -> Void)?

    // Override to prevent actual text editing — we just capture keys
    override func keyDown(with event: NSEvent) {
        guard let onKeyPress = onKeyPress else {
            super.keyDown(with: event)
            return
        }

        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)

        // Ctrl+C → send ETX (0x03)
        if flags.contains(.control), let chars = event.charactersIgnoringModifiers {
            if chars == "c" {
                onKeyPress("\u{03}")
                return
            }
            if chars == "d" {
                onKeyPress("\u{04}")
                return
            }
            if chars == "z" {
                onKeyPress("\u{1A}")
                return
            }
            if chars == "l" {
                onKeyPress("\u{0C}")
                return
            }
        }

        // Special keys
        switch event.keyCode {
        case 36:  // Return
            onKeyPress("\n")
        case 48:  // Tab
            onKeyPress("\t")
        case 51:  // Backspace
            onKeyPress("\u{7F}")
        case 123: // Left arrow
            onKeyPress("\u{1B}[D")
        case 124: // Right arrow
            onKeyPress("\u{1B}[C")
        case 125: // Down arrow
            onKeyPress("\u{1B}[B")
        case 126: // Up arrow
            onKeyPress("\u{1B}[A")
        case 53:  // Escape
            onKeyPress("\u{1B}")
        default:
            // Regular character input
            if let chars = event.characters, !chars.isEmpty {
                onKeyPress(chars)
            }
        }
    }

    // Prevent inserting text (all input goes through keyDown → PTY)
    override func insertText(_ string: Any, replacementRange: NSRange) {
        // Don't insert — keyDown handles it
    }

    override func doCommand(by selector: Selector) {
        // Prevent default commands (beep, etc.)
    }

    // Allow becoming first responder for key events
    override var acceptsFirstResponder: Bool { true }
}

// Required to make NSTextView.scrollableTextView() use our custom class
extension NSTextView {
    // We override the scrollable factory in makeNSView by replacing documentView
}

// MARK: - Factory helper

struct TerminalViewFactory {
    /// Create a scrollable terminal view with our custom NSTextView subclass.
    static func makeScrollView() -> NSScrollView {
        let textView = TerminalNSTextView()
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)

        let scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.hasVerticalScroller = true
        scrollView.autoresizingMask = [.width, .height]

        return scrollView
    }
}
