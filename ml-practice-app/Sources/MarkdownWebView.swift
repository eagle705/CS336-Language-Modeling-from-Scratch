import SwiftUI
import WebKit

/// Renders Markdown content as styled HTML in a WKWebView (dark theme).
struct MarkdownWebView: NSViewRepresentable {
    let markdown: String

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")  // transparent bg
        loadHTML(into: webView)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        loadHTML(into: webView)
    }

    private func loadHTML(into webView: WKWebView) {
        let html = Self.wrapInHTML(markdown: markdown)
        webView.loadHTMLString(html, baseURL: nil)
    }

    // MARK: - Markdown to HTML

    static func wrapInHTML(markdown: String) -> String {
        let bodyHTML = markdownToHTML(markdown)
        return """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
            font-size: 13px;
            line-height: 1.6;
            color: #e8e8e8;
            background: #212125;
            padding: 14px;
            -webkit-font-smoothing: antialiased;
        }
        h1, h2, h3, h4 {
            color: #f0f0f0;
            margin-top: 16px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        h1 { font-size: 20px; border-bottom: 1px solid #3a3a3e; padding-bottom: 6px; }
        h2 { font-size: 17px; border-bottom: 1px solid #3a3a3e; padding-bottom: 4px; }
        h3 { font-size: 15px; }
        h4 { font-size: 13px; }
        p { margin-bottom: 10px; }
        ul, ol { margin-bottom: 10px; padding-left: 24px; }
        li { margin-bottom: 4px; }
        a { color: #6cb4ee; text-decoration: none; }
        a:hover { text-decoration: underline; }
        strong { color: #f5f5f5; font-weight: 600; }
        em { font-style: italic; color: #d0d0d0; }
        code {
            font-family: "SF Mono", Menlo, Monaco, monospace;
            font-size: 12px;
            background: #2a2a2e;
            color: #e8b86d;
            padding: 1px 5px;
            border-radius: 4px;
        }
        pre {
            background: #1a1a1e;
            border: 1px solid #3a3a3e;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            overflow-x: auto;
        }
        pre code {
            background: none;
            color: #c8d0d8;
            padding: 0;
            font-size: 12px;
            line-height: 1.5;
        }
        blockquote {
            border-left: 3px solid #6cb4ee;
            margin: 10px 0;
            padding: 4px 12px;
            color: #b0b0b0;
            background: #1e1e22;
            border-radius: 0 6px 6px 0;
        }
        hr {
            border: none;
            border-top: 1px solid #3a3a3e;
            margin: 16px 0;
        }
        table {
            border-collapse: collapse;
            margin: 10px 0;
            width: 100%;
        }
        th, td {
            border: 1px solid #3a3a3e;
            padding: 6px 10px;
            text-align: left;
        }
        th { background: #2a2a2e; font-weight: 600; }

        /* Python syntax highlighting in code blocks */
        .kw { color: #c586c0; }
        .str { color: #8ec07c; }
        .num { color: #b5cea8; }
        .cmt { color: #6a9955; }
        .fn { color: #dcdcaa; }
        .bi { color: #4ec9b0; }
        </style>
        </head>
        <body>\(bodyHTML)</body>
        </html>
        """
    }

    /// Simple Markdown → HTML converter handling common elements.
    static func markdownToHTML(_ md: String) -> String {
        let lines = md.components(separatedBy: "\n")
        var html = ""
        var i = 0
        var inList = false
        var listType = ""  // "ul" or "ol"

        while i < lines.count {
            let line = lines[i]

            // Fenced code blocks
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                let lang = line.trimmingCharacters(in: .whitespaces)
                    .dropFirst(3).trimmingCharacters(in: .whitespaces)
                if inList { html += "</\(listType)>"; inList = false }
                var codeLines: [String] = []
                i += 1
                while i < lines.count {
                    if lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                        i += 1; break
                    }
                    codeLines.append(lines[i])
                    i += 1
                }
                let code = escapeHTML(codeLines.joined(separator: "\n"))
                let highlighted = lang.lowercased().contains("python") || lang.isEmpty
                    ? highlightPython(code) : code
                html += "<pre><code>\(highlighted)</code></pre>\n"
                continue
            }

            // Headers
            if line.hasPrefix("#### ") {
                if inList { html += "</\(listType)>"; inList = false }
                html += "<h4>\(inlineFormat(String(line.dropFirst(5))))</h4>\n"
                i += 1; continue
            }
            if line.hasPrefix("### ") {
                if inList { html += "</\(listType)>"; inList = false }
                html += "<h3>\(inlineFormat(String(line.dropFirst(4))))</h3>\n"
                i += 1; continue
            }
            if line.hasPrefix("## ") {
                if inList { html += "</\(listType)>"; inList = false }
                html += "<h2>\(inlineFormat(String(line.dropFirst(3))))</h2>\n"
                i += 1; continue
            }
            if line.hasPrefix("# ") {
                if inList { html += "</\(listType)>"; inList = false }
                html += "<h1>\(inlineFormat(String(line.dropFirst(2))))</h1>\n"
                i += 1; continue
            }

            // Horizontal rule
            if line.trimmingCharacters(in: .whitespaces).range(of: #"^[-*_]{3,}$"#,
                options: .regularExpression) != nil {
                if inList { html += "</\(listType)>"; inList = false }
                html += "<hr>\n"
                i += 1; continue
            }

            // Blockquote
            if line.hasPrefix("> ") {
                if inList { html += "</\(listType)>"; inList = false }
                html += "<blockquote>\(inlineFormat(String(line.dropFirst(2))))</blockquote>\n"
                i += 1; continue
            }

            // Unordered list
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("- ")
                || line.trimmingCharacters(in: .whitespaces).hasPrefix("* ") {
                if !inList || listType != "ul" {
                    if inList { html += "</\(listType)>" }
                    html += "<ul>\n"; inList = true; listType = "ul"
                }
                let content = line.trimmingCharacters(in: .whitespaces)
                let stripped = String(content.dropFirst(2))
                html += "<li>\(inlineFormat(stripped))</li>\n"
                i += 1; continue
            }

            // Ordered list
            if line.trimmingCharacters(in: .whitespaces)
                .range(of: #"^\d+\.\s"#, options: .regularExpression) != nil {
                if !inList || listType != "ol" {
                    if inList { html += "</\(listType)>" }
                    html += "<ol>\n"; inList = true; listType = "ol"
                }
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if let dotRange = trimmed.range(of: #"^\d+\.\s"#, options: .regularExpression) {
                    let content = String(trimmed[dotRange.upperBound...])
                    html += "<li>\(inlineFormat(content))</li>\n"
                }
                i += 1; continue
            }

            // Close list if no longer in list context
            if inList {
                html += "</\(listType)>\n"
                inList = false
            }

            // Empty line
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                i += 1; continue
            }

            // Regular paragraph
            html += "<p>\(inlineFormat(line))</p>\n"
            i += 1
        }

        if inList { html += "</\(listType)>\n" }
        return html
    }

    /// Process inline formatting: bold, italic, inline code, links.
    static func inlineFormat(_ text: String) -> String {
        var s = escapeHTML(text)

        // Inline code (must be before bold/italic to avoid conflicts)
        s = s.replacingOccurrences(
            of: #"`([^`]+)`"#,
            with: "<code>$1</code>",
            options: .regularExpression
        )
        // Bold + italic
        s = s.replacingOccurrences(
            of: #"\*\*\*(.+?)\*\*\*"#,
            with: "<strong><em>$1</em></strong>",
            options: .regularExpression
        )
        // Bold
        s = s.replacingOccurrences(
            of: #"\*\*(.+?)\*\*"#,
            with: "<strong>$1</strong>",
            options: .regularExpression
        )
        // Italic
        s = s.replacingOccurrences(
            of: #"\*(.+?)\*"#,
            with: "<em>$1</em>",
            options: .regularExpression
        )
        // Links [text](url)
        s = s.replacingOccurrences(
            of: #"\[([^\]]+)\]\(([^)]+)\)"#,
            with: "<a href=\"$2\">$1</a>",
            options: .regularExpression
        )
        return s
    }

    static func escapeHTML(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;")
         .replacingOccurrences(of: "<", with: "&lt;")
         .replacingOccurrences(of: ">", with: "&gt;")
         .replacingOccurrences(of: "\"", with: "&quot;")
    }

    /// Basic Python syntax highlighting for code blocks (already HTML-escaped).
    static func highlightPython(_ code: String) -> String {
        var s = code

        // Comments
        s = s.replacingOccurrences(
            of: #"(#.*?)(\n|$)"#,
            with: "<span class=\"cmt\">$1</span>$2",
            options: .regularExpression
        )

        // Strings (double and single quoted, simple)
        s = s.replacingOccurrences(
            of: #"(&quot;.*?&quot;)"#,
            with: "<span class=\"str\">$1</span>",
            options: .regularExpression
        )
        s = s.replacingOccurrences(
            of: #"(&#39;.*?&#39;)"#,
            with: "<span class=\"str\">$1</span>",
            options: .regularExpression
        )
        // Also handle 'text' in escaped form
        s = s.replacingOccurrences(
            of: #"('.*?')"#,
            with: "<span class=\"str\">$1</span>",
            options: .regularExpression
        )

        // Keywords
        let keywords = ["def", "class", "import", "from", "return", "if", "elif", "else",
                        "for", "while", "with", "as", "in", "not", "and", "or", "is",
                        "True", "False", "None", "self", "yield", "lambda", "try", "except",
                        "raise", "finally", "pass", "break", "continue", "assert",
                        "async", "await", "del", "global", "nonlocal"]
        for kw in keywords {
            s = s.replacingOccurrences(
                of: "\\b\(kw)\\b",
                with: "<span class=\"kw\">\(kw)</span>",
                options: .regularExpression
            )
        }

        // Numbers
        s = s.replacingOccurrences(
            of: #"\b(\d+\.?\d*)\b"#,
            with: "<span class=\"num\">$1</span>",
            options: .regularExpression
        )

        return s
    }
}
