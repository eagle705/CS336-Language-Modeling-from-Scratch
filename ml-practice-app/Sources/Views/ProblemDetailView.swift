import SwiftUI

struct ProblemDetailView: View {
    let problem: Problem
    @EnvironmentObject var store: ProblemStore
    @StateObject private var runner = RunnerService()

    @State private var selectedFileId: String?
    @State private var codeContent: String = ""
    @State private var hasUnsavedChanges = false
    @State private var claudePrompt: String = "Review this code. Explain key concepts, find issues, suggest improvements."
    @State private var activePanel: Panel = .claude
    @State private var activeTab: EditorTab = .reference
    @State private var scratchContent: String = "# Write your solution from scratch here\nimport numpy as np\n\n"
    @State private var scratchSaved = false
    @State private var pyenvVenvs: [PyenvVenv] = []
    @State private var editorSelection: String = ""
    @State private var pinnedSelection: String = ""  // preserved when focus leaves editor
    @State private var codeContext: CodeContext = .fullCode
    @State private var showInlineChat = false
    @State private var inlineChatPrompt: String = ""
    @FocusState private var inlineChatFocused: Bool

    enum Panel: String, CaseIterable {
        case claude = "Claude"
        case output = "Output"
    }

    enum EditorTab: String, CaseIterable {
        case reference = "Reference"
        case scratch = "Scratch Pad"
    }

    enum CodeContext: String, CaseIterable {
        case noCode = "No Code"
        case selection = "Selection"
        case fullCode = "Full Code"
    }

    private var selectedFile: ProblemFile? {
        problem.files.first { $0.id == selectedFileId } ?? problem.files.first
    }

    /// The code currently active (reference file or scratch pad).
    private var activeCode: String {
        activeTab == .scratch ? scratchContent : codeContent
    }

    /// Code to send to Claude based on context mode.
    private var codeForClaude: String? {
        switch codeContext {
        case .noCode: return nil
        case .selection: return pinnedSelection.isEmpty ? nil : pinnedSelection
        case .fullCode: return activeCode
        }
    }

    /// File path reflecting the current active tab context.
    private var activeFilePath: String {
        activeTab == .scratch ? scratchFilePath : (selectedFile?.path ?? "")
    }

    /// File label for Claude prompt context.
    private var activeFileLabel: String {
        if activeTab == .scratch {
            return "Scratch Pad (solution.py)"
        }
        return selectedFile?.name ?? "unknown"
    }

    /// Path for saving scratch pad files.
    private var scratchDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ml-practice/scratch/\(problem.id)")
    }

    private var scratchFilePath: String {
        scratchDir.appendingPathComponent("solution.py").path
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            HSplitView {
                editorPane
                    .frame(minWidth: 400)

                rightPanel
                    .frame(minWidth: 300, idealWidth: 420)
            }
        }
        .onAppear {
            loadFile()
            loadScratch()
            pyenvVenvs = PyenvVenv.detect()
        }
        .onChange(of: problem.id) { _, _ in
            selectedFileId = nil
            loadFile()
            loadScratch()
            editorSelection = ""
            pinnedSelection = ""
            codeContext = .fullCode
        }
        .onChange(of: selectedFileId) { _, _ in loadFile() }
        .onChange(of: editorSelection) { _, newValue in
            if !newValue.isEmpty {
                pinnedSelection = newValue
                codeContext = .selection
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(problem.title)
                    .font(.headline)
                HStack(spacing: 8) {
                    Label(problem.category.rawValue, systemImage: "folder")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if hasUnsavedChanges {
                        Text("Modified")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
            }

            Spacer()

            Picker("", selection: Binding(
                get: { problem.state },
                set: { store.updateState(for: problem.id, to: $0) }
            )) {
                ForEach(ProblemState.allCases, id: \.self) { state in
                    Label(state.label, systemImage: state.icon).tag(state)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 260)

            // Python environment picker
            Picker(selection: $store.pythonPath) {
                Label("System python3", systemImage: "apple.terminal")
                    .tag("/usr/bin/env python3")

                if !pyenvVenvs.isEmpty {
                    Divider()
                    ForEach(pyenvVenvs, id: \.path) { venv in
                        Label(venv.name, systemImage: "shippingbox")
                            .tag(venv.path)
                    }
                }
            } label: {
                Image(systemName: "chevron.left.forwardslash.chevron.right")
            }
            .frame(maxWidth: 180)
            .help("Python Environment")

            HStack(spacing: 6) {
                Button {
                    saveCurrentFile()
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
                .keyboardShortcut("s", modifiers: .command)
                .disabled(!hasUnsavedChanges)
                .help("Save (Cmd+S)")

                Button {
                    saveCurrentFile()
                    let path = activeTab == .scratch ? scratchFilePath : (selectedFile?.path ?? "")
                    if !path.isEmpty {
                        runner.runPython(filePath: path, pythonPath: store.pythonPath)
                        activePanel = .output
                    }
                } label: {
                    Label("Run", systemImage: runner.isPythonRunning ? "stop.fill" : "play.fill")
                }
                .keyboardShortcut("r", modifiers: .command)
                .help("Run (Cmd+R)")

                Button {
                    if let file = selectedFile {
                        runner.openClaudeInTerminal(filePath: file.path)
                    }
                } label: {
                    Image(systemName: "terminal")
                }
                .help("Open Claude in Terminal")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }

    // MARK: - Editor Pane

    private var editorPane: some View {
        VStack(spacing: 0) {
            // Top: Reference / Scratch Pad toggle + file tabs
            HStack(spacing: 0) {
                // Editor mode tabs
                ForEach(EditorTab.allCases, id: \.self) { tab in
                    Button {
                        if hasUnsavedChanges { saveCurrentFile() }
                        activeTab = tab
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: tab == .reference ? "doc.text" : "pencil.and.outline")
                                .font(.caption)
                            Text(tab.rawValue)
                                .font(.system(size: 12, weight: .medium))
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 7)
                        .background(activeTab == tab ? Color.accentColor.opacity(0.2) : Color.clear)
                    }
                    .buttonStyle(.plain)

                    Divider().frame(height: 20)
                }

                // File tabs (only in reference mode, if multiple files)
                if activeTab == .reference && problem.files.count > 1 {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 0) {
                            ForEach(problem.files) { file in
                                Button {
                                    if hasUnsavedChanges { saveCurrentFile() }
                                    selectedFileId = file.id
                                } label: {
                                    Text(file.name + ".py")
                                        .font(.system(size: 11, design: .monospaced))
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 7)
                                        .background(
                                            (selectedFileId ?? problem.files.first?.id) == file.id
                                                ? Color.secondary.opacity(0.15)
                                                : Color.clear
                                        )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }

                Spacer()
            }
            .background(.bar)

            Divider()

            // Editor with inline chat overlay
            ZStack(alignment: .bottom) {
                if activeTab == .reference {
                    CodeEditorView(code: $codeContent, selectedText: $editorSelection)
                        .onChange(of: codeContent) { _, _ in hasUnsavedChanges = true }
                } else {
                    CodeEditorView(code: $scratchContent, selectedText: $editorSelection)
                        .onChange(of: scratchContent) { _, _ in hasUnsavedChanges = true }
                }

                // Inline chat popup (Cmd+L)
                if showInlineChat {
                    inlineChatView
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                        .padding(.horizontal, 16)
                        .padding(.bottom, 12)
                }
            }
            .animation(.easeInOut(duration: 0.2), value: showInlineChat)
        }
        // Cmd+L shortcut — hidden button to trigger inline chat
        .background {
            Button("") {
                if !editorSelection.isEmpty {
                    pinnedSelection = editorSelection
                }
                showInlineChat = true
                // Delay focus slightly so the TextField is in the view hierarchy
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    inlineChatFocused = true
                }
            }
            .keyboardShortcut("l", modifiers: .command)
            .opacity(0)
        }
    }

    // MARK: - Inline Chat (Cmd+L)

    private var inlineChatView: some View {
        VStack(spacing: 0) {
            // Selection preview
            if !pinnedSelection.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "text.cursor")
                        .font(.system(size: 10))
                        .foregroundStyle(.orange)
                    Text("\(pinnedSelection.components(separatedBy: "\n").count) lines selected from \(activeTab.rawValue)")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        showInlineChat = false
                        inlineChatPrompt = ""
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, 4)
            } else {
                HStack {
                    Text("Ask Claude")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        showInlineChat = false
                        inlineChatPrompt = ""
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, 4)
            }

            HStack(spacing: 8) {
                TextField("Ask about this code... (Enter to send, Esc to close)", text: $inlineChatPrompt)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .padding(8)
                    .focused($inlineChatFocused)
                    .onSubmit { sendInlineChat() }

                Button {
                    sendInlineChat()
                } label: {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 12))
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(inlineChatPrompt.isEmpty || runner.isClaudeRunning)
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 10)
        }
        .background {
            RoundedRectangle(cornerRadius: 10)
                .fill(.ultraThickMaterial)
                .shadow(color: .black.opacity(0.3), radius: 8, y: 2)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.accentColor.opacity(0.4), lineWidth: 1)
        }
    }

    // MARK: - Right Panel

    private var rightPanel: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                ForEach(Panel.allCases, id: \.self) { panel in
                    Button {
                        activePanel = panel
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: panel == .claude ? "brain" : "text.below.photo")
                                .font(.caption)
                            Text(panel.rawValue)
                                .font(.system(size: 12, weight: .medium))
                            if panel == .claude && runner.isClaudeRunning {
                                ProgressView().scaleEffect(0.5).frame(width: 12, height: 12)
                            }
                            if panel == .output && runner.isPythonRunning {
                                ProgressView().scaleEffect(0.5).frame(width: 12, height: 12)
                            }
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 7)
                        .background(activePanel == panel ? Color.accentColor.opacity(0.15) : Color.clear)
                    }
                    .buttonStyle(.plain)
                }
                Spacer()
            }
            .background(.bar)

            Divider()

            switch activePanel {
            case .claude:
                claudePanel
            case .output:
                outputPanel
            }
        }
    }

    // MARK: - Claude Panel

    private var claudePanel: some View {
        VStack(spacing: 0) {
            // Code context selector
            HStack(spacing: 8) {
                Text("Context:")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                Picker("", selection: $codeContext) {
                    Text("No Code").tag(CodeContext.noCode)
                    Text("Selection").tag(CodeContext.selection)
                    Text("Full Code (\(activeTab.rawValue))").tag(CodeContext.fullCode)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)

                Spacer()

                // Show selection info
                if codeContext == .selection {
                    if pinnedSelection.isEmpty {
                        Text("No selection - drag to select code in editor")
                            .font(.system(size: 10))
                            .foregroundStyle(.orange)
                    } else {
                        Text("\(pinnedSelection.components(separatedBy: "\n").count) lines selected")
                            .font(.system(size: 10))
                            .foregroundStyle(.green)
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

            Divider()

            // Prompt input
            HStack(spacing: 8) {
                TextField("Ask Claude anything...", text: $claudePrompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .lineLimit(1...6)
                    .padding(8)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(8)

                Button {
                    sendToClaude(prompt: claudePrompt)
                } label: {
                    Image(systemName: "paperplane.fill")
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.borderedProminent)
                .disabled(runner.isClaudeRunning)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding(10)

            Divider()

            // Claude output — Markdown rendered
            if runner.claudeOutput.isEmpty {
                VStack(spacing: 8) {
                    Spacer()
                    Image(systemName: "brain")
                        .font(.system(size: 28))
                        .foregroundStyle(.secondary)
                    Text("Claude responses will appear here.")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                    Text("Type a question and press Cmd+Enter,\nor select code and press Cmd+L.")
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: NSColor(red: 0.13, green: 0.13, blue: 0.15, alpha: 1.0)))
            } else if runner.isClaudeRunning {
                // While streaming, show plain text (updates in real-time)
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(runner.claudeOutput)
                            .font(.system(size: 13))
                            .foregroundColor(Color(nsColor: NSColor(red: 0.92, green: 0.92, blue: 0.92, alpha: 1.0)))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .id("bottom")
                    }
                    .onChange(of: runner.claudeOutput) { _, _ in
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
                .background(Color(nsColor: NSColor(red: 0.13, green: 0.13, blue: 0.15, alpha: 1.0)))
            } else {
                // Finished — render as Markdown
                MarkdownWebView(markdown: runner.claudeOutput)
            }
        }
    }

    // MARK: - Output Panel

    private var outputPanel: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Python Output")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if runner.isPythonRunning {
                    Button("Stop") { runner.stopPython() }
                        .controlSize(.small)
                }
                Button("Clear") { runner.pythonOutput = "" }
                    .controlSize(.small)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            // Output — bright text on dark background
            ScrollViewReader { proxy in
                ScrollView {
                    Text(runner.pythonOutput.isEmpty
                         ? "Run the file (Cmd+R) to see output here."
                         : runner.pythonOutput)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundColor(runner.pythonOutput.isEmpty
                                         ? .gray
                                         : Color(nsColor: NSColor(red: 0.90, green: 0.95, blue: 0.90, alpha: 1.0)))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                        .id("bottom")
                }
                .onChange(of: runner.pythonOutput) { _, _ in
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .background(Color(nsColor: NSColor(red: 0.13, green: 0.13, blue: 0.15, alpha: 1.0)))
        }
    }

    // MARK: - Claude Helpers

    private func sendToClaude(prompt: String) {
        let dir = activeTab == .scratch
            ? scratchDir.path
            : (selectedFile.map { URL(fileURLWithPath: $0.path).deletingLastPathComponent().path })

        let code = codeForClaude
        // Show context info in output
        var contextInfo = "Source: \(activeFileLabel)"
        switch codeContext {
        case .noCode: contextInfo += " (no code attached)"
        case .selection: contextInfo += " (selection: \(pinnedSelection.components(separatedBy: "\n").count) lines)"
        case .fullCode: contextInfo += " (full code)"
        }
        runner.claudeOutput = "\(contextInfo)\n\n"

        runner.askClaude(
            prompt: prompt,
            fileLabel: activeFileLabel,
            workingDirectory: dir,
            code: code
        )
        activePanel = .claude
    }

    private func sendInlineChat() {
        guard !inlineChatPrompt.isEmpty else { return }
        // Pin current selection before sending
        if !editorSelection.isEmpty {
            pinnedSelection = editorSelection
        }
        let code = pinnedSelection.isEmpty ? nil : pinnedSelection
        let dir = activeTab == .scratch
            ? scratchDir.path
            : (selectedFile.map { URL(fileURLWithPath: $0.path).deletingLastPathComponent().path })

        runner.askClaude(
            prompt: inlineChatPrompt,
            fileLabel: activeFileLabel,
            workingDirectory: dir,
            code: code
        )
        activePanel = .claude
        showInlineChat = false
        inlineChatPrompt = ""
    }

    // MARK: - File I/O

    private func loadFile() {
        guard let file = selectedFile else {
            codeContent = "// No file selected"
            return
        }
        codeContent = (try? String(contentsOfFile: file.path, encoding: .utf8)) ?? "// Failed to read file"
        hasUnsavedChanges = false
    }

    private func loadScratch() {
        if FileManager.default.fileExists(atPath: scratchFilePath) {
            scratchContent = (try? String(contentsOfFile: scratchFilePath, encoding: .utf8)) ?? scratchContent
        }
    }

    private func saveCurrentFile() {
        if activeTab == .reference {
            guard let file = selectedFile else { return }
            try? codeContent.write(toFile: file.path, atomically: true, encoding: .utf8)
        } else {
            try? FileManager.default.createDirectory(at: scratchDir, withIntermediateDirectories: true)
            try? scratchContent.write(toFile: scratchFilePath, atomically: true, encoding: .utf8)
        }
        hasUnsavedChanges = false
    }
}
