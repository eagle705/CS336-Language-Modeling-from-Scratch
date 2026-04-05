import SwiftUI

struct ProblemDetailView: View {
    let problem: Problem
    @EnvironmentObject var store: ProblemStore
    @StateObject private var runner = RunnerService()

    @State private var selectedFileId: String?
    @State private var codeContent: String = ""
    @State private var hasUnsavedChanges = false
    @State private var claudePrompt: String = ""
    @State private var activePanel: Panel = .claude
    @State private var claudeMode: ClaudeMode = .explain
    @State private var claudeCache: [ClaudeMode: String] = [:]
    @State private var activeTab: EditorTab = .reference
    @State private var scratchContent: String = "# Write your solution from scratch here\nimport numpy as np\n\n"
    @State private var scratchSaved = false
    @State private var pyenvVenvs: [PyenvVenv] = []
    @State private var editorSelection: String = ""
    @State private var pinnedSelection: String = ""  // preserved when focus leaves editor
    @State private var codeContext: CodeContext = .fullCode
    @State private var showInlineChat = false
    @State private var scriptArgs: String = ""
    @State private var showArgsInput = false
    @State private var inlineChatPrompt: String = ""
    @FocusState private var inlineChatFocused: Bool

    enum Panel: String, CaseIterable {
        case claude = "Claude"
        case output = "Output"
        case terminal = "Terminal"
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

    enum ClaudeMode: String, CaseIterable, Hashable {
        case explain = "Explain"
        case review = "Code Review"
        case interview = "Mock Interview"
        case custom = "Custom"

        var icon: String {
            switch self {
            case .explain: return "text.book.closed"
            case .review: return "checkmark.circle"
            case .interview: return "person.fill.questionmark"
            case .custom: return "text.bubble"
            }
        }

        var prompt: String {
            switch self {
            case .explain:
                return "Explain this code step by step, as if teaching a beginner. Include what each part does and why. Answer in Korean."
            case .review:
                return "Review this code. Explain key concepts, find issues, suggest improvements. Answer in Korean."
            case .interview:
                return """
                You are an ML interview expert. Based on the following code, generate exactly 3 likely interview questions that could be asked about the concepts in this code.

                For each question:
                1. State the question clearly
                2. Provide a concise but thorough model answer
                3. Note common mistakes or follow-up questions the interviewer might ask

                Format as:
                ## Q1: [question]
                **Answer:** [answer]
                **Follow-up:** [follow-up]

                ## Q2: ...
                ## Q3: ...

                Answer in Korean.
                """
            case .custom:
                return ""
            }
        }
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
            claudeCache = [:]
            runner.claudeOutput = ""
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
        VStack(spacing: 0) {
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

            // Theme picker
            Picker(selection: $store.codeThemeRaw) {
                ForEach(CodeTheme.allCases) { theme in
                    Text(theme.rawValue).tag(theme.rawValue)
                }
            } label: {
                Image(systemName: "paintpalette")
            }
            .frame(maxWidth: 140)
            .help("Code Theme")

            // Execution environment picker
            executionPicker

            HStack(spacing: 6) {
                Button {
                    saveCurrentFile()
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
                .keyboardShortcut("s", modifiers: .command)
                .disabled(!hasUnsavedChanges)
                .help("Save (Cmd+S)")

                // Run button + args toggle
                Button {
                    runCurrentFile()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: runner.isPythonRunning ? "stop.fill" : "play.fill")
                        Text("Run")
                        if store.isRemoteExecution {
                            Image(systemName: "network")
                                .font(.system(size: 9))
                        }
                    }
                }
                .keyboardShortcut("r", modifiers: .command)
                .help(store.isRemoteExecution ? "Run on remote (Cmd+R)" : "Run (Cmd+R)")

                // Args toggle button
                Button {
                    showArgsInput.toggle()
                } label: {
                    Image(systemName: "text.append")
                        .foregroundColor(scriptArgs.isEmpty ? Color.secondary : Color.orange)
                }
                .help("Script arguments (sys.argv)")

                Button {
                    activePanel = .terminal
                    if !runner.isTerminalRunning {
                        startTerminalSession()
                    }
                } label: {
                    Image(systemName: "terminal")
                }
                .help("Open Terminal")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)

        // Script args input bar
        if showArgsInput {
            HStack(spacing: 8) {
                Text("sys.argv:")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                TextField("e.g. dtensor --batch-size 32", text: $scriptArgs)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12, design: .monospaced))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(4)
                if !scriptArgs.isEmpty {
                    Button {
                        scriptArgs = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        }
        } // VStack
    }

    private func runCurrentFile() {
        if runner.isPythonRunning {
            runner.stopPython()
            return
        }
        saveCurrentFile()
        let path = activeTab == .scratch ? scratchFilePath : (selectedFile?.path ?? "")
        guard !path.isEmpty else { return }
        if store.isRemoteExecution {
            runner.runPythonRemote(filePath: path, config: makeRemoteConfig(containerOverride: false), scriptArgs: scriptArgs)
        } else {
            runner.runPython(filePath: path, pythonPath: store.pythonPath, scriptArgs: scriptArgs)
        }
        activePanel = .output
    }

    // MARK: - Execution Environment Picker

    private var executionPicker: some View {
        Menu {
            // Mode toggle
            Section("Execution Mode") {
                Button {
                    store.execModeRaw = "local"
                } label: {
                    Label("Local", systemImage: store.execModeRaw == "local" ? "checkmark" : "")
                }

                Button {
                    store.execModeRaw = "remote"
                } label: {
                    Label("Remote (SSH)", systemImage: store.execModeRaw == "remote" ? "checkmark" : "")
                }
            }

            Divider()

            if store.isRemoteExecution {
                // Remote info
                Section("Remote: \(store.sshUser)@\(store.sshHost)") {
                    if !store.containerName.isEmpty {
                        Label("\(store.containerRuntime): \(store.containerName)", systemImage: "shippingbox")
                            .disabled(true)
                    }
                    Label("Python: \(store.remotePythonPath)", systemImage: "chevron.left.forwardslash.chevron.right")
                        .disabled(true)
                }
            } else {
                // Local pyenv selection
                Section("Python Environment") {
                    Button {
                        store.pythonPath = "/usr/bin/env python3"
                    } label: {
                        HStack {
                            Label("System python3", systemImage: "apple.terminal")
                            if store.pythonPath == "/usr/bin/env python3" { Image(systemName: "checkmark") }
                        }
                    }

                    ForEach(pyenvVenvs, id: \.path) { venv in
                        Button {
                            store.pythonPath = venv.path
                        } label: {
                            HStack {
                                Label("\(venv.name) (\(venv.version))", systemImage: "shippingbox")
                                if store.pythonPath == venv.path { Image(systemName: "checkmark") }
                            }
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: store.isRemoteExecution ? "network" : "laptopcomputer")
                    .font(.system(size: 11))
                if store.isRemoteExecution {
                    Text(store.sshHost.isEmpty ? "Remote" : "\(store.sshUser.prefix(8))@\(store.sshHost.prefix(15))")
                        .font(.system(size: 11))
                        .lineLimit(1)
                } else {
                    let envName = pyenvVenvs.first(where: { $0.path == store.pythonPath })?.name ?? "python3"
                    Text(envName)
                        .font(.system(size: 11))
                        .lineLimit(1)
                }
                Image(systemName: "chevron.down")
                    .font(.system(size: 8))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(store.isRemoteExecution
                          ? Color.blue.opacity(0.15)
                          : Color(nsColor: .controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(store.isRemoteExecution ? Color.blue.opacity(0.3) : Color.clear, lineWidth: 1)
            )
        }
        .menuStyle(.borderlessButton)
        .frame(maxWidth: 200)
        .help(store.isRemoteExecution ? "Remote: \(store.sshUser)@\(store.sshHost)" : "Local Python Environment")
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
                    CodeEditorWithFind(code: $codeContent, selectedText: $editorSelection, theme: store.codeTheme)
                        .onChange(of: codeContent) { _, _ in hasUnsavedChanges = true }
                } else {
                    CodeEditorWithFind(code: $scratchContent, selectedText: $editorSelection, theme: store.codeTheme)
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

    private func panelIcon(_ panel: Panel) -> String {
        switch panel {
        case .claude: return "brain"
        case .output: return "text.below.photo"
        case .terminal: return "terminal"
        }
    }

    private func panelIsActive(_ panel: Panel) -> Bool {
        switch panel {
        case .claude: return runner.isClaudeRunning
        case .output: return runner.isPythonRunning
        case .terminal: return runner.isTerminalRunning
        }
    }

    private var rightPanel: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                ForEach(Panel.allCases, id: \.self) { panel in
                    Button {
                        activePanel = panel
                        if panel == .terminal && !runner.isTerminalRunning {
                            startTerminalSession()
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: panelIcon(panel))
                                .font(.caption)
                            Text(panel.rawValue)
                                .font(.system(size: 12, weight: .medium))
                            if panelIsActive(panel) {
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
            case .terminal:
                terminalPanel
            }
        }
    }

    // MARK: - Claude Panel

    /// The output to display: live runner output during streaming, cached result for current mode otherwise.
    private var displayedClaudeOutput: String {
        if runner.isClaudeRunning {
            return runner.claudeOutput
        }
        return claudeCache[claudeMode] ?? ""
    }

    private var claudePanel: some View {
        VStack(spacing: 0) {
            // Mode tabs
            HStack(spacing: 0) {
                ForEach(ClaudeMode.allCases, id: \.self) { mode in
                    Button {
                        claudeMode = mode
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: mode.icon)
                                .font(.system(size: 10))
                            Text(mode.rawValue)
                                .font(.system(size: 11, weight: .medium))
                            // Show dot if cached result exists
                            if claudeCache[mode] != nil {
                                Circle()
                                    .fill(.green)
                                    .frame(width: 5, height: 5)
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(claudeMode == mode ? Color.accentColor.opacity(0.2) : Color.clear)
                    }
                    .buttonStyle(.plain)

                    if mode != ClaudeMode.allCases.last {
                        Divider().frame(height: 16)
                    }
                }
                Spacer()
            }
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

            Divider()

            // Context + action row
            HStack(spacing: 8) {
                // Code context picker
                Picker("", selection: $codeContext) {
                    Text("No Code").tag(CodeContext.noCode)
                    Text("Selection").tag(CodeContext.selection)
                    Text("Full (\(activeTab.rawValue))").tag(CodeContext.fullCode)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 280)

                if codeContext == .selection && !pinnedSelection.isEmpty {
                    Text("\(pinnedSelection.components(separatedBy: "\n").count) lines")
                        .font(.system(size: 10))
                        .foregroundStyle(.green)
                }

                Spacer()

                // Run/regenerate button
                Button {
                    runCurrentMode()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: runner.isClaudeRunning ? "stop.fill" : "arrow.clockwise")
                            .font(.system(size: 10))
                        Text(runner.isClaudeRunning ? "Stop" : "Run")
                            .font(.system(size: 11))
                    }
                }
                .controlSize(.small)
                .disabled(claudeMode == .custom && claudePrompt.isEmpty)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)

            // Custom prompt input (only for Custom mode)
            if claudeMode == .custom {
                Divider()
                HStack(spacing: 8) {
                    TextField("Ask Claude anything...", text: $claudePrompt, axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .lineLimit(1...6)
                        .padding(8)
                        .background(Color(nsColor: .controlBackgroundColor))
                        .cornerRadius(8)

                    Button {
                        runCurrentMode()
                    } label: {
                        Image(systemName: "paperplane.fill")
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(runner.isClaudeRunning || claudePrompt.isEmpty)
                    .keyboardShortcut(.return, modifiers: .command)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
            }

            Divider()

            // Claude output
            if displayedClaudeOutput.isEmpty && !runner.isClaudeRunning {
                VStack(spacing: 8) {
                    Spacer()
                    Image(systemName: claudeMode.icon)
                        .font(.system(size: 28))
                        .foregroundColor(Color(nsColor: NSColor(red: 0.55, green: 0.55, blue: 0.60, alpha: 1.0)))
                    Text(claudeMode == .custom
                         ? "Type a question and press Cmd+Enter"
                         : "Press Run or Cmd+Enter to generate")
                        .font(.system(size: 13))
                        .foregroundColor(Color(nsColor: NSColor(red: 0.60, green: 0.60, blue: 0.65, alpha: 1.0)))
                    Text("Cmd+L for inline chat")
                        .font(.system(size: 11))
                        .foregroundColor(Color(nsColor: NSColor(red: 0.45, green: 0.45, blue: 0.50, alpha: 1.0)))
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: NSColor(red: 0.13, green: 0.13, blue: 0.15, alpha: 1.0)))
            } else if runner.isClaudeRunning {
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
                MarkdownWebView(markdown: displayedClaudeOutput)
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

    // MARK: - Terminal Panel

    private var terminalPanel: some View {
        VStack(spacing: 0) {
            HStack {
                Text(runner.isTerminalRunning
                     ? (store.isRemoteExecution ? "\(store.sshUser)@\(store.sshHost)" : "Local Shell")
                     : "Terminal")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if runner.isTerminalRunning {
                    Button("Disconnect") { runner.stopTerminal() }
                        .controlSize(.small)
                } else {
                    Button("Connect") { startTerminalSession() }
                        .controlSize(.small)
                }
                Button("Clear") { runner.terminalOutput = "" }
                    .controlSize(.small)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            // Integrated terminal: output + direct keyboard input
            TerminalView(text: runner.terminalOutput) { key in
                runner.sendRawToTerminal(key)
            }
        }
    }

    private func makeRemoteConfig(containerOverride: Bool = true) -> RunnerService.RemoteConfig {
        RunnerService.RemoteConfig(
            host: store.sshHost,
            port: store.sshPort,
            user: store.sshUser,
            keyPath: store.sshKeyPath,
            containerRuntime: containerOverride && store.containerRuntime == "none" ? "" : store.containerRuntime,
            containerName: containerOverride && store.containerRuntime == "none" ? "" : store.containerName,
            pythonPath: store.remotePythonPath,
            workDir: store.remoteWorkDir,
            portForwardArgs: store.portForwardArgs,
            controlMaster: store.sshControlMaster,
            controlSocketPath: store.controlSocketPath
        )
    }

    private func startTerminalSession() {
        let dir = activeTab == .scratch
            ? scratchDir.path
            : (selectedFile.map { URL(fileURLWithPath: $0.path).deletingLastPathComponent().path })

        if store.isRemoteExecution {
            runner.startTerminal(workingDirectory: dir, remoteConfig: makeRemoteConfig())
        } else {
            runner.startTerminal(workingDirectory: dir)
        }

    }

    // MARK: - Claude Helpers

    private func runCurrentMode() {
        if runner.isClaudeRunning {
            runner.stopClaude()
            return
        }
        let prompt = claudeMode == .custom ? claudePrompt : claudeMode.prompt
        guard !prompt.isEmpty else { return }
        if claudeMode != .custom {
            codeContext = .fullCode
        }
        sendToClaude(prompt: prompt, forMode: claudeMode)
    }

    private func sendToClaude(prompt: String, forMode mode: ClaudeMode? = nil) {
        let dir = activeTab == .scratch
            ? scratchDir.path
            : (selectedFile.map { URL(fileURLWithPath: $0.path).deletingLastPathComponent().path })

        let code = codeForClaude
        let targetMode = mode ?? .custom

        // Set mode so we see streaming output
        claudeMode = targetMode

        runner.askClaude(
            prompt: prompt,
            fileLabel: activeFileLabel,
            workingDirectory: dir,
            code: code
        )
        activePanel = .claude

        // Watch for completion and cache the result
        let currentMode = targetMode
        Task { @MainActor in
            // Poll until done (runner.isClaudeRunning becomes false)
            while runner.isClaudeRunning {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            if !runner.claudeOutput.isEmpty {
                claudeCache[currentMode] = runner.claudeOutput
            }
        }
    }

    private func sendInlineChat() {
        guard !inlineChatPrompt.isEmpty else { return }
        if !editorSelection.isEmpty {
            pinnedSelection = editorSelection
        }
        let code = pinnedSelection.isEmpty ? nil : pinnedSelection
        let dir = activeTab == .scratch
            ? scratchDir.path
            : (selectedFile.map { URL(fileURLWithPath: $0.path).deletingLastPathComponent().path })

        // Inline chat goes to Explain mode cache
        claudeMode = .explain

        runner.askClaude(
            prompt: inlineChatPrompt,
            fileLabel: activeFileLabel,
            workingDirectory: dir,
            code: code
        )
        activePanel = .claude
        showInlineChat = false
        inlineChatPrompt = ""

        Task { @MainActor in
            while runner.isClaudeRunning {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            if !runner.claudeOutput.isEmpty {
                claudeCache[.explain] = runner.claudeOutput
            }
        }
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
