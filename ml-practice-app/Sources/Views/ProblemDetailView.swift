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

    enum Panel: String, CaseIterable {
        case claude = "Claude"
        case output = "Output"
    }

    private var selectedFile: ProblemFile? {
        problem.files.first { $0.id == selectedFileId } ?? problem.files.first
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            // Main content: Editor (left) + Panel (right)
            HSplitView {
                // Left: Code editor
                editorPane
                    .frame(minWidth: 400)

                // Right: Claude / Output panels
                rightPanel
                    .frame(minWidth: 300, idealWidth: 400)
            }
        }
        .onAppear { loadFile() }
        .onChange(of: selectedFileId) { _, _ in loadFile() }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 12) {
            // Problem title + state
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

            // State picker
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

            // Action buttons
            HStack(spacing: 6) {
                // Save
                Button {
                    saveFile()
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
                .keyboardShortcut("s", modifiers: .command)
                .disabled(!hasUnsavedChanges)
                .help("Save (Cmd+S)")

                // Run
                Button {
                    saveFile()
                    if let file = selectedFile {
                        runner.runPython(filePath: file.path)
                        activePanel = .output
                    }
                } label: {
                    Label("Run", systemImage: runner.isPythonRunning ? "stop.fill" : "play.fill")
                }
                .keyboardShortcut("r", modifiers: .command)
                .help("Run Python (Cmd+R)")

                // Open interactive Claude in Terminal
                Button {
                    if let file = selectedFile {
                        runner.openClaudeInTerminal(filePath: file.path)
                    }
                } label: {
                    Image(systemName: "terminal")
                }
                .help("Open Claude Code in Terminal")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }

    // MARK: - Editor Pane (left)

    private var editorPane: some View {
        VStack(spacing: 0) {
            // File tabs
            if problem.files.count > 1 {
                fileTabs
                Divider()
            }

            // Editable code editor
            CodeEditorView(code: $codeContent)
                .onChange(of: codeContent) { _, _ in
                    hasUnsavedChanges = true
                }
        }
    }

    private var fileTabs: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 0) {
                ForEach(problem.files) { file in
                    Button {
                        if hasUnsavedChanges { saveFile() }
                        selectedFileId = file.id
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "doc.text")
                                .font(.caption2)
                            Text(file.name + ".py")
                                .font(.system(size: 12, design: .monospaced))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(
                            (selectedFileId ?? problem.files.first?.id) == file.id
                                ? Color.accentColor.opacity(0.15)
                                : Color.clear
                        )
                    }
                    .buttonStyle(.plain)

                    Divider().frame(height: 20)
                }
            }
        }
        .background(.bar)
    }

    // MARK: - Right Panel (Claude + Output)

    private var rightPanel: some View {
        VStack(spacing: 0) {
            // Panel tabs
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
                                ProgressView()
                                    .scaleEffect(0.5)
                                    .frame(width: 12, height: 12)
                            }
                            if panel == .output && runner.isPythonRunning {
                                ProgressView()
                                    .scaleEffect(0.5)
                                    .frame(width: 12, height: 12)
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

            // Panel content
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
            // Prompt input
            HStack(spacing: 8) {
                TextField("Ask Claude about this code...", text: $claudePrompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .lineLimit(1...4)
                    .padding(8)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .cornerRadius(8)

                Button {
                    if let file = selectedFile {
                        runner.askClaude(prompt: claudePrompt, filePath: file.path, code: codeContent)
                    }
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

            // Claude output
            ScrollView {
                Text(runner.claudeOutput.isEmpty ? "Claude responses will appear here.\n\nType a question and press Cmd+Enter." : runner.claudeOutput)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(runner.claudeOutput.isEmpty ? .secondary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .background(Color(nsColor: SyntaxHighlighter.backgroundColor))
        }
    }

    // MARK: - Output Panel

    private var outputPanel: some View {
        VStack(spacing: 0) {
            // Toolbar
            HStack {
                Text("Python Output")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if runner.isPythonRunning {
                    Button("Stop") {
                        runner.stopPython()
                    }
                    .controlSize(.small)
                }
                Button("Clear") {
                    runner.pythonOutput = ""
                }
                .controlSize(.small)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            // Output text
            ScrollView {
                Text(runner.pythonOutput.isEmpty ? "Run the file to see output here." : runner.pythonOutput)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(runner.pythonOutput.isEmpty ? .secondary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .background(Color(nsColor: SyntaxHighlighter.backgroundColor))
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

    private func saveFile() {
        guard let file = selectedFile else { return }
        try? codeContent.write(toFile: file.path, atomically: true, encoding: .utf8)
        hasUnsavedChanges = false
    }
}
