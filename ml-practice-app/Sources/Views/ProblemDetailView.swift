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

    enum Panel: String, CaseIterable {
        case claude = "Claude"
        case output = "Output"
    }

    enum EditorTab: String, CaseIterable {
        case reference = "Reference"
        case scratch = "Scratch Pad"
    }

    private var selectedFile: ProblemFile? {
        problem.files.first { $0.id == selectedFileId } ?? problem.files.first
    }

    /// The code currently active (reference file or scratch pad).
    private var activeCode: String {
        activeTab == .scratch ? scratchContent : codeContent
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
        }
        .onChange(of: selectedFileId) { _, _ in loadFile() }
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
                        runner.runPython(filePath: path)
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

            // Editor
            if activeTab == .reference {
                CodeEditorView(code: $codeContent)
                    .onChange(of: codeContent) { _, _ in hasUnsavedChanges = true }
            } else {
                CodeEditorView(code: $scratchContent)
                    .onChange(of: scratchContent) { _, _ in hasUnsavedChanges = true }
            }
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
                    runner.askClaude(prompt: claudePrompt, filePath: selectedFile?.path ?? "", code: activeCode)
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

            // Claude output — bright text on dark background
            ScrollViewReader { proxy in
                ScrollView {
                    Text(runner.claudeOutput.isEmpty
                         ? "Claude responses will appear here.\n\nType a question and press Cmd+Enter."
                         : runner.claudeOutput)
                        .font(.system(size: 13))
                        .foregroundColor(runner.claudeOutput.isEmpty
                                         ? .gray
                                         : Color(nsColor: NSColor(red: 0.92, green: 0.92, blue: 0.92, alpha: 1.0)))
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
