import SwiftUI

struct ProblemDetailView: View {
    let problem: Problem
    @EnvironmentObject var store: ProblemStore
    @State private var selectedFileId: String?
    @State private var codeContent: String = ""

    private var selectedFile: ProblemFile? {
        problem.files.first { $0.id == selectedFileId } ?? problem.files.first
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            header
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(.ultraThinMaterial)

            Divider()

            // File tabs (if multiple files)
            if problem.files.count > 1 {
                fileTabs
                Divider()
            }

            // Code view
            CodeView(code: codeContent)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onAppear { loadFile() }
        .onChange(of: selectedFileId) { _, _ in loadFile() }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(problem.title)
                    .font(.title2.bold())
                HStack(spacing: 12) {
                    Label(problem.category.rawValue, systemImage: "folder")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Label("\(problem.files.count) files", systemImage: "doc.text")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            // State picker
            Picker("State", selection: Binding(
                get: { problem.state },
                set: { store.updateState(for: problem.id, to: $0) }
            )) {
                ForEach(ProblemState.allCases, id: \.self) { state in
                    Label(state.label, systemImage: state.icon)
                        .tag(state)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 280)

            // Actions
            HStack(spacing: 8) {
                Button {
                    if let file = selectedFile {
                        store.openFileInEditor(file: file)
                    }
                } label: {
                    Label("Open", systemImage: "doc.text.magnifyingglass")
                }
                .help("Open in default editor")

                Button {
                    if let file = selectedFile {
                        store.openClaudeForFeedback(problem: problem, file: file)
                    }
                } label: {
                    Label("Claude Feedback", systemImage: "terminal")
                }
                .buttonStyle(.borderedProminent)
                .help("Open Claude Code in Terminal for feedback")
            }
        }
    }

    // MARK: - File Tabs

    private var fileTabs: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 0) {
                ForEach(problem.files) { file in
                    Button {
                        selectedFileId = file.id
                    } label: {
                        Text(file.name + ".py")
                            .font(.system(size: 12, design: .monospaced))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(
                                (selectedFileId ?? problem.files.first?.id) == file.id
                                    ? Color.accentColor.opacity(0.15)
                                    : Color.clear
                            )
                    }
                    .buttonStyle(.plain)

                    Divider()
                        .frame(height: 20)
                }
            }
        }
        .background(.ultraThinMaterial)
    }

    // MARK: - Load File

    private func loadFile() {
        guard let file = selectedFile else {
            codeContent = "// No file selected"
            return
        }
        codeContent = (try? String(contentsOfFile: file.path, encoding: .utf8)) ?? "// Failed to read file"
    }
}
