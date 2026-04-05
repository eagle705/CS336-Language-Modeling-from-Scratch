import Foundation
import SwiftUI

/// Runs Python code and Claude Code as subprocesses.
class RunnerService: ObservableObject {
    @Published var pythonOutput: String = ""
    @Published var claudeOutput: String = ""
    @Published var isPythonRunning = false
    @Published var isClaudeRunning = false

    private var pythonProcess: Process?
    private var claudeProcess: Process?

    // MARK: - Python Execution

    func runPython(filePath: String) {
        stopPython()
        isPythonRunning = true
        pythonOutput = "Running \(URL(fileURLWithPath: filePath).lastPathComponent)...\n\n"

        let process = Process()
        let pipe = Pipe()

        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", filePath]
        process.currentDirectoryURL = URL(fileURLWithPath: filePath).deletingLastPathComponent()
        process.standardOutput = pipe
        process.standardError = pipe

        // Read output in real-time
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async {
                self?.pythonOutput += str
            }
        }

        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                self?.isPythonRunning = false
                self?.pythonOutput += "\n--- Exited with code \(proc.terminationStatus) ---"
                pipe.fileHandleForReading.readabilityHandler = nil
            }
        }

        do {
            try process.run()
            pythonProcess = process
        } catch {
            pythonOutput += "Failed to run: \(error.localizedDescription)"
            isPythonRunning = false
        }
    }

    func stopPython() {
        pythonProcess?.terminate()
        pythonProcess = nil
        isPythonRunning = false
    }

    // MARK: - Claude Code Feedback

    func askClaude(prompt: String, filePath: String, code: String) {
        stopClaude()
        isClaudeRunning = true
        claudeOutput = "Asking Claude...\n\n"

        let process = Process()
        let stdoutPipe = Pipe()

        // claude -p "prompt" for non-interactive (print mode)
        // We pass the code via stdin context
        let fullPrompt = """
        File: \(URL(fileURLWithPath: filePath).lastPathComponent)

        \(prompt)

        Code:
        ```python
        \(code)
        ```
        """

        // Try to find claude in PATH
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["claude", "-p", fullPrompt]
        process.currentDirectoryURL = URL(fileURLWithPath: filePath).deletingLastPathComponent()
        process.standardOutput = stdoutPipe
        process.standardError = stdoutPipe

        // Read output in real-time
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async {
                self?.claudeOutput += str
            }
        }

        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                self?.isClaudeRunning = false
                if proc.terminationStatus != 0 {
                    self?.claudeOutput += "\n--- Claude exited with code \(proc.terminationStatus) ---"
                    self?.claudeOutput += "\n\nMake sure 'claude' CLI is installed and in your PATH."
                    self?.claudeOutput += "\nInstall: npm install -g @anthropic-ai/claude-code"
                }
                stdoutPipe.fileHandleForReading.readabilityHandler = nil
            }
        }

        do {
            try process.run()
            claudeProcess = process
        } catch {
            claudeOutput += "Failed to run claude: \(error.localizedDescription)"
            claudeOutput += "\n\nMake sure 'claude' CLI is installed:"
            claudeOutput += "\n  npm install -g @anthropic-ai/claude-code"
            isClaudeRunning = false
        }
    }

    func stopClaude() {
        claudeProcess?.terminate()
        claudeProcess = nil
        isClaudeRunning = false
    }

    // MARK: - Open in Terminal (interactive Claude)

    func openClaudeInTerminal(filePath: String) {
        let dir = (filePath as NSString).deletingLastPathComponent
            .replacingOccurrences(of: "'", with: "'\\''")
        let appleScript = """
        tell application "Terminal"
            activate
            do script "cd '\(dir)' && claude"
        end tell
        """
        if let script = NSAppleScript(source: appleScript) {
            var error: NSDictionary?
            script.executeAndReturnError(&error)
        }
    }
}
