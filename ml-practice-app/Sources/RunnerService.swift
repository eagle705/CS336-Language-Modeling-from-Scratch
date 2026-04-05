import Foundation
import SwiftUI

/// Runs Python code and Claude Code as subprocesses.
class RunnerService: ObservableObject {
    @Published var pythonOutput: String = ""
    @Published var claudeOutput: String = ""
    @Published var terminalOutput: String = ""
    @Published var isPythonRunning = false
    @Published var isClaudeRunning = false
    @Published var isTerminalRunning = false

    private var pythonProcess: Process?
    private var claudeProcess: Process?
    private var terminalProcess: Process?
    private var terminalInputPipe: Pipe?

    // MARK: - Python Execution (Local)

    func runPython(filePath: String, pythonPath: String = "/usr/bin/env python3", scriptArgs: String = "") {
        stopPython()
        isPythonRunning = true
        pythonOutput = "Running \(URL(fileURLWithPath: filePath).lastPathComponent)...\n"
        pythonOutput += "Python: \(pythonPath)\n"
        if !scriptArgs.isEmpty {
            pythonOutput += "Args: \(scriptArgs)\n"
        }
        pythonOutput += "\n"

        let process = Process()
        let pipe = Pipe()

        // Parse pythonPath: if it contains a space (e.g. "/usr/bin/env python3"), split into executable + args
        let parts = pythonPath.split(separator: " ", maxSplits: 1).map(String.init)
        let extraArgs = scriptArgs.isEmpty ? [] : scriptArgs.split(separator: " ").map(String.init)
        if parts.count == 2 {
            process.executableURL = URL(fileURLWithPath: parts[0])
            process.arguments = [parts[1], filePath] + extraArgs
        } else {
            process.executableURL = URL(fileURLWithPath: pythonPath)
            process.arguments = [filePath] + extraArgs
        }
        process.currentDirectoryURL = URL(fileURLWithPath: filePath).deletingLastPathComponent()
        process.standardOutput = pipe
        process.standardError = pipe

        setupOutputHandler(pipe: pipe, isForPython: true)
        setupTerminationHandler(process: process, pipe: pipe, isForPython: true)

        do {
            try process.run()
            pythonProcess = process
        } catch {
            pythonOutput += "Failed to run: \(error.localizedDescription)"
            isPythonRunning = false
        }
    }

    // MARK: - Python Execution (Remote SSH + Container)

    struct RemoteConfig {
        var host: String
        var port: String
        var user: String
        var keyPath: String
        var containerRuntime: String  // docker, podman
        var containerName: String
        var pythonPath: String
        var workDir: String
        var portForwardArgs: [String] = []
        var controlMaster: Bool = true
        var controlSocketPath: String = ""
    }

    /// Build common SSH args from config (shared between run/terminal/test).
    static func buildSSHArgs(config: RemoteConfig) -> [String] {
        var args: [String] = []
        if !config.keyPath.isEmpty { args += ["-i", config.keyPath] }
        args += ["-p", config.port]
        args += ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
        // ControlMaster for connection reuse
        if config.controlMaster && !config.controlSocketPath.isEmpty {
            args += ["-o", "ControlMaster=auto"]
            args += ["-o", "ControlPath=\(config.controlSocketPath)"]
            args += ["-o", "ControlPersist=600"]
        }
        // Port forwarding
        args += config.portForwardArgs
        // Target
        let target = config.user.isEmpty ? config.host : "\(config.user)@\(config.host)"
        args.append(target)
        return args
    }

    func runPythonRemote(filePath: String, config: RemoteConfig, scriptArgs: String = "") {
        stopPython()
        isPythonRunning = true

        let fileName = URL(fileURLWithPath: filePath).lastPathComponent
        pythonOutput = "Running \(fileName) on remote...\n"
        pythonOutput += "Host: \(config.user)@\(config.host):\(config.port)\n"
        if !config.containerName.isEmpty {
            pythonOutput += "Container: \(config.containerRuntime) → \(config.containerName)\n"
        }
        pythonOutput += "Python: \(config.pythonPath)\n"
        if !scriptArgs.isEmpty {
            pythonOutput += "Args: \(scriptArgs)\n"
        }
        pythonOutput += "\n"

        // Read the local file content to pipe via stdin
        guard let codeData = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else {
            pythonOutput += "Failed to read file: \(filePath)"
            isPythonRunning = false
            return
        }

        let process = Process()
        let outputPipe = Pipe()
        let inputPipe = Pipe()

        process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")

        var sshArgs = RunnerService.buildSSHArgs(config: config)

        // Build the remote command
        let argsStr = scriptArgs.isEmpty ? "" : " \(scriptArgs)"
        let remoteCmd: String
        if !config.containerName.isEmpty {
            remoteCmd = "\(config.containerRuntime) exec -i \(config.containerName) \(config.pythonPath) -u -\(argsStr)"
        } else {
            if !config.workDir.isEmpty {
                remoteCmd = "cd \(config.workDir) 2>/dev/null; \(config.pythonPath) -u -\(argsStr)"
            } else {
                remoteCmd = "\(config.pythonPath) -u -\(argsStr)"
            }
        }
        sshArgs.append(remoteCmd)

        if config.controlMaster {
            pythonOutput += "ControlMaster: enabled (connection reuse)\n"
        }
        if !config.portForwardArgs.isEmpty {
            pythonOutput += "Port forwards: \(config.portForwardArgs.enumerated().filter { $0.offset % 2 == 1 }.map { $0.element }.joined(separator: ", "))\n"
        }
        pythonOutput += "\n"

        process.arguments = sshArgs
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        process.standardInput = inputPipe

        setupOutputHandler(pipe: outputPipe, isForPython: true)
        setupTerminationHandler(process: process, pipe: outputPipe, isForPython: true)

        do {
            try process.run()
            pythonProcess = process

            // Send code via stdin then close
            inputPipe.fileHandleForWriting.write(codeData)
            inputPipe.fileHandleForWriting.closeFile()
        } catch {
            pythonOutput += "Failed to connect: \(error.localizedDescription)\n"
            pythonOutput += "Check SSH settings and ensure the server is reachable."
            isPythonRunning = false
        }
    }

    /// Test SSH connection.
    func testSSHConnection(config: RemoteConfig, completion: @escaping (Bool, String) -> Void) {
        let process = Process()
        let pipe = Pipe()

        process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
        var args = RunnerService.buildSSHArgs(config: config)

        // Test: echo + check container if configured
        if !config.containerName.isEmpty {
            args.append("\(config.containerRuntime) exec \(config.containerName) \(config.pythonPath) --version")
        } else {
            args.append("\(config.pythonPath) --version")
        }

        process.arguments = args
        process.standardOutput = pipe
        process.standardError = pipe

        process.terminationHandler = { proc in
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? ""
            DispatchQueue.main.async {
                completion(proc.terminationStatus == 0, output.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }

        do {
            try process.run()
        } catch {
            completion(false, error.localizedDescription)
        }
    }

    // MARK: - Text Helpers

    /// Remove ANSI escape sequences and other terminal control codes from output.
    static func stripANSI(_ str: String) -> String {
        var result = str
        // ESC[ ... m (SGR - colors/styles)
        // ESC[ ... (any letter) (cursor movement, erase, etc.)
        // ESC] ... BEL or ST (OSC - title, hyperlinks, etc.)
        // ESC( and ESC) (charset switching)
        // Single ESC codes
        let patterns = [
            #"\x1B\[[0-9;]*[a-zA-Z]"#,          // CSI sequences (colors, cursor, erase)
            #"\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)"#, // OSC sequences (title etc.)
            #"\x1B[\(\)][A-B0-2]"#,               // Charset switching
            #"\x1B[>=<]"#,                         // Keypad modes
            #"\x1B\[[\?]?[0-9;]*[hl]"#,          // DEC private modes
            #"\r"#,                                // Carriage returns (cause overwriting)
        ]
        for pattern in patterns {
            result = result.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        }
        return result
    }

    // MARK: - Shared Helpers

    private func setupOutputHandler(pipe: Pipe, isForPython: Bool) {
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async {
                if isForPython {
                    self?.pythonOutput += str
                } else {
                    self?.claudeOutput += str
                }
            }
        }
    }

    private func setupTerminationHandler(process: Process, pipe: Pipe, isForPython: Bool) {
        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                if isForPython {
                    self?.isPythonRunning = false
                    self?.pythonOutput += "\n--- Exited with code \(proc.terminationStatus) ---"
                } else {
                    self?.isClaudeRunning = false
                    if proc.terminationStatus != 0 {
                        self?.claudeOutput += "\n--- Claude exited with code \(proc.terminationStatus) ---"
                    }
                }
                pipe.fileHandleForReading.readabilityHandler = nil
            }
        }
    }

    func stopPython() {
        pythonProcess?.terminate()
        pythonProcess = nil
        isPythonRunning = false
    }

    // MARK: - Claude Code Feedback

    func askClaude(prompt: String, fileLabel: String, workingDirectory: String?, code: String?) {
        stopClaude()
        isClaudeRunning = true

        // Show the query that was sent
        var header = "**Q:** \(prompt)\n"
        if let code = code, !code.isEmpty {
            let lineCount = code.components(separatedBy: "\n").count
            header += "**Context:** \(fileLabel) (\(lineCount) lines)\n"
        }
        header += "\n---\n\n"
        claudeOutput = header

        let process = Process()
        let stdoutPipe = Pipe()

        // Build prompt: include code only if provided
        let fullPrompt: String
        if let code = code, !code.isEmpty {
            fullPrompt = """
            File: \(fileLabel)

            \(prompt)

            Code:
            ```python
            \(code)
            ```
            """
        } else {
            fullPrompt = prompt
        }

        // Try to find claude in PATH
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["claude", "-p", fullPrompt]
        if let dir = workingDirectory, !dir.isEmpty {
            process.currentDirectoryURL = URL(fileURLWithPath: dir)
        }
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

    // MARK: - Embedded Terminal (PTY-based)

    private var masterFD: Int32 = -1

    /// Start an interactive shell session using a pseudo-terminal (supports tab completion).
    func startTerminal(workingDirectory: String?, remoteConfig: RemoteConfig? = nil) {
        stopTerminal()
        isTerminalRunning = true
        terminalOutput = ""

        // Build command and args
        let execPath: String
        var args: [String]
        var env = ProcessInfo.processInfo.environment
        env["TERM"] = "xterm-256color"
        env["LANG"] = "en_US.UTF-8"

        if let config = remoteConfig {
            let target = config.user.isEmpty ? config.host : "\(config.user)@\(config.host)"
            terminalOutput = "Connecting to \(target)...\n"

            execPath = "/usr/bin/ssh"
            args = RunnerService.buildSSHArgs(config: config)

            if !config.containerName.isEmpty {
                args.append("\(config.containerRuntime) exec -it \(config.containerName) /bin/bash")
            }
        } else {
            execPath = "/bin/zsh"
            args = ["-l"]  // login shell for full env
            if let dir = workingDirectory, !dir.isEmpty {
                env["PWD"] = dir
            }
        }

        // Open pseudo-terminal
        var slaveFD: Int32 = -1
        var masterFD: Int32 = -1
        var winSize = winsize(ws_row: 40, ws_col: 120, ws_xpixel: 0, ws_ypixel: 0)

        guard openpty(&masterFD, &slaveFD, nil, nil, &winSize) == 0 else {
            terminalOutput += "Failed to open pseudo-terminal"
            isTerminalRunning = false
            return
        }

        self.masterFD = masterFD

        let process = Process()
        process.executableURL = URL(fileURLWithPath: execPath)
        process.arguments = args
        process.environment = env

        // Use slave end of pty for the process
        let slaveHandle = FileHandle(fileDescriptor: slaveFD, closeOnDealloc: false)
        process.standardInput = slaveHandle
        process.standardOutput = slaveHandle
        process.standardError = slaveHandle

        if remoteConfig == nil, let dir = workingDirectory, !dir.isEmpty {
            process.currentDirectoryURL = URL(fileURLWithPath: dir)
        }

        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                self?.isTerminalRunning = false
                self?.terminalOutput += "\n--- Session ended (code \(proc.terminationStatus)) ---"
            }
        }

        do {
            try process.run()
            terminalProcess = process
            // Close slave fd in parent process — child owns it now
            close(slaveFD)
        } catch {
            terminalOutput += "Failed to start: \(error.localizedDescription)"
            isTerminalRunning = false
            close(slaveFD)
            close(masterFD)
            self.masterFD = -1
            return
        }

        // Read from master fd in background
        let fd = masterFD
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            let bufferSize = 4096
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
            defer { buffer.deallocate() }

            while true {
                let bytesRead = read(fd, buffer, bufferSize)
                if bytesRead <= 0 { break }
                if let str = String(bytes: UnsafeBufferPointer(start: buffer, count: bytesRead), encoding: .utf8) {
                    let clean = RunnerService.stripANSI(str)
                    DispatchQueue.main.async {
                        self?.terminalOutput += clean
                    }
                }
            }
        }
    }

    /// Send text to the running terminal session (supports Tab, Ctrl+C, etc.).
    func sendToTerminal(_ command: String) {
        guard isTerminalRunning, masterFD >= 0 else { return }
        let cmd = command + "\n"
        cmd.withCString { ptr in
            _ = write(masterFD, ptr, strlen(ptr))
        }
    }

    /// Send a raw character (e.g. Tab = \t, Ctrl+C = \u{3}).
    func sendRawToTerminal(_ char: String) {
        guard isTerminalRunning, masterFD >= 0 else { return }
        char.withCString { ptr in
            _ = write(masterFD, ptr, strlen(ptr))
        }
    }

    func stopTerminal() {
        terminalProcess?.terminate()
        terminalProcess = nil
        if masterFD >= 0 {
            close(masterFD)
            masterFD = -1
        }
        terminalInputPipe = nil
        isTerminalRunning = false
    }
}
