import SwiftUI
import UserNotifications

struct ContentView: View {
    @EnvironmentObject var store: ProblemStore
    @State private var selectedProblemId: String?
    @State private var showDashboard = false

    var body: some View {
        if store.showDirectoryPicker {
            DirectoryPickerView()
        } else {
            NavigationSplitView {
                sidebar
                    .navigationSplitViewColumnWidth(min: 260, ideal: 300)
            } detail: {
                if showDashboard {
                    DashboardView()
                } else if let id = selectedProblemId,
                          let problem = store.problems.first(where: { $0.id == id }) {
                    ProblemDetailView(problem: problem)
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 48))
                            .foregroundStyle(.secondary)
                        Text("Select a problem to start")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(selection: $selectedProblemId) {
            // Dashboard button
            Button {
                showDashboard = true
                selectedProblemId = nil
            } label: {
                Label("Dashboard", systemImage: "chart.bar.fill")
            }
            .buttonStyle(.plain)
            .padding(.vertical, 4)

            // Today's problems
            let today = store.todayProblems
            if !today.isEmpty {
                Section("Today's Practice (\(today.count))") {
                    ForEach(today) { problem in
                        problemRow(problem)
                            .tag(problem.id)
                    }
                }
            }

            // All problems by category
            ForEach(Category.allCases, id: \.self) { category in
                let catProblems = store.problems.filter { $0.category == category }
                if !catProblems.isEmpty {
                    Section(category.rawValue) {
                        ForEach(catProblems) { problem in
                            problemRow(problem)
                                .tag(problem.id)
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .onChange(of: selectedProblemId) { _, newValue in
            if newValue != nil { showDashboard = false }
        }
        .toolbar {
            ToolbarItem {
                Button {
                    store.rescheduleToday()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Reschedule today's problems")
            }
        }
    }

    private func problemRow(_ problem: Problem) -> some View {
        HStack(spacing: 8) {
            Image(systemName: problem.state.icon)
                .foregroundStyle(problem.state.color)
                .font(.system(size: 14))

            VStack(alignment: .leading, spacing: 2) {
                Text(problem.title)
                    .font(.system(size: 13))
                    .lineLimit(1)
                Text("\(problem.files.count) files")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Directory Picker

struct DirectoryPickerView: View {
    @EnvironmentObject var store: ProblemStore

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "folder.badge.gearshape")
                .font(.system(size: 64))
                .foregroundStyle(.blue)

            Text("ML Practice")
                .font(.largeTitle.bold())

            Text("Select the implementation-practice directory to get started.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Button("Choose Directory...") {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = true
                panel.canChooseFiles = false
                panel.allowsMultipleSelection = false
                panel.message = "Select the implementation-practice directory"

                if panel.runModal() == .OK, let url = panel.url {
                    store.setPracticeRoot(url.path)
                    store.showDirectoryPicker = false
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}

// MARK: - Settings

struct SettingsView: View {
    @EnvironmentObject var store: ProblemStore
    @State private var pyenvVenvs: [PyenvVenv] = []
    @State private var customPythonPath: String = ""
    @State private var useCustomPath = false
    @State private var notificationStatus: String = "Checking..."
    @State private var testSent = false
    @State private var sshTestResult: String = ""
    @State private var sshTesting = false
    @State private var sshHosts: [SSHHost] = []

    var body: some View {
        Form {
            Section("Daily Practice") {
                Stepper("Problems per day: \(store.problemsPerDay)",
                        value: $store.problemsPerDay, in: 1...10)

                Picker("Hour", selection: $store.notificationHour) {
                    ForEach(0..<24, id: \.self) { hour in
                        Text(String(format: "%02d", hour)).tag(hour)
                    }
                }

                Picker("Minute", selection: $store.notificationMinute) {
                    ForEach(Array(stride(from: 0, to: 60, by: 5)), id: \.self) { min in
                        Text(String(format: "%02d", min)).tag(min)
                    }
                }

                HStack {
                    Text("Scheduled")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%02d:%02d", store.notificationHour, store.notificationMinute))
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.primary)
                }
            }

            Section("Notification") {
                HStack {
                    Text("Permission")
                    Spacer()
                    Text(notificationStatus)
                        .foregroundStyle(notificationStatus == "Authorized" ? .green : .orange)
                }

                Button("Request Permission") {
                    UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, _ in
                        DispatchQueue.main.async {
                            checkNotificationStatus()
                            if granted { store.scheduleNotifications() }
                        }
                    }
                }

                HStack {
                    Button("Send Test Notification (3s)") {
                        store.sendTestNotification()
                        testSent = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 4) { testSent = false }
                    }
                    .disabled(testSent)

                    if testSent {
                        Text("Sent!")
                            .font(.caption)
                            .foregroundStyle(.green)
                    }
                }

                if notificationStatus == "Denied (enable in System Settings)" {
                    Button("Open System Settings") {
                        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.Notifications-Settings")!)
                    }
                }
            }

            Section("Python Environment") {
                // Current selection display
                HStack {
                    Image(systemName: "terminal")
                        .foregroundStyle(.blue)
                    Text(store.pythonPath)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }

                // System default option
                Picker("Virtual Environment", selection: $store.pythonPath) {
                    Text("System Default (python3)").tag("/usr/bin/env python3")

                    if !pyenvVenvs.isEmpty {
                        Divider()
                        ForEach(pyenvVenvs, id: \.path) { venv in
                            Text("\(venv.name)  (\(venv.version))")
                                .tag(venv.path)
                        }
                    }
                }

                // Custom path input
                Toggle("Use custom Python path", isOn: $useCustomPath)

                if useCustomPath {
                    HStack {
                        TextField("e.g. /usr/local/bin/python3", text: $customPythonPath)
                            .font(.system(size: 12, design: .monospaced))
                            .textFieldStyle(.roundedBorder)

                        Button("Browse...") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = true
                            panel.canChooseDirectories = false
                            panel.allowsMultipleSelection = false
                            panel.message = "Select a Python executable"
                            if panel.runModal() == .OK, let url = panel.url {
                                customPythonPath = url.path
                                store.pythonPath = url.path
                            }
                        }

                        Button("Set") {
                            if !customPythonPath.isEmpty {
                                store.pythonPath = customPythonPath
                            }
                        }
                        .disabled(customPythonPath.isEmpty)
                    }
                }

                Button("Refresh Environments") {
                    pyenvVenvs = PyenvVenv.detect()
                }
                .controlSize(.small)
            }

            Section("Execution Mode") {
                Picker("Mode", selection: $store.execModeRaw) {
                    Text("Local").tag("local")
                    Text("Remote (SSH)").tag("remote")
                }
                .pickerStyle(.segmented)

                if store.isRemoteExecution {
                    // SSH Config import
                    if !sshHosts.isEmpty {
                        Picker("From ~/.ssh/config", selection: Binding(
                            get: { store.sshHost },
                            set: { _ in }
                        )) {
                            Text("Manual").tag("")
                            Divider()
                            ForEach(sshHosts) { host in
                                Text("\(host.name)\(host.hostname != host.name ? " (\(host.hostname))" : "")")
                                    .tag(host.hostname)
                            }
                        }
                        .onChange(of: store.sshHost) { _, _ in }  // handled by picker
                    }

                    HStack {
                        Button("Load SSH Config") {
                            sshHosts = SSHHost.parseConfig()
                        }
                        .controlSize(.small)

                        if !sshHosts.isEmpty {
                            Menu("Apply Host") {
                                ForEach(sshHosts) { host in
                                    Button("\(host.name) → \(host.hostname)") {
                                        store.sshHost = host.hostname
                                        if !host.user.isEmpty { store.sshUser = host.user }
                                        store.sshPort = host.port
                                        if !host.keyPath.isEmpty { store.sshKeyPath = host.keyPath }
                                    }
                                }
                            }
                            .controlSize(.small)

                            Text("\(sshHosts.count) hosts found")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Divider()

                    Group {
                        HStack {
                            Text("SSH Host")
                                .frame(width: 100, alignment: .leading)
                            TextField("gpu-server.example.com", text: $store.sshHost)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                        }

                        HStack {
                            Text("SSH User")
                                .frame(width: 100, alignment: .leading)
                            TextField("username", text: $store.sshUser)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                        }

                        HStack {
                            Text("SSH Port")
                                .frame(width: 100, alignment: .leading)
                            TextField("22", text: $store.sshPort)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                                .frame(width: 80)
                            Spacer()
                        }

                        HStack {
                            Text("SSH Key")
                                .frame(width: 100, alignment: .leading)
                            TextField("~/.ssh/id_rsa (optional)", text: $store.sshKeyPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                            Button("Browse") {
                                let panel = NSOpenPanel()
                                panel.canChooseFiles = true
                                panel.canChooseDirectories = false
                                panel.directoryURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".ssh")
                                if panel.runModal() == .OK, let url = panel.url {
                                    store.sshKeyPath = url.path
                                }
                            }
                            .controlSize(.small)
                        }
                    }

                    Divider()

                    Group {
                        Picker("Container Runtime", selection: $store.containerRuntime) {
                            Text("Docker").tag("docker")
                            Text("Podman").tag("podman")
                            Text("None (direct SSH)").tag("none")
                        }

                        if store.containerRuntime != "none" {
                            HStack {
                                Text("Container")
                                    .frame(width: 100, alignment: .leading)
                                TextField("container name or ID", text: $store.containerName)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.system(size: 12, design: .monospaced))
                            }
                        }

                        HStack {
                            Text("Python Path")
                                .frame(width: 100, alignment: .leading)
                            TextField("python3", text: $store.remotePythonPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                        }

                        if store.containerRuntime == "none" {
                            HStack {
                                Text("Work Dir")
                                    .frame(width: 100, alignment: .leading)
                                TextField("/workspace", text: $store.remoteWorkDir)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.system(size: 12, design: .monospaced))
                            }
                        }
                    }

                    Divider()

                    Group {
                        Toggle("SSH ControlMaster (connection reuse)", isOn: $store.sshControlMaster)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Port Forwarding (-L)")
                                .font(.system(size: 12))
                            TextField("8890:localhost:8890, 6005:localhost:6005", text: $store.sshPortForwards)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 11, design: .monospaced))
                            Text("Comma-separated. Format: localPort:host:remotePort")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                        }
                    }

                    Divider()

                    HStack {
                        Button("Test Connection") {
                            sshTesting = true
                            sshTestResult = ""
                            let config = RunnerService.RemoteConfig(
                                host: store.sshHost,
                                port: store.sshPort,
                                user: store.sshUser,
                                keyPath: store.sshKeyPath,
                                containerRuntime: store.containerRuntime == "none" ? "" : store.containerRuntime,
                                containerName: store.containerRuntime == "none" ? "" : store.containerName,
                                pythonPath: store.remotePythonPath,
                                workDir: store.remoteWorkDir,
                                portForwardArgs: store.portForwardArgs,
                                controlMaster: store.sshControlMaster,
                                controlSocketPath: store.controlSocketPath
                            )
                            RunnerService().testSSHConnection(config: config) { success, output in
                                sshTesting = false
                                sshTestResult = success ? "Connected: \(output)" : "Failed: \(output)"
                            }
                        }
                        .disabled(store.sshHost.isEmpty || sshTesting)

                        if sshTesting {
                            ProgressView()
                                .scaleEffect(0.5)
                                .frame(width: 16, height: 16)
                        }
                    }

                    if !sshTestResult.isEmpty {
                        Text(sshTestResult)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(sshTestResult.hasPrefix("Connected") ? .green : .red)
                            .lineLimit(3)
                    }
                }
            }

            Section("Practice Directory") {
                Text(store.practiceRootPath ?? "Not set")
                    .foregroundStyle(.secondary)
                Button("Change Directory...") {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true
                    panel.canChooseFiles = false
                    if panel.runModal() == .OK, let url = panel.url {
                        store.setPracticeRoot(url.path)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 680)
        .onAppear {
            pyenvVenvs = PyenvVenv.detect()
            sshHosts = SSHHost.parseConfig()
            customPythonPath = store.pythonPath
            useCustomPath = !store.pythonPath.starts(with: "/usr/bin/env")
                && !pyenvVenvs.contains(where: { $0.path == store.pythonPath })
            checkNotificationStatus()
        }
        .onChange(of: store.problemsPerDay) { _, _ in store.scheduleNotifications() }
        .onChange(of: store.notificationHour) { _, _ in store.scheduleNotifications() }
        .onChange(of: store.notificationMinute) { _, _ in store.scheduleNotifications() }
    }

    private func checkNotificationStatus() {
        UNUserNotificationCenter.current().getNotificationSettings { settings in
            DispatchQueue.main.async {
                switch settings.authorizationStatus {
                case .authorized: notificationStatus = "Authorized"
                case .denied: notificationStatus = "Denied (enable in System Settings)"
                case .notDetermined: notificationStatus = "Not requested"
                case .provisional: notificationStatus = "Provisional"
                case .ephemeral: notificationStatus = "Ephemeral"
                @unknown default: notificationStatus = "Unknown"
                }
            }
        }
    }
}

// MARK: - Pyenv Virtual Environment Detection

struct PyenvVenv: Identifiable {
    let id = UUID()
    let name: String
    let version: String
    let path: String  // full path to python binary

    /// Detects pyenv virtualenvs by scanning ~/.pyenv/versions/
    static func detect() -> [PyenvVenv] {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let versionsDir = "\(home)/.pyenv/versions"
        let fm = FileManager.default

        guard let entries = try? fm.contentsOfDirectory(atPath: versionsDir) else {
            return []
        }

        var venvs: [PyenvVenv] = []

        for entry in entries.sorted() {
            let pythonBin = "\(versionsDir)/\(entry)/bin/python"
            guard fm.fileExists(atPath: pythonBin) else { continue }

            // Check if it's a virtualenv (has pyvenv.cfg) or a base version
            let pyvenvCfg = "\(versionsDir)/\(entry)/pyvenv.cfg"
            let isVenv = fm.fileExists(atPath: pyvenvCfg)

            // Skip base Python versions (like "3.11.8") — only show named envs
            // A base version's name is purely numeric+dots
            let isBaseVersion = entry.allSatisfy { $0.isNumber || $0 == "." }

            if isVenv || !isBaseVersion {
                // Try to determine the base Python version
                var version = entry
                if isVenv, let cfg = try? String(contentsOfFile: pyvenvCfg, encoding: .utf8) {
                    for line in cfg.split(separator: "\n") {
                        if line.starts(with: "version") {
                            version = line.split(separator: "=").last?
                                .trimmingCharacters(in: .whitespaces) ?? entry
                            break
                        }
                    }
                }

                venvs.append(PyenvVenv(
                    name: entry,
                    version: isVenv ? "Python \(version)" : "Python \(entry)",
                    path: pythonBin
                ))
            }
        }

        return venvs
    }
}

// MARK: - SSH Config Parser

struct SSHHost: Identifiable {
    let id = UUID()
    let name: String       // Host alias
    let hostname: String   // HostName (actual address)
    let user: String       // User
    let port: String       // Port
    let keyPath: String    // IdentityFile

    /// Parse ~/.ssh/config and extract Host entries.
    static func parseConfig() -> [SSHHost] {
        let configPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ssh/config").path

        guard let content = try? String(contentsOfFile: configPath, encoding: .utf8) else {
            return []
        }

        var hosts: [SSHHost] = []
        var currentName = ""
        var hostname = ""
        var user = ""
        var port = "22"
        var keyPath = ""

        func flushHost() {
            if !currentName.isEmpty && currentName != "*" {
                hosts.append(SSHHost(
                    name: currentName,
                    hostname: hostname.isEmpty ? currentName : hostname,
                    user: user,
                    port: port,
                    keyPath: keyPath.replacingOccurrences(of: "~", with: FileManager.default.homeDirectoryForCurrentUser.path)
                ))
            }
            currentName = ""
            hostname = ""
            user = ""
            port = "22"
            keyPath = ""
        }

        for line in content.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }

            let parts = trimmed.split(separator: " ", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
            guard parts.count == 2 else { continue }

            let key = parts[0].lowercased()
            let value = parts[1]

            switch key {
            case "host":
                flushHost()
                currentName = value
            case "hostname":
                hostname = value
            case "user":
                user = value
            case "port":
                port = value
            case "identityfile":
                keyPath = value
            default:
                break
            }
        }
        flushHost()

        return hosts
    }
}
