import SwiftUI

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

    var body: some View {
        Form {
            Section("Daily Practice") {
                Stepper("Problems per day: \(store.problemsPerDay)",
                        value: $store.problemsPerDay, in: 1...10)
                Picker("Notification time", selection: $store.notificationHour) {
                    ForEach(6..<23) { hour in
                        Text("\(hour):00").tag(hour)
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
        .frame(width: 400, height: 250)
        .onChange(of: store.problemsPerDay) { _, _ in store.scheduleNotifications() }
        .onChange(of: store.notificationHour) { _, _ in store.scheduleNotifications() }
    }
}
