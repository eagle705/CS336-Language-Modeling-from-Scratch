import SwiftUI
import Combine
import UserNotifications

// MARK: - Problem State

enum ProblemState: String, Codable, CaseIterable {
    case unsolved
    case inProgress
    case solved

    var icon: String {
        switch self {
        case .unsolved: return "circle"
        case .inProgress: return "circle.lefthalf.filled"
        case .solved: return "checkmark.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .unsolved: return .gray
        case .inProgress: return .orange
        case .solved: return .green
        }
    }

    var label: String {
        switch self {
        case .unsolved: return "Unsolved"
        case .inProgress: return "In Progress"
        case .solved: return "Solved"
        }
    }
}

// MARK: - Problem

struct ProblemFile: Identifiable, Hashable {
    let id: String  // filename
    let path: String
    let name: String
}

struct Problem: Identifiable, Hashable {
    let id: String          // directory name, e.g. "01-backpropagation"
    let title: String       // from docstring
    let category: Category
    let files: [ProblemFile]
    var state: ProblemState = .unsolved
    var lastPracticed: Date?
    var scheduledDate: Date?

    static func == (lhs: Problem, rhs: Problem) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

enum Category: String, Codable, CaseIterable {
    case fundamentals = "Fundamentals"
    case parallelism = "Parallelism"
    case trainingSystems = "Training Systems"
    case modelAndData = "Model & Data"

    static func from(directoryName: String) -> Category {
        guard let prefix = directoryName.split(separator: "-").first,
              let num = Int(prefix) else { return .fundamentals }
        switch num {
        case 1, 3, 4, 13, 14, 15: return .fundamentals
        case 2, 20, 22: return .parallelism
        case 5, 6, 7, 8, 11, 16, 17, 19, 21, 23: return .trainingSystems
        case 9, 10, 12, 18: return .modelAndData
        default: return .fundamentals
        }
    }
}

// MARK: - Persisted State

struct PersistedProblemState: Codable {
    var state: ProblemState
    var lastPracticed: Date?
    var scheduledDate: Date?
}

struct AppState: Codable {
    var problemsPerDay: Int = 3
    var notificationHour: Int = 9
    var problems: [String: PersistedProblemState] = [:]
}

// MARK: - Problem Store

class ProblemStore: ObservableObject {
    @Published var problems: [Problem] = []
    @Published var practiceRootPath: String?
    @Published var showDirectoryPicker = false
    @AppStorage("problemsPerDay") var problemsPerDay: Int = 3
    @AppStorage("notificationHour") var notificationHour: Int = 9
    @AppStorage("practiceRoot") var savedRootPath: String = ""

    private let stateFileURL: URL

    init() {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ml-practice")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.stateFileURL = dir.appendingPathComponent("state.json")

        if !savedRootPath.isEmpty {
            practiceRootPath = savedRootPath
            loadProblems()
        }
    }

    // MARK: - Load / Save

    func setPracticeRoot(_ path: String) {
        practiceRootPath = path
        savedRootPath = path
        loadProblems()
        scheduleNotifications()
    }

    func loadProblems() {
        guard let root = practiceRootPath else { return }

        // Load persisted state
        var appState = AppState()
        if let data = try? Data(contentsOf: stateFileURL),
           let decoded = try? JSONDecoder().decode(AppState.self, from: data) {
            appState = decoded
        }

        // Scan filesystem
        let scanned = ProblemLoader.scan(rootPath: root)

        // Merge with persisted state
        problems = scanned.map { problem in
            var p = problem
            if let persisted = appState.problems[problem.id] {
                p.state = persisted.state
                p.lastPracticed = persisted.lastPracticed
                p.scheduledDate = persisted.scheduledDate
            }
            return p
        }

        // Schedule today's problems if not yet
        scheduleTodayIfNeeded()
    }

    func saveState() {
        var appState = AppState()
        appState.problemsPerDay = problemsPerDay
        appState.notificationHour = notificationHour
        for problem in problems {
            appState.problems[problem.id] = PersistedProblemState(
                state: problem.state,
                lastPracticed: problem.lastPracticed,
                scheduledDate: problem.scheduledDate
            )
        }
        if let data = try? JSONEncoder().encode(appState) {
            try? data.write(to: stateFileURL)
        }
    }

    // MARK: - State Management

    func updateState(for problemId: String, to newState: ProblemState) {
        guard let idx = problems.firstIndex(where: { $0.id == problemId }) else { return }
        problems[idx].state = newState
        problems[idx].lastPracticed = Date()
        saveState()
    }

    // MARK: - Today's Problems

    var todayProblems: [Problem] {
        let today = Calendar.current.startOfDay(for: Date())
        return problems.filter { p in
            guard let scheduled = p.scheduledDate else { return false }
            return Calendar.current.isDate(scheduled, inSameDayAs: today)
        }
    }

    func scheduleTodayIfNeeded() {
        let today = Calendar.current.startOfDay(for: Date())
        let alreadyScheduled = problems.filter { p in
            guard let d = p.scheduledDate else { return false }
            return Calendar.current.isDate(d, inSameDayAs: today)
        }

        if alreadyScheduled.isEmpty {
            // Pick N problems: prioritize inProgress, then unsolved
            let inProgress = problems.filter { $0.state == .inProgress }
            let unsolved = problems.filter { $0.state == .unsolved }.shuffled()
            let candidates = inProgress + unsolved
            let count = min(problemsPerDay, candidates.count)

            for i in 0..<count {
                if let idx = problems.firstIndex(where: { $0.id == candidates[i].id }) {
                    problems[idx].scheduledDate = today
                }
            }
            saveState()
        }
    }

    func rescheduleToday() {
        // Clear today's schedule and re-pick
        let today = Calendar.current.startOfDay(for: Date())
        for i in 0..<problems.count {
            if let d = problems[i].scheduledDate, Calendar.current.isDate(d, inSameDayAs: today) {
                problems[i].scheduledDate = nil
            }
        }
        scheduleTodayIfNeeded()
    }

    // MARK: - Stats

    var solvedCount: Int { problems.filter { $0.state == .solved }.count }
    var inProgressCount: Int { problems.filter { $0.state == .inProgress }.count }
    var unsolvedCount: Int { problems.filter { $0.state == .unsolved }.count }

    func stats(for category: Category) -> (solved: Int, total: Int) {
        let catProblems = problems.filter { $0.category == category }
        return (catProblems.filter { $0.state == .solved }.count, catProblems.count)
    }

    // MARK: - Notifications

    func scheduleNotifications() {
        guard Bundle.main.bundleIdentifier != nil else { return }
        let center = UNUserNotificationCenter.current()
        center.removeAllPendingNotificationRequests()

        let content = UNMutableNotificationContent()
        content.title = "ML Practice Time!"
        content.body = "You have \(problemsPerDay) problems to practice today."
        content.sound = .default

        var dateComponents = DateComponents()
        dateComponents.hour = notificationHour
        dateComponents.minute = 0

        let trigger = UNCalendarNotificationTrigger(dateMatching: dateComponents, repeats: true)
        let request = UNNotificationRequest(identifier: "daily-practice", content: content, trigger: trigger)
        center.add(request)
    }

}
