import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers
import UserNotifications

@MainActor
final class QuizStore: ObservableObject {
    @Published private(set) var cards: [QuizCard] = []
    @Published var currentCard: QuizCard?
    @Published var selectedChoice: String?
    @Published var answerVisible = false
    @Published var searchText = ""
    @Published var selectedTag: String?
    @Published var reminderInterval: ReminderInterval = .off {
        didSet {
            UserDefaults.standard.set(reminderInterval.rawValue, forKey: "reminderInterval")
            configureReminder()
        }
    }

    @Published private(set) var correctCount = UserDefaults.standard.integer(forKey: "correctCount")
    @Published private(set) var attemptedCount = UserDefaults.standard.integer(forKey: "attemptedCount")
    @Published private(set) var lastError: String?
    @Published var exportMessage: String?

    init() {
        let savedInterval = UserDefaults.standard.integer(forKey: "reminderInterval")
        reminderInterval = ReminderInterval(rawValue: savedInterval) ?? .off
        loadDeck()
        pickRandom()
    }

    var filteredCards: [QuizCard] {
        cards.filter { card in
            let matchesTag = selectedTag == nil || card.tags.contains(selectedTag!)
            guard matchesTag else { return false }
            guard !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return true }
            let query = searchText.lowercased()
            return card.topic.lowercased().contains(query)
                || card.prompt.lowercased().contains(query)
                || card.answer.lowercased().contains(query)
                || card.tags.joined(separator: " ").lowercased().contains(query)
        }
    }

    var tags: [String] {
        Array(Set(cards.flatMap(\.tags))).sorted()
    }

    var accuracyText: String {
        guard attemptedCount > 0 else { return "0%" }
        return "\(Int((Double(correctCount) / Double(attemptedCount)) * 100))%"
    }

    func loadDeck() {
        do {
            let url = try locateQuizBank()
            let data = try Data(contentsOf: url)
            let deck = try JSONDecoder().decode(QuizDeck.self, from: data)
            cards = deck.cards
            lastError = nil
        } catch {
            cards = []
            lastError = error.localizedDescription
        }
    }

    func pickRandom() {
        let pool = filteredCards.isEmpty ? cards : filteredCards
        currentCard = pool.randomElement()
        selectedChoice = nil
        answerVisible = false
    }

    func select(_ choice: String) {
        guard selectedChoice == nil, let card = currentCard else { return }
        selectedChoice = choice
        answerVisible = true
        attemptedCount += 1
        if choice == card.answer {
            correctCount += 1
        }
        UserDefaults.standard.set(correctCount, forKey: "correctCount")
        UserDefaults.standard.set(attemptedCount, forKey: "attemptedCount")
    }

    func reveal() {
        answerVisible = true
    }

    func resetStats() {
        correctCount = 0
        attemptedCount = 0
        UserDefaults.standard.set(0, forKey: "correctCount")
        UserDefaults.standard.set(0, forKey: "attemptedCount")
    }

    func openSource() {
        guard let source = currentCard?.source else { return }
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let candidates = [
            root.appendingPathComponent(source),
            root.deletingLastPathComponent().appendingPathComponent(source)
        ]
        if let url = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            NSWorkspace.shared.open(url)
        }
    }

    func exportDeckForExcel() {
        let panel = NSSavePanel()
        panel.title = "Export Quiz Deck"
        panel.nameFieldStringValue = "ml-popup-quiz.csv"
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.canCreateDirectories = true

        guard panel.runModal() == .OK, let url = panel.url else { return }

        do {
            let csv = makeCSV(cards: cards)
            // UTF-8 BOM helps Excel detect Korean text correctly.
            let bom = Data([0xEF, 0xBB, 0xBF])
            var output = bom
            output.append(Data(csv.utf8))
            try output.write(to: url)
            exportMessage = "Exported \(cards.count) cards"
        } catch {
            exportMessage = error.localizedDescription
        }
    }

    func configureReminder() {
        UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: ["ml-popup-quiz-reminder"])
        guard reminderInterval != .off else { return }

        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { [reminderInterval] granted, _ in
            guard granted else { return }
            let content = UNMutableNotificationContent()
            content.title = "ML Popup Quiz"
            content.body = "메뉴바에서 Megatron/Core 개념 하나만 확인하세요."
            content.sound = .default

            let trigger = UNTimeIntervalNotificationTrigger(
                timeInterval: TimeInterval(reminderInterval.rawValue * 60),
                repeats: true
            )
            let request = UNNotificationRequest(
                identifier: "ml-popup-quiz-reminder",
                content: content,
                trigger: trigger
            )
            UNUserNotificationCenter.current().add(request)
        }
    }

    private func locateQuizBank() throws -> URL {
        if let bundled = Bundle.module.url(forResource: "quiz_bank", withExtension: "json") {
            return bundled
        }

        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let candidates = [
            cwd.appendingPathComponent("Sources/Resources/quiz_bank.json"),
            cwd.appendingPathComponent("ml-popup-quiz/Sources/Resources/quiz_bank.json")
        ]

        if let local = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            return local
        }

        throw CocoaError(.fileNoSuchFile)
    }

    private func makeCSV(cards: [QuizCard]) -> String {
        let header = [
            "id", "topic", "difficulty", "prompt", "choice_1", "choice_2", "choice_3",
            "answer", "details", "tags", "source"
        ]
        let rows = cards.map { card in
            [
                card.id,
                card.topic,
                card.difficulty.rawValue,
                card.prompt,
                card.choices[safe: 0] ?? "",
                card.choices[safe: 1] ?? "",
                card.choices[safe: 2] ?? "",
                card.answer,
                card.details,
                card.tags.joined(separator: " | "),
                card.source
            ]
        }
        return ([header] + rows)
            .map { $0.map(csvEscape).joined(separator: ",") }
            .joined(separator: "\n")
            + "\n"
    }

    private func csvEscape(_ value: String) -> String {
        let normalized = value.replacingOccurrences(of: "\r\n", with: "\n")
        let escaped = normalized.replacingOccurrences(of: "\"", with: "\"\"")
        return "\"\(escaped)\""
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
