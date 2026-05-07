import Foundation

struct QuizDeck: Codable {
    let version: Int
    let generatedFrom: [String]
    let cards: [QuizCard]

    enum CodingKeys: String, CodingKey {
        case version
        case generatedFrom = "generated_from"
        case cards
    }
}

struct QuizCard: Codable, Identifiable, Hashable {
    let id: String
    let topic: String
    let source: String
    let difficulty: Difficulty
    let prompt: String
    let choices: [String]
    let answer: String
    let details: String
    let tags: [String]
}

enum Difficulty: String, Codable, CaseIterable, Hashable {
    case warmup
    case core
    case deep

    var label: String {
        switch self {
        case .warmup: "Warmup"
        case .core: "Core"
        case .deep: "Deep"
        }
    }
}

enum ReminderInterval: Int, CaseIterable, Identifiable {
    case off = 0
    case fifteen = 15
    case thirty = 30
    case sixty = 60

    var id: Int { rawValue }

    var label: String {
        switch self {
        case .off: "Off"
        case .fifteen: "15m"
        case .thirty: "30m"
        case .sixty: "60m"
        }
    }
}

