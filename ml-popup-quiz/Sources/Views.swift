import AppKit
import SwiftUI

struct MenuBarQuizView: View {
    @EnvironmentObject private var store: QuizStore
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(spacing: 0) {
            HeaderView {
                openWindow(id: "main")
                NSApp.activate(ignoringOtherApps: true)
            }

            Divider()

            if let error = store.lastError {
                ErrorView(message: error)
            } else {
                ScrollView {
                    QuizCardView()
                        .padding(14)
                }
            }

            Divider()

            FooterView()
        }
    }
}

struct MainQuizWindowView: View {
    @EnvironmentObject private var store: QuizStore

    var body: some View {
        VStack(spacing: 0) {
            HeaderView(openSticky: nil)
            Divider()
            ScrollView {
                QuizCardView()
                    .padding(20)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            FooterView()
        }
    }
}

struct HeaderView: View {
    @EnvironmentObject private var store: QuizStore
    var openSticky: (() -> Void)?

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "brain.head.profile")
                .font(.title3)
                .foregroundStyle(.blue)
            VStack(alignment: .leading, spacing: 2) {
                Text("ML Popup Quiz")
                    .font(.headline)
                Text("\(store.cards.count) bilingual cards · \(store.accuracyText)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if let openSticky {
                Button(action: openSticky) {
                    Image(systemName: "macwindow.on.rectangle")
                }
                .buttonStyle(.borderless)
                .help("Open sticky window")
            }
            Button {
                store.pickRandom()
            } label: {
                Image(systemName: "shuffle")
            }
            .buttonStyle(.borderless)
            .help("Next random card")
        }
        .padding(14)
        .background(.regularMaterial)
    }
}

struct QuizCardView: View {
    @EnvironmentObject private var store: QuizStore

    var body: some View {
        if let card = store.currentCard {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Label(card.topic, systemImage: "tag")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(card.difficulty.label)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(difficultyColor(card.difficulty).opacity(0.15))
                        .foregroundStyle(difficultyColor(card.difficulty))
                        .clipShape(Capsule())
                }

                Text(card.prompt)
                    .font(.title3.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)

                VStack(spacing: 8) {
                    ForEach(card.choices, id: \.self) { choice in
                        ChoiceButton(
                            choice: choice,
                            isSelected: store.selectedChoice == choice,
                            isCorrect: choice == card.answer,
                            showResult: store.answerVisible
                        ) {
                            store.select(choice)
                        }
                    }
                }

                if store.answerVisible {
                    AnswerView(card: card)
                } else {
                    Button {
                        store.reveal()
                    } label: {
                        Label("Reveal", systemImage: "eye")
                    }
                    .buttonStyle(.bordered)
                }

                Spacer(minLength: 0)

                HStack {
                    Button {
                        store.openSource()
                    } label: {
                        Label("Source", systemImage: "doc.text.magnifyingglass")
                    }
                    .buttonStyle(.borderless)

                    Spacer()

                    Button {
                        store.pickRandom()
                    } label: {
                        Label("Next", systemImage: "arrow.right.circle")
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        } else {
            ContentUnavailableView("No Cards", systemImage: "questionmark.square.dashed")
        }
    }

    private func difficultyColor(_ difficulty: Difficulty) -> Color {
        switch difficulty {
        case .warmup: .green
        case .core: .blue
        case .deep: .orange
        }
    }
}

struct ChoiceButton: View {
    let choice: String
    let isSelected: Bool
    let isCorrect: Bool
    let showResult: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: iconName)
                    .foregroundStyle(iconColor)
                    .frame(width: 18)
                Text(choice)
                    .font(choice.contains("\n") ? .system(.callout, design: .monospaced) : .body)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                Spacer()
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(background)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .disabled(showResult)
    }

    private var iconName: String {
        guard showResult else { return "circle" }
        if isCorrect { return "checkmark.circle.fill" }
        if isSelected { return "xmark.circle.fill" }
        return "circle"
    }

    private var iconColor: Color {
        guard showResult else { return .secondary }
        if isCorrect { return .green }
        if isSelected { return .red }
        return .secondary
    }

    private var background: Color {
        guard showResult else { return Color(nsColor: .controlBackgroundColor) }
        if isCorrect { return .green.opacity(0.12) }
        if isSelected { return .red.opacity(0.10) }
        return Color(nsColor: .controlBackgroundColor)
    }
}

struct AnswerView: View {
    let card: QuizCard

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(card.answer, systemImage: "checkmark.seal")
                .font(card.answer.contains("\n") ? .system(.callout, design: .monospaced) : .subheadline.weight(.semibold))
                .foregroundStyle(.green)
            Text(card.details)
                .font(.callout)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            Text(card.source)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        .padding(12)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct FooterView: View {
    @EnvironmentObject private var store: QuizStore

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Picker("Tag", selection: $store.selectedTag) {
                    Text("All").tag(nil as String?)
                    ForEach(store.tags, id: \.self) { tag in
                        Text(tag).tag(Optional(tag))
                    }
                }
                .labelsHidden()

                Picker("Reminder", selection: $store.reminderInterval) {
                    ForEach(ReminderInterval.allCases) { interval in
                        Text(interval.label).tag(interval)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
            }

            HStack {
                TextField("Search", text: $store.searchText)
                    .textFieldStyle(.roundedBorder)
                Button {
                    store.exportDeckForExcel()
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
                .help("Export Excel CSV")
                Button {
                    store.pickRandom()
                } label: {
                    Image(systemName: "line.3.horizontal.decrease.circle")
                }
                .help("Apply filter")
            }
            if let exportMessage = store.exportMessage {
                Text(exportMessage)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(12)
        .background(.regularMaterial)
    }
}

struct SettingsView: View {
    @EnvironmentObject private var store: QuizStore

    var body: some View {
        Form {
            Section("Deck") {
                LabeledContent("Cards", value: "\(store.cards.count)")
                LabeledContent("Accuracy", value: "\(store.correctCount)/\(store.attemptedCount) · \(store.accuracyText)")
                Button {
                    store.exportDeckForExcel()
                } label: {
                    Label("Export Excel CSV", systemImage: "square.and.arrow.down")
                }
                Button("Reset Stats") {
                    store.resetStats()
                }
                if let exportMessage = store.exportMessage {
                    Text(exportMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Reminders") {
                Picker("Interval", selection: $store.reminderInterval) {
                    ForEach(ReminderInterval.allCases) { interval in
                        Text(interval.label).tag(interval)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
        .padding(20)
    }
}

struct ErrorView: View {
    let message: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.orange)
            Text(message)
                .font(.callout)
                .multilineTextAlignment(.center)
        }
        .padding(24)
    }
}

struct FloatingWindowConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            guard let window = view.window else { return }
            window.level = .floating
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            window.titleVisibility = .hidden
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
