import SwiftUI
import Charts

struct DashboardView: View {
    @EnvironmentObject var store: ProblemStore

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Title
                HStack {
                    VStack(alignment: .leading) {
                        Text("Dashboard")
                            .font(.largeTitle.bold())
                        Text(Date(), format: .dateTime.month().day().weekday(.wide))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }

                // Summary cards
                summaryCards

                // Today's quiz
                todaySection

                // Category breakdown
                categoryBreakdown

                // Motivation
                motivationSection
            }
            .padding(24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Summary Cards

    private var summaryCards: some View {
        HStack(spacing: 16) {
            StatCard(title: "Total", value: "\(store.problems.count)",
                     icon: "book.closed", color: .blue)
            StatCard(title: "Solved", value: "\(store.solvedCount)",
                     icon: "checkmark.circle.fill", color: .green)
            StatCard(title: "In Progress", value: "\(store.inProgressCount)",
                     icon: "circle.lefthalf.filled", color: .orange)
            StatCard(title: "Unsolved", value: "\(store.unsolvedCount)",
                     icon: "circle", color: .gray)
        }
    }

    // MARK: - Today

    private var todaySection: some View {
        GroupBox("Today's Quiz") {
            let today = store.todayProblems
            if today.isEmpty {
                Text("No problems scheduled today. Press refresh to schedule.")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                VStack(spacing: 8) {
                    ForEach(today) { problem in
                        HStack {
                            Image(systemName: problem.state.icon)
                                .foregroundStyle(problem.state.color)
                            Text(problem.title)
                                .font(.system(size: 13))
                            Spacer()
                            Text(problem.state.label)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }

    // MARK: - Category Breakdown

    private var categoryBreakdown: some View {
        GroupBox("Progress by Category") {
            VStack(spacing: 12) {
                ForEach(Category.allCases, id: \.self) { category in
                    let stat = store.stats(for: category)
                    if stat.total > 0 {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(category.rawValue)
                                    .font(.system(size: 13, weight: .medium))
                                Spacer()
                                Text("\(stat.solved)/\(stat.total)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            ProgressView(value: Double(stat.solved), total: Double(stat.total))
                                .tint(stat.solved == stat.total ? .green : .blue)
                        }
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Motivation

    private var motivationSection: some View {
        GroupBox {
            VStack(spacing: 8) {
                let pct = store.problems.isEmpty ? 0 :
                    Double(store.solvedCount) / Double(store.problems.count) * 100
                Text(motivationalMessage(percent: pct))
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
            }
            .padding(8)
        }
    }

    private func motivationalMessage(percent: Double) -> String {
        switch percent {
        case 0..<10: return "Every expert was once a beginner. Start with one problem today!"
        case 10..<30: return "Great start! Consistency beats intensity. Keep going."
        case 30..<50: return "You're building solid foundations. The patterns will start clicking soon."
        case 50..<70: return "More than halfway! You're well-prepared for most interview topics."
        case 70..<90: return "Impressive progress. Focus on the remaining weak areas."
        case 90..<100: return "Almost there! Review the solved problems to keep them fresh."
        default: return "You've completed everything! Consider revisiting for deeper understanding."
        }
    }
}

// MARK: - Stat Card

struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundStyle(color)
            Text(value)
                .font(.title.bold())
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(color.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }
}
