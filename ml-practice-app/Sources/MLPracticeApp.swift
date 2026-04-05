import SwiftUI
import UserNotifications

@main
struct MLPracticeApp: App {
    @StateObject private var store = ProblemStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
                .frame(minWidth: 900, minHeight: 600)
                .onAppear {
                    if store.practiceRootPath == nil {
                        store.showDirectoryPicker = true
                    }
                    // Request notification permission after app window is up
                    requestNotificationPermission()
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1100, height: 750)

        Settings {
            SettingsView()
                .environmentObject(store)
        }
    }

    private func requestNotificationPermission() {
        guard Bundle.main.bundleIdentifier != nil else { return }
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { _, _ in }
    }
}
