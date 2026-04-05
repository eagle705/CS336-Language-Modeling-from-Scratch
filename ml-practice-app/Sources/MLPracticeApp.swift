import SwiftUI
import UserNotifications

@main
struct MLPracticeApp: App {
    @StateObject private var store = ProblemStore()

    init() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { _, _ in }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
                .frame(minWidth: 900, minHeight: 600)
                .onAppear {
                    if store.practiceRootPath == nil {
                        store.showDirectoryPicker = true
                    }
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1100, height: 750)

        Settings {
            SettingsView()
                .environmentObject(store)
        }
    }
}
