import SwiftUI
import UserNotifications

@main
struct MLPopupQuizApp: App {
    @StateObject private var store = QuizStore()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        MenuBarExtra {
            MenuBarQuizView()
                .environmentObject(store)
                .frame(width: 460, height: 640)
        } label: {
            Image(systemName: "brain.head.profile")
        }
        .menuBarExtraStyle(.window)

        WindowGroup("ML Popup Quiz", id: "main") {
            MainQuizWindowView()
                .environmentObject(store)
                .frame(minWidth: 520, minHeight: 560)
        }
        .defaultSize(width: 720, height: 820)

        Settings {
            SettingsView()
                .environmentObject(store)
                .frame(width: 460, height: 360)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        UNUserNotificationCenter.current().delegate = self
        NSApp.setActivationPolicy(.regular)
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
    }
}
