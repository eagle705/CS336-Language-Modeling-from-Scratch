// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLPopupQuiz",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "MLPopupQuiz",
            path: "Sources",
            exclude: ["Resources/Info.plist", "Resources/AppIcon.icns"],
            resources: [
                .process("Resources/quiz_bank.json")
            ]
        )
    ]
)
