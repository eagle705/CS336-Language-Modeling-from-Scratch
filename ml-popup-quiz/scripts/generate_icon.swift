#!/usr/bin/env swift
import AppKit

func drawIcon(size: CGFloat) -> NSImage {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()

    guard let context = NSGraphicsContext.current?.cgContext else {
        image.unlockFocus()
        return image
    }

    let rect = CGRect(x: 0, y: 0, width: size, height: size)
    let cornerRadius = size * 0.22
    let bgPath = CGPath(roundedRect: rect, cornerWidth: cornerRadius, cornerHeight: cornerRadius, transform: nil)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let gradient = CGGradient(
        colorsSpace: colorSpace,
        colors: [
            CGColor(red: 0.08, green: 0.18, blue: 0.24, alpha: 1.0),
            CGColor(red: 0.02, green: 0.45, blue: 0.50, alpha: 1.0),
            CGColor(red: 0.90, green: 0.72, blue: 0.25, alpha: 1.0)
        ] as CFArray,
        locations: [0.0, 0.62, 1.0]
    )!

    context.saveGState()
    context.addPath(bgPath)
    context.clip()
    context.drawLinearGradient(
        gradient,
        start: CGPoint(x: 0, y: size),
        end: CGPoint(x: size, y: 0),
        options: []
    )
    context.restoreGState()

    context.saveGState()
    context.addPath(bgPath)
    context.clip()

    let cardRect = CGRect(
        x: size * 0.18,
        y: size * 0.20,
        width: size * 0.64,
        height: size * 0.58
    )
    let cardPath = CGPath(roundedRect: cardRect, cornerWidth: size * 0.055, cornerHeight: size * 0.055, transform: nil)
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 0.92))
    context.addPath(cardPath)
    context.fillPath()

    context.setStrokeColor(CGColor(red: 0.05, green: 0.16, blue: 0.20, alpha: 0.20))
    context.setLineWidth(size * 0.012)
    context.addPath(cardPath)
    context.strokePath()

    let checkCenter = CGPoint(x: size * 0.67, y: size * 0.63)
    context.setFillColor(CGColor(red: 0.04, green: 0.55, blue: 0.36, alpha: 1.0))
    context.fillEllipse(in: CGRect(
        x: checkCenter.x - size * 0.095,
        y: checkCenter.y - size * 0.095,
        width: size * 0.19,
        height: size * 0.19
    ))
    context.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1.0))
    context.setLineWidth(size * 0.022)
    context.setLineCap(.round)
    context.setLineJoin(.round)
    context.move(to: CGPoint(x: size * 0.62, y: size * 0.63))
    context.addLine(to: CGPoint(x: size * 0.655, y: size * 0.59))
    context.addLine(to: CGPoint(x: size * 0.72, y: size * 0.68))
    context.strokePath()

    let lineColor = CGColor(red: 0.10, green: 0.18, blue: 0.22, alpha: 0.82)
    context.setStrokeColor(lineColor)
    context.setLineCap(.round)
    context.setLineWidth(size * 0.018)
    for i in 0..<4 {
        let y = size * (0.50 - CGFloat(i) * 0.085)
        context.move(to: CGPoint(x: size * 0.29, y: y))
        context.addLine(to: CGPoint(x: size * (i == 0 ? 0.53 : 0.70), y: y))
        context.strokePath()
    }

    let font = NSFont.monospacedSystemFont(ofSize: size * 0.16, weight: .bold)
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: NSColor(red: 0.08, green: 0.20, blue: 0.24, alpha: 1.0)
    ]
    let text = "ML?" as NSString
    let textSize = text.size(withAttributes: attrs)
    text.draw(at: CGPoint(x: size * 0.27, y: size * 0.60 - textSize.height / 2), withAttributes: attrs)

    context.restoreGState()
    image.unlockFocus()
    return image
}

let outputPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "Sources/Resources/AppIcon.icns"

let iconsetURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("MLPopupQuiz.iconset")
let fileManager = FileManager.default
try? fileManager.removeItem(at: iconsetURL)
try fileManager.createDirectory(at: iconsetURL, withIntermediateDirectories: true)

let sizes: [(String, CGFloat)] = [
    ("icon_16x16", 16), ("icon_16x16@2x", 32),
    ("icon_32x32", 32), ("icon_32x32@2x", 64),
    ("icon_128x128", 128), ("icon_128x128@2x", 256),
    ("icon_256x256", 256), ("icon_256x256@2x", 512),
    ("icon_512x512", 512), ("icon_512x512@2x", 1024)
]

for (name, size) in sizes {
    let image = drawIcon(size: size)
    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        continue
    }
    try png.write(to: iconsetURL.appendingPathComponent("\(name).png"))
}

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconsetURL.path, "-o", outputPath]
try process.run()
process.waitUntilExit()

if process.terminationStatus != 0 {
    throw NSError(domain: "IconGeneration", code: Int(process.terminationStatus))
}

print("Icon created: \(outputPath)")
