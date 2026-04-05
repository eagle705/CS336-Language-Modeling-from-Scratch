#!/usr/bin/env swift
// Generates the ML Practice app icon as an .icns file using AppKit drawing.

import AppKit

func drawIcon(size: CGFloat) -> NSImage {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()

    let ctx = NSGraphicsContext.current!.cgContext
    let s = size  // shorthand

    // === Background: rounded rect with gradient ===
    let cornerRadius = s * 0.22
    let bgRect = CGRect(x: 0, y: 0, width: s, height: s)
    let bgPath = CGPath(roundedRect: bgRect, cornerWidth: cornerRadius, cornerHeight: cornerRadius, transform: nil)

    // Gradient: deep purple-blue to dark blue
    let colors = [
        CGColor(red: 0.25, green: 0.12, blue: 0.55, alpha: 1.0),
        CGColor(red: 0.10, green: 0.08, blue: 0.35, alpha: 1.0),
    ] as CFArray
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let gradient = CGGradient(colorsSpace: colorSpace, colors: colors, locations: [0, 1])!

    ctx.saveGState()
    ctx.addPath(bgPath)
    ctx.clip()
    ctx.drawLinearGradient(gradient, start: CGPoint(x: 0, y: s), end: CGPoint(x: s, y: 0), options: [])
    ctx.restoreGState()

    // === Subtle grid pattern (neural network feel) ===
    ctx.saveGState()
    ctx.addPath(bgPath)
    ctx.clip()
    ctx.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 0.04))
    ctx.setLineWidth(s * 0.004)
    let gridSpacing = s / 8
    for i in 1..<8 {
        let pos = gridSpacing * CGFloat(i)
        ctx.move(to: CGPoint(x: pos, y: 0))
        ctx.addLine(to: CGPoint(x: pos, y: s))
        ctx.move(to: CGPoint(x: 0, y: pos))
        ctx.addLine(to: CGPoint(x: s, y: pos))
    }
    ctx.strokePath()
    ctx.restoreGState()

    // === Brain/network nodes ===
    let nodeRadius = s * 0.04
    let glowRadius = s * 0.07

    // Node positions (3 layers like a neural network)
    let layer1: [(CGFloat, CGFloat)] = [(0.28, 0.72), (0.28, 0.52), (0.28, 0.32)]
    let layer2: [(CGFloat, CGFloat)] = [(0.50, 0.78), (0.50, 0.58), (0.50, 0.38), (0.50, 0.18)]
    let layer3: [(CGFloat, CGFloat)] = [(0.72, 0.65), (0.72, 0.45)]

    let allNodes = layer1 + layer2 + layer3

    // Draw connections
    ctx.saveGState()
    ctx.addPath(bgPath)
    ctx.clip()
    ctx.setLineWidth(s * 0.008)

    for n1 in layer1 {
        for n2 in layer2 {
            let alpha: CGFloat = 0.2
            ctx.setStrokeColor(CGColor(red: 0.5, green: 0.7, blue: 1.0, alpha: alpha))
            ctx.move(to: CGPoint(x: n1.0 * s, y: n1.1 * s))
            ctx.addLine(to: CGPoint(x: n2.0 * s, y: n2.1 * s))
            ctx.strokePath()
        }
    }
    for n2 in layer2 {
        for n3 in layer3 {
            let alpha: CGFloat = 0.2
            ctx.setStrokeColor(CGColor(red: 0.5, green: 0.7, blue: 1.0, alpha: alpha))
            ctx.move(to: CGPoint(x: n2.0 * s, y: n2.1 * s))
            ctx.addLine(to: CGPoint(x: n3.0 * s, y: n3.1 * s))
            ctx.strokePath()
        }
    }
    ctx.restoreGState()

    // Draw node glows + nodes
    for (nx, ny) in allNodes {
        let center = CGPoint(x: nx * s, y: ny * s)

        // Glow
        let glowColors = [
            CGColor(red: 0.4, green: 0.6, blue: 1.0, alpha: 0.4),
            CGColor(red: 0.4, green: 0.6, blue: 1.0, alpha: 0.0),
        ] as CFArray
        let glowGrad = CGGradient(colorsSpace: colorSpace, colors: glowColors, locations: [0, 1])!
        ctx.saveGState()
        ctx.drawRadialGradient(glowGrad, startCenter: center, startRadius: 0, endCenter: center, endRadius: glowRadius, options: [])
        ctx.restoreGState()

        // Node circle
        let nodeRect = CGRect(x: center.x - nodeRadius, y: center.y - nodeRadius,
                             width: nodeRadius * 2, height: nodeRadius * 2)
        ctx.setFillColor(CGColor(red: 0.6, green: 0.8, blue: 1.0, alpha: 0.95))
        ctx.fillEllipse(in: nodeRect)
    }

    // === "ML" text ===
    let fontSize = s * 0.18
    let font = NSFont.systemFont(ofSize: fontSize, weight: .heavy)
    let textAttrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: NSColor(red: 0.95, green: 0.95, blue: 1.0, alpha: 0.95),
    ]
    let text = "ML" as NSString
    let textSize = text.size(withAttributes: textAttrs)
    let textOrigin = CGPoint(x: (s - textSize.width) / 2, y: s * 0.04)
    text.draw(at: textOrigin, withAttributes: textAttrs)

    image.unlockFocus()
    return image
}

// Generate iconset
let iconsetPath = "/tmp/MLPractice.iconset"
let fm = FileManager.default
try? fm.removeItem(atPath: iconsetPath)
try! fm.createDirectory(atPath: iconsetPath, withIntermediateDirectories: true)

let sizes: [(String, CGFloat)] = [
    ("icon_16x16", 16),
    ("icon_16x16@2x", 32),
    ("icon_32x32", 32),
    ("icon_32x32@2x", 64),
    ("icon_128x128", 128),
    ("icon_128x128@2x", 256),
    ("icon_256x256", 256),
    ("icon_256x256@2x", 512),
    ("icon_512x512", 512),
    ("icon_512x512@2x", 1024),
]

for (name, size) in sizes {
    let image = drawIcon(size: size)
    guard let tiff = image.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else {
        print("Failed to generate \(name)")
        continue
    }
    let path = "\(iconsetPath)/\(name).png"
    try! png.write(to: URL(fileURLWithPath: path))
    print("Generated \(name) (\(Int(size))x\(Int(size)))")
}

print("\nConverting to .icns...")
let outputPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/tmp/MLPractice.icns"

let proc = Process()
proc.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
proc.arguments = ["-c", "icns", iconsetPath, "-o", outputPath]
try! proc.run()
proc.waitUntilExit()

if proc.terminationStatus == 0 {
    print("Icon created: \(outputPath)")
} else {
    print("iconutil failed with status \(proc.terminationStatus)")
}
