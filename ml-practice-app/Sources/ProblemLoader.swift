import Foundation

enum ProblemLoader {
    /// Scan the implementation-practice directory and build Problem list.
    static func scan(rootPath: String) -> [Problem] {
        let fm = FileManager.default
        let rootURL = URL(fileURLWithPath: rootPath)

        guard let contents = try? fm.contentsOfDirectory(
            at: rootURL, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        let dirs = contents
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .filter { $0.lastPathComponent.first?.isNumber == true }  // "01-xxx", "02-xxx"
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        return dirs.compactMap { dir in
            let dirName = dir.lastPathComponent

            // Find .py files
            guard let pyFiles = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .filter({ $0.pathExtension == "py" })
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            else { return nil }

            if pyFiles.isEmpty { return nil }

            let files = pyFiles.map { url in
                ProblemFile(
                    id: url.lastPathComponent,
                    path: url.path,
                    name: url.deletingPathExtension().lastPathComponent
                )
            }

            // Extract title from first file's docstring
            let title = extractTitle(from: pyFiles.first!) ?? dirName

            return Problem(
                id: dirName,
                title: title,
                category: Category.from(directoryName: dirName),
                files: files
            )
        }
    }

    /// Extract the title from a Python file's docstring (first line after """).
    private static func extractTitle(from url: URL) -> String? {
        guard let content = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let lines = content.components(separatedBy: .newlines)

        // Find opening """
        guard let docStart = lines.firstIndex(where: { $0.trimmingCharacters(in: .whitespaces).hasPrefix("\"\"\"") }) else {
            return nil
        }

        // Title is the first non-empty line (could be on same line as """ or next line)
        let firstLine = lines[docStart].trimmingCharacters(in: .whitespaces)
        if firstLine.count > 3 {
            // Title on same line as """
            return String(firstLine.dropFirst(3)).trimmingCharacters(in: .whitespaces)
        }

        // Title on next line
        for i in (docStart + 1)..<min(docStart + 5, lines.count) {
            let line = lines[i].trimmingCharacters(in: .whitespaces)
            if !line.isEmpty && !line.allSatisfy({ $0 == "=" || $0 == "-" }) {
                return line
            }
        }

        return nil
    }
}
