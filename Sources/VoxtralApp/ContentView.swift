/**
 * ContentView - Main view for Voxtral audio app
 * Supports both transcription and chat modes
 */

import SwiftUI
import VoxtralCore
import MLX

// MARK: - Mode Enum

enum VoxtralMode: String, CaseIterable {
    case transcription = "Transcription"
    case chat = "Chat"

    var icon: String {
        switch self {
        case .transcription: return "waveform"
        case .chat: return "bubble.left.and.bubble.right"
        }
    }
}

struct ContentView: View {
    @StateObject private var manager = TranscriptionManager()

    var body: some View {
        VStack(spacing: 0) {
            HeaderView(manager: manager)
            Divider()

            HStack(spacing: 0) {
                ControlPanelView(manager: manager)
                    .frame(width: 300)
                Divider()
                OutputPanelView(manager: manager)
            }
        }
        .frame(minWidth: 800, minHeight: 550)
    }
}

// MARK: - Header View

struct HeaderView: View {
    @ObservedObject var manager: TranscriptionManager

    var body: some View {
        HStack {
            Image(systemName: "waveform.circle.fill")
                .font(.title)
                .foregroundStyle(.blue)

            Text("Voxtral")
                .font(.title2.bold())

            Text(manager.mode.rawValue)
                .font(.title3)
                .foregroundStyle(.secondary)

            Spacer()

            if manager.isModelLoaded {
                if let model = manager.selectedModel {
                    Text(model.name)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Label("Ready", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else if manager.isLoading {
                ProgressView()
                    .scaleEffect(0.8)
                Text(manager.loadingStatus.isEmpty ? "Loading..." : manager.loadingStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            } else {
                Button("Load Model") {
                    Task { await manager.loadModel() }
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial)
    }
}

// MARK: - Control Panel View

struct ControlPanelView: View {
    @ObservedObject var manager: TranscriptionManager

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Mode selector
                VStack(alignment: .leading, spacing: 8) {
                    Text("Mode")
                        .font(.headline)

                    Picker("", selection: $manager.mode) {
                        ForEach(VoxtralMode.allCases, id: \.self) { mode in
                            Label(mode.rawValue, systemImage: mode.icon).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Divider()

                // Audio file
                VStack(alignment: .leading, spacing: 8) {
                    Text("Audio File")
                        .font(.headline)

                    AudioDropZone(manager: manager)

                    if let audioPath = manager.selectedAudioPath {
                        HStack {
                            Image(systemName: "waveform")
                                .foregroundStyle(.blue)
                            Text(URL(fileURLWithPath: audioPath).lastPathComponent)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button(action: { manager.selectedAudioPath = nil }) {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(8)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(8)
                    }
                }

                // Chat prompt (only in chat mode)
                if manager.mode == .chat {
                    Divider()

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Prompt")
                            .font(.headline)

                        TextEditor(text: $manager.chatPrompt)
                            .font(.body)
                            .frame(height: 80)
                            .padding(4)
                            .background(Color(nsColor: .textBackgroundColor))
                            .cornerRadius(8)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                            )

                        Text("Ask a question about the audio")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Divider()

                // Model selector
                VStack(alignment: .leading, spacing: 8) {
                    Text("Model")
                        .font(.headline)

                    Picker("", selection: $manager.selectedModelId) {
                        ForEach(manager.availableModels, id: \.id) { model in
                            HStack {
                                Text(model.name)
                                if manager.downloadedModels.contains(model.id) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.green)
                                }
                            }
                            .tag(model.id)
                        }
                    }
                    .labelsHidden()

                    if let model = manager.selectedModel {
                        HStack(spacing: 4) {
                            Text(model.size)
                            Text("•")
                            Text(model.quantization)
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)

                        if !manager.isCurrentModelLoaded && manager.isModelLoaded {
                            Button("Switch Model") {
                                manager.unloadModel()
                                Task { await manager.loadModel() }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    }
                }

                Divider()

                // Settings
                VStack(alignment: .leading, spacing: 12) {
                    Text("Settings")
                        .font(.headline)

                    HStack {
                        Text("Max tokens:")
                        Spacer()
                        TextField("", value: $manager.maxTokens, format: .number)
                            .frame(width: 80)
                            .textFieldStyle(.roundedBorder)
                            .multilineTextAlignment(.trailing)
                    }

                    HStack {
                        Text("Temperature:")
                        Spacer()
                        Slider(value: $manager.temperature, in: 0...1, step: 0.1)
                            .frame(width: 100)
                        Text(String(format: "%.1f", manager.temperature))
                            .frame(width: 30)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer(minLength: 20)

                // Action button
                Button(action: {
                    Task { await manager.run() }
                }) {
                    HStack {
                        if manager.isTranscribing {
                            ProgressView()
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: manager.mode == .transcription ? "text.bubble" : "paperplane.fill")
                        }
                        Text(manager.isTranscribing ? "Processing..." : (manager.mode == .transcription ? "Transcribe" : "Send"))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!manager.canRun)
            }
            .padding()
        }
    }
}

// MARK: - Audio Drop Zone

struct AudioDropZone: View {
    @ObservedObject var manager: TranscriptionManager
    @State private var isTargeted = false

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "arrow.down.doc")
                .font(.title2)
                .foregroundStyle(isTargeted ? .blue : .secondary)

            Text("Drop audio file")
                .font(.callout)

            Button("Browse...") {
                selectAudioFile()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 100)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(
                    isTargeted ? Color.blue : Color.gray.opacity(0.3),
                    style: StrokeStyle(lineWidth: 2, dash: [6])
                )
        )
        .background(isTargeted ? Color.blue.opacity(0.05) : Color.clear)
        .cornerRadius(10)
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            handleDrop(providers: providers)
        }
    }

    private func selectAudioFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .mpeg, .wav, .mp3]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false

        if panel.runModal() == .OK, let url = panel.url {
            manager.selectedAudioPath = url.path
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }

        provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { item, _ in
            if let data = item as? Data,
               let url = URL(dataRepresentation: data, relativeTo: nil) {
                DispatchQueue.main.async {
                    manager.selectedAudioPath = url.path
                }
            }
        }
        return true
    }
}

// MARK: - Output Panel View

struct OutputPanelView: View {
    @ObservedObject var manager: TranscriptionManager

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Toolbar
            HStack {
                Text(manager.mode == .transcription ? "Transcription" : "Response")
                    .font(.headline)

                Spacer()

                if !manager.transcription.isEmpty {
                    Button(action: copyToClipboard) {
                        Image(systemName: "doc.on.doc")
                    }
                    .help("Copy to clipboard")

                    Button(action: { manager.transcription = "" }) {
                        Image(systemName: "trash")
                    }
                    .help("Clear")
                }
            }
            .padding()

            Divider()

            // Content
            ScrollView {
                if manager.transcription.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: manager.mode == .transcription ? "text.quote" : "bubble.left.and.bubble.right")
                            .font(.system(size: 48))
                            .foregroundStyle(.secondary.opacity(0.5))

                        Text(manager.mode == .transcription
                             ? "Transcription will appear here"
                             : "Chat response will appear here")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(.top, 100)
                } else {
                    Text(manager.transcription)
                        .font(.body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
            }

            // Stats bar - show live during generation, final stats after
            if manager.isTranscribing {
                Divider()
                LiveStatsBarView(tokenCount: manager.currentTokenCount)
            } else if let stats = manager.lastGenerationStats {
                Divider()
                StatsBarView(stats: stats)
            }
        }
    }

    private func copyToClipboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(manager.transcription, forType: .string)
    }
}

// MARK: - Live Stats Bar View (during generation)

struct LiveStatsBarView: View {
    let tokenCount: Int

    var body: some View {
        HStack(spacing: 20) {
            ProgressView()
                .scaleEffect(0.7)
            Label("\(tokenCount) tokens", systemImage: "number")
            Text("Generating...")
                .foregroundStyle(.blue)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Stats Bar View (final)

struct StatsBarView: View {
    let stats: GenerationStats

    var body: some View {
        HStack(spacing: 20) {
            Label("\(stats.tokenCount) tokens", systemImage: "number")
            Label(String(format: "%.1fs", stats.duration), systemImage: "clock")
            Label(String(format: "%.1f tok/s", stats.tokensPerSecond), systemImage: "speedometer")
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Generation Stats

struct GenerationStats {
    let tokenCount: Int
    let duration: Double
    var tokensPerSecond: Double { Double(tokenCount) / max(duration, 0.001) }
}

#Preview {
    ContentView()
}
