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

// MARK: - App Tab Enum

enum AppTab: String, CaseIterable {
    case main = "Main"
    case models = "Models"

    var icon: String {
        switch self {
        case .main: return "waveform.circle"
        case .models: return "square.stack.3d.down.right"
        }
    }
}

struct ContentView: View {
    @StateObject private var manager = TranscriptionManager()
    @State private var selectedTab: AppTab = .main

    var body: some View {
        VStack(spacing: 0) {
            HeaderView(manager: manager, selectedTab: $selectedTab)
            Divider()

            switch selectedTab {
            case .main:
                HStack(spacing: 0) {
                    ControlPanelView(manager: manager)
                        .frame(width: 300)
                    Divider()
                    OutputPanelView(manager: manager)
                }
            case .models:
                ModelsManagementView(manager: manager)
            }
        }
        .frame(minWidth: 800, minHeight: 550)
    }
}

// MARK: - Header View

struct HeaderView: View {
    @ObservedObject var manager: TranscriptionManager
    @Binding var selectedTab: AppTab

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "waveform.circle.fill")
                    .font(.title)
                    .foregroundStyle(.blue)

                Text("Voxtral")
                    .font(.title2.bold())

                // Tab picker
                Picker("", selection: $selectedTab) {
                    ForEach(AppTab.allCases, id: \.self) { tab in
                        Label(tab.rawValue, systemImage: tab.icon).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 180)

                Spacer()

                if manager.isModelLoaded {
                    if let model = manager.selectedModel {
                        Text(model.name)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Label("Ready", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)

                    // Unload button in main header
                    Button(action: {
                        manager.unloadModel()
                    }) {
                        Label("Unload", systemImage: "xmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .tint(.orange)
                    .help("Unload model to free GPU memory")
                } else if manager.isLoading {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text(manager.loadingStatus.isEmpty ? "Loading..." : manager.loadingStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                } else if let error = manager.errorMessage {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .font(.caption)
                        .lineLimit(2)
                    Button("Retry") {
                        manager.errorMessage = nil
                        Task { await manager.loadModel() }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                } else {
                    Button("Load Model") {
                        Task { await manager.loadModel() }
                    }
                }
            }
            .padding()

            // Download progress bar
            if manager.isDownloading {
                VStack(spacing: 4) {
                    ProgressView(value: manager.downloadProgress)
                        .progressViewStyle(.linear)
                    Text(manager.downloadMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                .padding(.bottom, 8)
            }
        }
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

                    // Context Size control for memory management
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Context Size:")
                            Spacer()
                            Text("\(manager.contextSize / 1024)k")
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        Slider(
                            value: Binding(
                                get: { Double(manager.contextSize) },
                                set: { manager.contextSize = Int($0) }
                            ),
                            in: 1024...32768,
                            step: 1024
                        )
                        Text("Lower = less GPU memory")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    Toggle("Detailed profiling", isOn: $manager.profilingEnabled)
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
                LiveStatsBarView(tokenCount: manager.currentTokenCount, currentStep: manager.currentStep)
            } else if let stats = manager.lastGenerationStats {
                Divider()
                StatsBarView(stats: stats, profileSummary: manager.lastProfileSummary)
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
    let currentStep: TranscriptionManager.ProcessingStep

    var body: some View {
        HStack(spacing: 16) {
            ProgressView()
                .scaleEffect(0.7)

            if !currentStep.icon.isEmpty {
                Image(systemName: currentStep.icon)
                    .foregroundStyle(.blue)
            }

            Text(currentStep == .generating ? "\(currentStep.rawValue) (\(tokenCount) tokens)..." : currentStep.rawValue)
                .foregroundStyle(.blue)
                .fontWeight(.medium)

            Spacer()

            if currentStep == .settingUpGeneration {
                Text("This may take a moment...")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .font(.caption)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Stats Bar View (final)

struct StatsBarView: View {
    let stats: GenerationStats
    let profileSummary: ProfileSummary?
    @State private var showProfileDetails = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 20) {
                Label("\(stats.tokenCount) tokens", systemImage: "number")
                Label(String(format: "%.1fs", stats.duration), systemImage: "clock")
                Label(String(format: "%.1f tok/s", stats.tokensPerSecond), systemImage: "speedometer")

                Spacer()

                if profileSummary != nil {
                    Button(action: { showProfileDetails.toggle() }) {
                        Label(showProfileDetails ? "Hide Profile" : "Show Profile",
                              systemImage: showProfileDetails ? "chevron.up" : "chevron.down")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.blue)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal)
            .padding(.vertical, 8)

            if showProfileDetails, let summary = profileSummary {
                ProfileDetailsView(summary: summary)
            }
        }
        .background(.ultraThinMaterial)
    }
}

// MARK: - Profile Details View

struct ProfileDetailsView: View {
    let summary: ProfileSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()

            // Device info
            HStack {
                Text("Device:")
                    .foregroundStyle(.secondary)
                Text(summary.deviceInfo.architecture)
                    .fontWeight(.medium)
                Spacer()
                Text("RAM: \(formatBytes(summary.deviceInfo.memorySize))")
                    .foregroundStyle(.secondary)
            }
            .font(.caption)

            Divider()

            // Steps table header
            HStack {
                Text("Step")
                    .frame(width: 140, alignment: .leading)
                Text("Time")
                    .frame(width: 70, alignment: .trailing)
                Text("MLX Δ")
                    .frame(width: 80, alignment: .trailing)
                Text("Process Δ")
                    .frame(width: 80, alignment: .trailing)
            }
            .font(.caption2.bold())
            .foregroundStyle(.secondary)

            // Steps
            ForEach(Array(summary.steps.enumerated()), id: \.offset) { _, step in
                HStack {
                    Text(step.name)
                        .frame(width: 140, alignment: .leading)
                        .lineLimit(1)
                    Text(String(format: "%.3fs", step.duration))
                        .frame(width: 70, alignment: .trailing)
                    Text(formatDeltaBytes(step.endMemory.mlxActive - step.startMemory.mlxActive))
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(step.endMemory.mlxActive > step.startMemory.mlxActive ? .orange : .green)
                    Text(formatDeltaBytes(Int(step.endMemory.processFootprint - step.startMemory.processFootprint)))
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(step.endMemory.processFootprint > step.startMemory.processFootprint ? .orange : .green)
                }
                .font(.caption)
            }

            Divider()

            // Totals
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Peak")
                        .foregroundStyle(.secondary)
                    Text(formatBytes(summary.peakMemoryUsed))
                        .fontWeight(.medium)
                        .foregroundStyle(.orange)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Active")
                        .foregroundStyle(.secondary)
                    Text(formatBytes(summary.finalSnapshot.mlxActive))
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Cache")
                        .foregroundStyle(.secondary)
                    Text(formatBytes(summary.finalSnapshot.mlxCache))
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("Process")
                        .foregroundStyle(.secondary)
                    Text(formatBytes(Int(summary.finalSnapshot.processFootprint)))
                        .fontWeight(.medium)
                        .foregroundStyle(.blue)
                }

                Spacer()
            }
            .font(.caption)
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }
}

// Helper functions for formatting
private func formatBytes(_ bytes: Int) -> String {
    let absBytes = abs(bytes)
    if absBytes >= 1024 * 1024 * 1024 {
        return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
    } else if absBytes >= 1024 * 1024 {
        return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
    } else if absBytes >= 1024 {
        return String(format: "%.1f KB", Double(bytes) / 1024)
    }
    return "\(bytes) B"
}

private func formatDeltaBytes(_ bytes: Int) -> String {
    let sign = bytes >= 0 ? "+" : ""
    return sign + formatBytes(bytes)
}

// MARK: - Generation Stats

struct GenerationStats {
    let tokenCount: Int
    let duration: Double
    var tokensPerSecond: Double { Double(tokenCount) / max(duration, 0.001) }
}

// MARK: - Models Management View

struct ModelsManagementView: View {
    @ObservedObject var manager: TranscriptionManager
    @State private var modelToDelete: VoxtralModelInfo?
    @State private var showDeleteConfirmation = false
    @State private var memoryRefreshTrigger = false

    var body: some View {
        VStack(spacing: 0) {
            // Memory status bar
            HStack(spacing: 16) {
                Label("MLX Memory", systemImage: "memorychip")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    Text("Active: \(TranscriptionManager.formatBytes(manager.memoryStats.active))")
                    Text("Cache: \(TranscriptionManager.formatBytes(manager.memoryStats.cache))")
                        .foregroundStyle(manager.memoryStats.cache > 0 ? .orange : .secondary)
                    Text("Peak: \(TranscriptionManager.formatBytes(manager.memoryStats.peak))")
                        .foregroundStyle(.blue)
                }
                .font(.caption.monospaced())

                Spacer()

                Button(action: {
                    manager.clearCache()
                    GPU.resetPeakMemory()
                    memoryRefreshTrigger.toggle()
                }) {
                    Label("Clear Cache", systemImage: "trash.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(manager.memoryStats.cache == 0)
                .help("Clear MLX recyclable cache")

                Button(action: {
                    manager.unloadModel()
                    memoryRefreshTrigger.toggle()
                }) {
                    Label("Unload Model", systemImage: "xmark.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(.orange)
                .disabled(!manager.isModelLoaded)
                .help("Unload model to free all GPU memory")
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial)
            .id(memoryRefreshTrigger)  // Force refresh

            Divider()

            // Toolbar
            HStack {
                Text("Downloaded Models")
                    .font(.headline)
                Spacer()
                Button(action: { manager.refreshDownloadedModels() }) {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh")
            }
            .padding()

            Divider()

            if manager.downloadedModels.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "square.stack.3d.down.right")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary.opacity(0.5))
                    Text("No models downloaded")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    Text("Download models from the Main tab to get started")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(manager.availableModels.filter { manager.downloadedModels.contains($0.id) }, id: \.id) { model in
                        ModelRowView(
                            model: model,
                            size: manager.modelSizes[model.id],
                            isLoaded: manager.currentLoadedModelId == model.id,
                            onDelete: {
                                modelToDelete = model
                                showDeleteConfirmation = true
                            },
                            onLoad: {
                                manager.selectedModelId = model.id
                                Task { await manager.loadModel() }
                            }
                        )
                    }
                }
                .listStyle(.inset)
            }

            Divider()

            // Available models section
            VStack(alignment: .leading, spacing: 8) {
                Text("Available Models")
                    .font(.headline)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(manager.availableModels.filter { !manager.downloadedModels.contains($0.id) }, id: \.id) { model in
                            AvailableModelCard(model: model, manager: manager)
                        }
                    }
                    .padding(.horizontal, 4)
                }
            }
            .padding()
            .background(.ultraThinMaterial)
        }
        .alert("Delete Model", isPresented: $showDeleteConfirmation, presenting: modelToDelete) { model in
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                Task {
                    try? await manager.deleteModel(model.id)
                }
            }
        } message: { model in
            Text("Are you sure you want to delete \(model.name)? This cannot be undone.")
        }
    }
}

struct ModelRowView: View {
    let model: VoxtralModelInfo
    let size: Int64?
    let isLoaded: Bool
    let onDelete: () -> Void
    let onLoad: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(model.name)
                        .font(.headline)
                    if isLoaded {
                        Text("Loaded")
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.2))
                            .foregroundStyle(.green)
                            .cornerRadius(4)
                    }
                }
                HStack(spacing: 8) {
                    Text(model.quantization)
                    Text("•")
                    Text(model.parameters)
                    if let size = size {
                        Text("•")
                        Text(ModelDownloader.formatSize(size))
                            .foregroundStyle(.blue)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            if !isLoaded {
                Button("Load") {
                    onLoad()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Button(action: onDelete) {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
            .disabled(isLoaded)
            .help(isLoaded ? "Unload model first" : "Delete model")
        }
        .padding(.vertical, 4)
    }
}

struct AvailableModelCard: View {
    let model: VoxtralModelInfo
    @ObservedObject var manager: TranscriptionManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.name)
                .font(.caption.bold())
                .lineLimit(1)

            HStack(spacing: 4) {
                Text(model.size)
                Text("•")
                Text(model.quantization)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            Button(action: {
                Task { await manager.downloadModel(model.id) }
            }) {
                HStack {
                    Image(systemName: "arrow.down.circle")
                    Text("Download")
                }
                .font(.caption)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(manager.isDownloading)
        }
        .padding(10)
        .frame(width: 140)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }
}

#Preview {
    ContentView()
}
