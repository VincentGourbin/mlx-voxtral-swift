import SwiftUI
import VoxtralCore

@available(macOS 14.0, *)
struct StreamingDemoView: View {
    @StateObject private var vm = StreamingDemoViewModel()

    var body: some View {
        VStack(spacing: 0) {
            // Header
            Text("Voxtral TTS Streaming Benchmark")
                .font(.title2.bold())
                .padding(.top, 16)
                .padding(.bottom, 8)

            Divider()

            // Controls
            VStack(spacing: 12) {
                // Model + Voice row
                HStack(spacing: 16) {
                    HStack(spacing: 6) {
                        Text("Model:")
                            .font(.caption).foregroundStyle(.secondary)
                        Picker("", selection: $vm.selectedModelId) {
                            ForEach(vm.availableModels, id: \.id) { model in
                                Text(model.name).tag(model.id)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 140)
                    }

                    Button(vm.isModelLoaded ? "Loaded" : "Load") {
                        Task { await vm.loadModel() }
                    }
                    .disabled(vm.isLoading)
                    .onChange(of: vm.selectedModelId) { _, _ in
                        vm.isModelLoaded = false
                    }

                    HStack(spacing: 6) {
                        Text("Voice:")
                            .font(.caption).foregroundStyle(.secondary)
                        Picker("", selection: $vm.selectedVoice) {
                            ForEach(vm.availableVoices, id: \.id) { voice in
                                Text(voice.label).tag(voice.id)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 150)
                    }

                    Toggle("Sanitize", isOn: $vm.sanitizeEnabled)
                        .toggleStyle(.checkbox)
                        .font(.caption)

                    if vm.isLoading {
                        ProgressView(value: vm.loadProgress)
                            .frame(width: 80)
                    }

                    Spacer()
                }

                // Text presets
                HStack(spacing: 8) {
                    Text("Presets:")
                        .font(.caption).foregroundStyle(.secondary)
                    ForEach(vm.textPresets, id: \.label) { preset in
                        Button(preset.label) {
                            vm.text = preset.text
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .font(.caption)
                    }
                    Spacer()
                }

                // Text input
                TextEditor(text: $vm.text)
                    .font(.system(.body, design: .default))
                    .frame(height: 90)
                    .border(Color.gray.opacity(0.3))
                    .cornerRadius(4)

                // Play button
                Button(action: {
                    if vm.isSynthesizing { vm.stop() } else { vm.startStreaming() }
                }) {
                    HStack {
                        Image(systemName: vm.isSynthesizing ? "stop.fill" : "play.fill")
                        Text(vm.isSynthesizing ? "Stop" : "Play Streaming")
                    }
                    .frame(width: 200, height: 36)
                }
                .buttonStyle(.borderedProminent)
                .tint(vm.isSynthesizing ? .red : .accentColor)
                .disabled(!vm.isModelLoaded || vm.isLoading)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            // Metrics
            HStack(spacing: 24) {
                metricView(label: "TTFT", value: vm.ttft.map { String(format: "%.0fms", $0 * 1000) } ?? "—", highlight: true)
                metricView(label: "Total", value: vm.totalTime.map { String(format: "%.2fs", $0) } ?? "—")
                metricView(label: "Audio", value: vm.audioDuration > 0 ? String(format: "%.2fs", vm.audioDuration) : "—")
                metricView(label: "RTF", value: vm.rtf.map { String(format: "%.2fx", $0) } ?? "—")
                metricView(label: "FPS", value: vm.fps > 0 ? String(format: "%.1f", vm.fps) : "—")
                metricView(label: "Frames", value: vm.framesGenerated > 0 ? "\(vm.framesGenerated)" : "—")
                metricView(label: "Chunks", value: vm.chunksReceived > 0 ? "\(vm.chunksReceived)" : "—")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()

            // Log console
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Console")
                        .font(.headline)
                    Spacer()
                    Button("Clear") { vm.logLines.removeAll() }
                        .buttonStyle(.borderless)
                        .font(.caption)
                }

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(vm.logLines.enumerated()), id: \.offset) { idx, line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(
                                        line.contains("TTFT") ? .green :
                                        line.contains("Error") ? .red :
                                        line.contains("Done") ? .blue : .primary
                                    )
                                    .id(idx)
                            }
                        }
                    }
                    .onChange(of: vm.logLines.count) { _, _ in
                        if let last = vm.logLines.indices.last {
                            proxy.scrollTo(last, anchor: .bottom)
                        }
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
    }

    @ViewBuilder
    private func metricView(label: String, value: String, highlight: Bool = false) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.callout, design: .monospaced).bold())
                .foregroundStyle(highlight && vm.ttft != nil ? .green : .primary)
        }
    }
}
