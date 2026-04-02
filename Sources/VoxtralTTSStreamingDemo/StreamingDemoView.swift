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
                // Model picker + load
                HStack {
                    Text("Model:")
                        .frame(width: 50, alignment: .trailing)
                    Picker("", selection: $vm.selectedModelId) {
                        ForEach(vm.availableModels, id: \.id) { model in
                            Text(model.name).tag(model.id)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 160)

                    Button(vm.isModelLoaded ? "Loaded" : "Load") {
                        Task { await vm.loadModel() }
                    }
                    .disabled(vm.isLoading)
                    .onChange(of: vm.selectedModelId) { _, _ in
                        vm.isModelLoaded = false
                    }

                    if vm.isLoading {
                        ProgressView(value: vm.loadProgress)
                            .frame(width: 100)
                        Text(vm.loadStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }

                    Spacer()
                }

                // Voice picker
                HStack {
                    Text("Voice:")
                        .frame(width: 50, alignment: .trailing)
                    Picker("", selection: $vm.selectedVoice) {
                        ForEach(["neutral_male", "neutral_female", "casual_male", "casual_female", "fr_male", "fr_female"], id: \.self) { v in
                            Text(v).tag(v)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 160)
                    Spacer()
                }

                // Text input
                TextEditor(text: $vm.text)
                    .font(.body)
                    .frame(height: 80)
                    .border(Color.gray.opacity(0.3))

                // Play button
                HStack(spacing: 16) {
                    Button(action: {
                        if vm.isSynthesizing {
                            vm.stop()
                        } else {
                            vm.startStreaming()
                        }
                    }) {
                        HStack {
                            Image(systemName: vm.isSynthesizing ? "stop.fill" : "play.fill")
                            Text(vm.isSynthesizing ? "Stop" : "Play Streaming")
                        }
                        .frame(width: 180, height: 36)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(vm.isSynthesizing ? .red : .blue)
                    .disabled(!vm.isModelLoaded || vm.isLoading)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            Divider()

            // Metrics
            VStack(alignment: .leading, spacing: 8) {
                Text("Metrics")
                    .font(.headline)

                HStack(spacing: 32) {
                    metricView(label: "TTFT", value: vm.ttft.map { String(format: "%.0f ms", $0 * 1000) } ?? "—", highlight: true)
                    metricView(label: "Total", value: vm.totalTime.map { String(format: "%.2fs", $0) } ?? "—")
                    metricView(label: "Audio", value: vm.audioDuration > 0 ? String(format: "%.2fs", vm.audioDuration) : "—")
                    metricView(label: "RTF", value: vm.rtf.map { String(format: "%.2fx", $0) } ?? "—")
                    metricView(label: "Frames", value: vm.framesGenerated > 0 ? "\(vm.framesGenerated)" : "—")
                    metricView(label: "Chunks", value: vm.chunksReceived > 0 ? "\(vm.chunksReceived)" : "—")
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

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
                                    .foregroundStyle(line.contains("TTFT") ? .green : line.contains("Error") ? .red : .primary)
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
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.body, design: .monospaced).bold())
                .foregroundStyle(highlight && vm.ttft != nil ? .green : .primary)
        }
    }
}
