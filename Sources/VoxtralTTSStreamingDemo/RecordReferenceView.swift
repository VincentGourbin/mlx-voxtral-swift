import SwiftUI

/// Record an enrollment reference from the microphone while the user reads a
/// prompt aloud. When the recording reaches the target length it becomes the
/// enrollment reference.
@available(macOS 14.0, *)
struct RecordReferenceView: View {
    @ObservedObject var vm: StreamingDemoViewModel
    @Environment(\.dismiss) private var dismiss

    private var target: Double { vm.cloneDuration }
    private var prompt: StreamingDemoViewModel.RecordPrompt { vm.recordPrompts[vm.recordPromptIndex] }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Record reference from microphone")
                .font(.headline)

            // Prompt selector
            HStack {
                Text("Read this aloud:").font(.caption.bold())
                Spacer()
                Picker("", selection: $vm.recordPromptIndex) {
                    ForEach(Array(vm.recordPrompts.enumerated()), id: \.offset) { i, p in
                        Text("\(p.lang) #\(i + 1)").tag(i)
                    }
                }
                .labelsHidden().frame(width: 110)
                .disabled(vm.isRecording)
            }

            // The reference text
            Text(prompt.text)
                .font(.title3)
                .fixedSize(horizontal: false, vertical: true)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.gray.opacity(0.1)))

            // Timer + progress toward the target length
            VStack(spacing: 6) {
                HStack {
                    Text(String(format: "%.1f s", vm.recordElapsed))
                        .font(.system(.title2, design: .monospaced))
                        .foregroundStyle(vm.recordElapsed >= target ? .green : .primary)
                    Text("/ \(String(format: "%.0f", target)) s target")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if vm.isRecording {
                        Circle().fill(.red).frame(width: 10, height: 10)
                            .opacity(0.6).symbolEffect(.pulse)
                        Text("REC").font(.caption.bold()).foregroundStyle(.red)
                    }
                }
                ProgressView(value: min(vm.recordElapsed, target), total: target)
            }

            if !vm.micStatus.isEmpty {
                Text(vm.micStatus).font(.caption2.monospaced()).foregroundStyle(.secondary)
            }

            Divider()
            HStack {
                Button("Cancel") { if vm.isRecording { vm.stopRecording() }; dismiss() }
                Spacer()
                if !vm.isRecording {
                    Button {
                        vm.startRecording()
                    } label: {
                        Label("Record", systemImage: "mic.fill")
                    }
                    .buttonStyle(.borderedProminent).tint(.red)
                } else {
                    Button {
                        vm.stopRecording()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.recordElapsed < target)
                }
            }
            // Close once a valid reference has been recorded.
            .onChange(of: vm.referenceURL) { _, url in
                if url != nil && !vm.isRecording && vm.micStatus.hasPrefix("Reference recorded") {
                    dismiss()
                }
            }
        }
        .padding(20)
        .frame(width: 520)
    }
}
