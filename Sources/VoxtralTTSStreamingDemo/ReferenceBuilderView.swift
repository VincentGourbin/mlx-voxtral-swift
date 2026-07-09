import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Build a voice-cloning reference from a video/audio file: pick the source,
/// select one or more extracts (start/end sliders + preview), and assemble
/// them into a single reference of the target length.
@available(macOS 14.0, *)
struct ReferenceBuilderView: View {
    @ObservedObject var vm: StreamingDemoViewModel
    @Environment(\.dismiss) private var dismiss

    private var target: Double { vm.cloneDuration }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Build reference from video/audio")
                .font(.headline)

            if !vm.ffmpegAvailable {
                Label("ffmpeg is required for video/webm. Install it: brew install ffmpeg",
                      systemImage: "exclamationmark.triangle")
                    .font(.callout).foregroundStyle(.orange)
            }

            // Source picker
            HStack {
                Button("Choose video/audio…") { pickSource() }
                    .disabled(!vm.ffmpegAvailable)
                if let src = vm.refSourceURL {
                    Text(src.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                }
                Spacer()
            }

            if vm.refSourceDuration > 0 {
                Divider()

                // Extract selector
                VStack(alignment: .leading, spacing: 8) {
                    Text("Select an extract")
                        .font(.caption.bold())

                    sliderRow("Start", value: $vm.segStart, range: 0...max(0, vm.refSourceDuration - 0.5))
                        .onChange(of: vm.segStart) { _, s in
                            if vm.segEnd <= s { vm.segEnd = min(s + 1, vm.refSourceDuration) }
                        }
                    sliderRow("End", value: $vm.segEnd, range: 0...vm.refSourceDuration)
                        .onChange(of: vm.segEnd) { _, e in
                            if e <= vm.segStart { vm.segStart = max(0, e - 1) }
                        }

                    HStack {
                        Text(String(format: "Extract length: %.1f s", max(0, vm.segEnd - vm.segStart)))
                            .font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button { vm.previewSegment() } label: { Label("Preview", systemImage: "play.circle") }
                            .controlSize(.small)
                        Button { vm.addExtract() } label: { Label("Add extract", systemImage: "plus.circle") }
                            .controlSize(.small)
                            .disabled(vm.segEnd - vm.segStart < 0.5)
                    }
                }

                Divider()

                // Chosen extracts + total vs target
                HStack {
                    Text("Extracts").font(.caption.bold())
                    Spacer()
                    Text(String(format: "%.1f / %.0f s", vm.refExtractsTotal, target))
                        .font(.caption.monospaced())
                        .foregroundStyle(vm.refExtractsTotal >= target ? .green : .secondary)
                }
                if vm.refExtracts.isEmpty {
                    Text("No extract yet — select a range and Add.")
                        .font(.caption2).foregroundStyle(.secondary)
                } else {
                    ForEach(vm.refExtracts) { e in
                        HStack {
                            Text(String(format: "%.1f – %.1f s  (%.1f s)", e.start, e.end, e.duration))
                                .font(.system(.caption, design: .monospaced))
                            Spacer()
                            Button(role: .destructive) { vm.removeExtract(e.id) } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.borderless).controlSize(.small)
                        }
                    }
                }
                ProgressView(value: min(vm.refExtractsTotal, target), total: target)
            }

            if !vm.refBuilderStatus.isEmpty {
                Text(vm.refBuilderStatus).font(.caption2.monospaced()).foregroundStyle(.secondary)
            }

            Divider()
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                if vm.refExtractsTotal > 0 && vm.refExtractsTotal < target {
                    Text("Need ~\(String(format: "%.0f", target - vm.refExtractsTotal)) s more")
                        .font(.caption2).foregroundStyle(.orange)
                }
                Button {
                    vm.buildReference()
                } label: {
                    if vm.refBuilderBusy { ProgressView().controlSize(.small) }
                    else { Text("Use as reference") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.refExtractsTotal < target || vm.refBuilderBusy)
            }
            // Dismiss once the reference has been assembled.
            .onChange(of: vm.referenceURL) { _, url in
                if url != nil && !vm.refBuilderBusy && vm.refBuilderStatus.hasPrefix("Reference ready") {
                    dismiss()
                }
            }
        }
        .padding(20)
        .frame(width: 520)
    }

    @ViewBuilder
    private func sliderRow(_ label: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.caption).frame(width: 38, alignment: .leading)
            Slider(value: value, in: range)
            Text(String(format: "%.1f s", value.wrappedValue))
                .font(.caption.monospaced()).frame(width: 52, alignment: .trailing)
        }
    }

    private func pickSource() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .video, .audio, .mpeg4Movie, .quickTimeMovie]
            + (UTType("org.webmproject.webm").map { [$0] } ?? [])
        panel.allowsOtherFileTypes = true
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            vm.loadRefSource(url)
        }
    }
}
