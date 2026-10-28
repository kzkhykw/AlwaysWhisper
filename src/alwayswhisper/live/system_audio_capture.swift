// Local ScreenCaptureKit bridge. stdout: AUD0 + uint32 frames + float64 host
// time (all little-endian), followed by mono Float32 PCM at 48 kHz.
// No audio files, network requests, or playback-device changes.
import Foundation
import ScreenCaptureKit
import CoreMedia
import CoreAudio
import CoreGraphics

func status(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

func packet(_ pcm: Data, frames: Int, time: Double) -> Data {
    var data = Data("AUD0".utf8)
    var count = UInt32(frames).littleEndian
    var timestamp = time.bitPattern.littleEndian
    withUnsafeBytes(of: &count) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: &timestamp) { data.append(contentsOf: $0) }
    data.append(pcm)
    return data
}

@available(macOS 13.0, *)
@MainActor
final class AudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    var capture: SCStream?
    let queue = DispatchQueue(label: "alwayswhisper.system-audio")

    func start(appIDs: [String]) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            throw NSError(domain: "AlwaysWhisper", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No display is available for system audio capture."])
        }
        let filter: SCContentFilter
        if appIDs.isEmpty {
            filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        } else {
            let apps = content.applications.filter { appIDs.contains($0.bundleIdentifier) }
            let missing = Set(appIDs).subtracting(apps.map { $0.bundleIdentifier })
            guard missing.isEmpty else {
                throw NSError(domain: "AlwaysWhisper", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Start the requested audio app(s): \(missing.sorted().joined(separator: ", "))"])
            }
            filter = SCContentFilter(display: display, including: apps, exceptingWindows: [])
        }
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 48000
        config.channelCount = 1
        // This helper never plays audio. Include the launching application's
        // playback too (macOS can attribute helpers to their responsible app).
        config.excludesCurrentProcessAudio = false
        // ScreenCaptureKit requires a display filter. Screen frames are tiny
        // and discarded; only audio leaves this helper.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.queueDepth = 3
        config.showsCursor = false
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        capture = stream
        try await stream.startCapture()
        status("READY")
    }

    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        status("ERROR: \(error.localizedDescription)")
        exit(1)
    }

    nonisolated func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of outputType: SCStreamOutputType) {
        guard outputType == .audio, sampleBuffer.isValid,
              let format = sampleBuffer.formatDescription,
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format) else { return }
        let frames = sampleBuffer.numSamples
        guard frames > 0 else { return }
        guard asbd.pointee.mFormatID == kAudioFormatLinearPCM,
              asbd.pointee.mSampleRate == 48000,
              asbd.pointee.mChannelsPerFrame == 1,
              asbd.pointee.mBitsPerChannel == 32,
              asbd.pointee.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              asbd.pointee.mFormatFlags & kAudioFormatFlagIsBigEndian == 0,
              frames <= 48000 else {
            status("ERROR: Unexpected system audio PCM format.")
            exit(1)
        }
        do {
            try sampleBuffer.withAudioBufferList { buffers, _ in
                guard buffers.count == 1, let bytes = buffers[0].mData,
                      Int(buffers[0].mDataByteSize) >= frames * 4 else {
                    throw NSError(domain: "AlwaysWhisper", code: 3,
                                  userInfo: [NSLocalizedDescriptionKey: "Invalid system audio buffer."])
                }
                // Map the sample PTS onto the same host uptime clock Python's
                // time.monotonic uses on macOS, retaining the capture instant.
                let age = CMClockGetTime(CMClockGetHostTimeClock()).seconds - sampleBuffer.presentationTimeStamp.seconds
                let timestamp = ProcessInfo.processInfo.systemUptime - age
                guard timestamp.isFinite else { return }
                let data = packet(Data(bytes: bytes, count: frames * 4), frames: frames, time: timestamp)
                try FileHandle.standardOutput.write(contentsOf: data)
            }
        } catch {
            status("ERROR: \(error.localizedDescription)")
            exit(1)
        }
    }

    func stop() async {
        try? await capture?.stopCapture()
        exit(0)
    }
}

@main
struct Main {
    @MainActor
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        if args == ["--check-permission"] {
            print(CGPreflightScreenCaptureAccess() ? "granted" : "not-granted")
            return
        }
        if args == ["--self-test"] {
            let samples: [Float] = [0.25, -0.5, 0, 1]
            samples.withUnsafeBytes {
                FileHandle.standardOutput.write(packet(Data($0), frames: samples.count, time: 123.5))
            }
            return
        }
        guard #available(macOS 13.0, *) else {
            status("ERROR: System audio capture requires macOS 13 or later.")
            exit(1)
        }
        let recorder = AudioCapture()
        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
            let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
            source.setEventHandler { Task { await recorder.stop() } }
            source.resume()
            return source
        }
        defer { withExtendedLifetime(signals) {} }
        // EOF also stops recording if the Python parent disappears.
        DispatchQueue.global().async {
            while !FileHandle.standardInput.readData(ofLength: 1).isEmpty {}
            Task { await recorder.stop() }
        }
        do {
            try await recorder.start(appIDs: args)
            while true { try await Task.sleep(nanoseconds: 1_000_000_000) }
        } catch {
            status("ERROR: \(error.localizedDescription). Allow Screen & System Audio Recording in macOS Privacy & Security, then restart transcribe.")
            exit(1)
        }
    }
}
