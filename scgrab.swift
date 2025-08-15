import Foundation
import AVFoundation
import ScreenCaptureKit
import Darwin

// ------- Keep these alive for the lifetime of the process -------
fileprivate var gStream: SCStream?
fileprivate var gOutput: Output?
fileprivate var gRing:   Ring?

// ------- CLI options -------
struct Opts {
    var x = 0, y = 0, w = 640, h = 480, fps = 60, slots = 3
    var out = "/tmp/scap.ring"
}
func parseArgs() -> Opts {
    var o = Opts()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        switch k {
        case "--x":     o.x = Int(it.next()!)!
        case "--y":     o.y = Int(it.next()!)!
        case "--w":     o.w = Int(it.next()!)!
        case "--h":     o.h = Int(it.next()!)!
        case "--fps":   o.fps = Int(it.next()!)!
        case "--out":   o.out = it.next()!
        case "--slots": o.slots = Int(it.next()!)!
        default: break
        }
    }
    return o
}

// ------- Shared-memory ring (packed, little-endian) -------
struct GlobalHeader {
    var magic:   UInt32 = 0x534E5247 // 'GRNS'
    var version: UInt32 = 1
    var slots:   UInt32
    var pixfmt:  UInt32   // 0 = NV12
    var width:   UInt32
    var height:  UInt32
    var strideY: UInt32   // packed to width
    var strideUV:UInt32   // packed to width
    var slotSize:UInt64   // bytes of Y + UV
    var headSeq: UInt64   // latest written seq
}
struct SlotHeader {
    var seqStart: UInt64 = 0
    var tNanos:   UInt64 = 0  // DispatchTime.now().uptimeNanoseconds
    var ready:    UInt32 = 0
    var _pad:     UInt32 = 0
    var seqEnd:   UInt64 = 0
}

final class Ring {
    let fd: Int32
    let ptr: UnsafeMutableRawPointer
    let totalSize: Int
    let slots: Int
    let slotSize: Int
    let hdrSize = MemoryLayout<GlobalHeader>.size
    let slotHdrSize = MemoryLayout<SlotHeader>.size
    var headSeq: UInt64 = 0
    let w: Int, h: Int

    init(path: String, slots: Int, w: Int, h: Int) {
        self.slots = slots; self.w = w; self.h = h
        self.slotSize = w*h + w*(h/2) // NV12 packed (Y then UV)
        let total = hdrSize + slots*(slotHdrSize + slotSize)
        self.totalSize = total

        fd = open(path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        ftruncate(fd, off_t(total))

        // ---- Correct mmap check on Darwin ----
        let p = mmap(nil, total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)
        guard p != MAP_FAILED else {
            perror("mmap")
            fatalError("mmap failed")
        }
        ptr = p!

        var gh = GlobalHeader(
            slots: UInt32(slots), pixfmt: 0,
            width: UInt32(w), height: UInt32(h),
            strideY: UInt32(w), strideUV: UInt32(w),
            slotSize: UInt64(slotSize), headSeq: 0
        )
        memcpy(ptr, &gh, hdrSize)
    }

    func offsets(_ i:Int) -> (UnsafeMutableRawPointer, UnsafeMutableRawPointer) {
        let base = hdrSize + i*(slotHdrSize + slotSize)
        return (ptr.advanced(by: base), ptr.advanced(by: base + slotHdrSize))
    }

    func writeNV12(yPlane: UnsafeRawPointer, yStride: Int,
                   uvPlane: UnsafeRawPointer, uvStride: Int) {
        let seq = headSeq &+ 1
        let i = Int(seq % UInt64(slots))
        let (hdrPtr, dataPtr) = offsets(i)

        var sh = SlotHeader(seqStart: seq,
                            tNanos: DispatchTime.now().uptimeNanoseconds,
                            ready: 0, _pad: 0, seqEnd: 0)
        memcpy(hdrPtr, &sh, slotHdrSize)

        // Pack to contiguous width bytes per row (ignore stride padding)
        var dst = dataPtr
        for row in 0..<h {
            memcpy(dst, yPlane.advanced(by: row*yStride), w)
            dst = dst.advanced(by: w)
        }
        for row in 0..<(h/2) {
            memcpy(dst, uvPlane.advanced(by: row*uvStride), w)
            dst = dst.advanced(by: w)
        }

        sh.ready = 1; sh.seqEnd = seq
        memcpy(hdrPtr, &sh, slotHdrSize)

        var gh = GlobalHeader(
            slots: UInt32(slots), pixfmt: 0,
            width: UInt32(w), height: UInt32(h),
            strideY: UInt32(w), strideUV: UInt32(w),
            slotSize: UInt64(slotSize), headSeq: seq
        )
        memcpy(ptr, &gh, hdrSize)
        headSeq = seq
    }

    deinit { munmap(ptr, totalSize); close(fd) }
}

// ------- ScreenCaptureKit output callback -------
final class Output: NSObject, SCStreamOutput {
    let ring: Ring
    let w: Int, h: Int
    private var frames = 0
    private var lastReport = DispatchTime.now().uptimeNanoseconds

    init(ring: Ring, w: Int, h: Int) { self.ring = ring; self.w = w; self.h = h }

    func stream(_ stream: SCStream,
                didOutputSampleBuffer sb: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .screen,
              let imgBuf = CMSampleBufferGetImageBuffer(sb) else { return }

        let pb: CVPixelBuffer = imgBuf
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!
        let uvBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1)!
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
        ring.writeNV12(yPlane: yBase, yStride: yStride, uvPlane: uvBase, uvStride: uvStride)
        CVPixelBufferUnlockBaseAddress(pb, .readOnly)

        // 1-sec heartbeat (helps debug when "nothing happens")
        frames += 1
        let now = DispatchTime.now().uptimeNanoseconds
        if now - lastReport > 1_000_000_000 {
            print("[producer] wrote \(frames) frames in the last second")
            frames = 0
            lastReport = now
        }
    }
}

// ------- App entry point (non-async main + Task for async setup) -------
@main
struct App {
    static func main() {
        let o = parseArgs()

        Task {
            do {
                // Pick first display
                let content = try await SCShareableContent.current
                guard let display = content.displays.first else { fatalError("No display found") }

                // Configure stream: NV12 + exact ROI
                let cfg = SCStreamConfiguration()
                cfg.width  = o.w
                cfg.height = o.h
                cfg.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(o.fps))
                cfg.pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange // NV12
                cfg.showsCursor = false
                cfg.capturesAudio = false
                cfg.sourceRect = CGRect(x: o.x, y: o.y, width: o.w, height: o.h)

                // Whole-display filter
                let filter = SCContentFilter(display: display,
                                             excludingApplications: [],
                                             exceptingWindows: [])

                // Stream + output
                let stream = SCStream(filter: filter, configuration: cfg, delegate: nil)
                let ring   = Ring(path: o.out, slots: o.slots, w: o.w, h: o.h)
                let output = Output(ring: ring, w: o.w, h: o.h)

                // Use a dedicated queue (avoid main-queue stalls)
                let outQ = DispatchQueue(label: "sc.output.q")
                try stream.addStreamOutput(output, type: .screen, sampleHandlerQueue: outQ)

                try await stream.startCapture()
                print("SCStream running: ROI \(o.w)x\(o.h) @\(o.fps) → \(o.out) (slots \(o.slots))")

                // Retain objects globally so they don't deallocate
                gRing   = ring
                gOutput = output
                gStream = stream

            } catch {
                fputs("Error: \(error)\n", stderr)
                exit(1)
            }
        }

        // Keep the process alive
        dispatchMain()
    }
}
