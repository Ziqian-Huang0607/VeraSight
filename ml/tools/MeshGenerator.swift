// MeshGenerator.swift
//
// Generates ARKit 1220-vertex face meshes from 52 blendshape coefficients plus
// the 9 Express4D rotation values (head + left/right eye), incorporating the
// rotation so the pose matches a real sensor capture.
//
// Usage (macOS host):
//   swift MeshGenerator.swift --input frames.json --output meshes.npy
//
// Input: JSON list of frames, each with:
//   {
//     "blendshapes": { "<ARKitName>": 0..1, ... },   // 52 values, ARKit names
//     "head_rotation": [yaw, pitch, roll],            // raw weights in ~[-1,1]
//     "left_eye_rotation": [yaw, pitch, roll],
//     "right_eye_rotation": [yaw, pitch, roll]
//   }
//
// Output: .npy (N, 1220, 3) float32 (C-order), ARKit face-coordinate convention.
//
// Requires a Mac with ARKit support (Apple Silicon or Intel with TrueDepth is
// not strictly required for generation: ARFaceGeometry(blendShapes:) needs a
// supported device; on macOS 13+ it works on Apple Silicon). If generation runs
// on an iPhone, wrap this in the app and stream frames out.

import Foundation
import ARKit

// MARK: - CLI args
let args = CommandLine.arguments
guard let inputFlag = args.firstIndex(of: "--input"), args.count > inputFlag + 1,
      let outputFlag = args.firstIndex(of: "--output"), args.count > outputFlag + 1 else {
    print("usage: MeshGenerator --input frames.json --output meshes.npy")
    exit(1)
}
let inputPath = args[inputFlag + 1]
let outputPath = args[outputFlag + 1]

// MARK: - Load frames
struct Frame: Codable {
    let blendshapes: [String: Double]
    let head_rotation: [Double]        // [yaw, pitch, roll]
    let left_eye_rotation: [Double]    // [yaw, pitch, roll]
    let right_eye_rotation: [Double]   // [yaw, pitch, roll]
}
let data = try Data(contentsOf: URL(fileURLWithPath: inputPath))
let frames = try JSONDecoder().decode([Frame].self, from: data)
print("loaded \(frames.count) frames")

// ARKit face geometry from blendshapes. ARFaceGeometry(blendShapes:) requires
// face tracking to be supported; on Apple Silicon Macs it works headless.
guard ARFaceTrackingConfiguration.isSupported else {
    print("ERROR: ARFaceTrackingConfiguration not supported on this host.")
    exit(1)
}

// Convert to ARFaceAnchor.BlendShapeLocation keys
func blendshapeDict(_ frame: Frame) -> [ARFaceAnchor.BlendShapeLocation: NSNumber] {
    var d: [ARFaceAnchor.BlendShapeLocation: NSNumber] = [:]
    for (name, value) in frame.blendshapes {
        // ARKit names are "com.apple.arkit.<name>"; accept both.
        let stripped = name.replacingOccurrences(of: "com.apple.arkit.", with: "")
        if let loc = ARFaceAnchor.BlendShapeLocation(rawValue: "com.apple.arkit." + stripped) {
            d[loc] = NSNumber(value: value)
        }
    }
    return d
}

// Rotation helpers (mirror Express4D's convention):
// - yaw/pitch sign-flipped
// - scale raw weight by 72 (90 * 0.8)
// - Euler order ZXY (yaw->pitch->roll as stored)
func rotationMatrix(yaw: Double, pitch: Double, roll: Double) -> simd_float3x3 {
    // Express4D: rotation = [yaw, pitch, roll] * 72, with yaw/pitch negated.
    let rx = yaw * -72.0 * .pi / 180.0
    let ry = pitch * -72.0 * .pi / 180.0
    let rz = roll * 72.0 * .pi / 180.0

    let c = cos(rx), s = sin(rx)
    let cx = simd_float3x3(
        SIMD3(1, 0, 0),
        SIMD3(0, Float(c), Float(-s)),
        SIMD3(0, Float(s), Float(c))
    )
    let cy = cos(ry), sy = sin(ry)
    let cyy = simd_float3x3(
        SIMD3(Float(cy), 0, Float(sy)),
        SIMD3(0, 1, 0),
        SIMD3(Float(-sy), 0, Float(cy))
    )
    let cz = cos(rz), sz = sin(rz)
    let czz = simd_float3x3(
        SIMD3(Float(cz), Float(-sz), 0),
        SIMD3(Float(sz), Float(cz), 0),
        SIMD3(0, 0, 1)
    )
    // ZXY: R = Rz * Rx * Ry (matches scipy 'zxy').
    return czz * cx * cyy
}

// Generate per-frame meshes.
var out = [Float](repeating: 0, count: frames.count * 1220 * 3)
let headOrigin = SIMD3<Float>(0, 3, 149)   // Express4D head origin

for (i, frame) in frames.enumerated() {
    let dict = blendshapeDict(frame)
    guard let geom = ARFaceGeometry(blendShapes: dict) else {
        print("ERROR: could not create ARFaceGeometry for frame \(i)")
        exit(1)
    }
    let verts = geom.vertices  // [simd_float3], 1220
    let headR = rotationMatrix(
        yaw: frame.head_rotation[0],
        pitch: frame.head_rotation[1],
        roll: frame.head_rotation[2]
    )
    for (j, v) in verts.enumerated() {
        // Express4D rotates only vertices with z > head_origin_z (the shell).
        // For the full ARKit face mesh, rotate all vertices around the head
        // origin; keep the same sign convention (mirror X after rotation).
        var p = v - headOrigin
        p = headR * p
        p += headOrigin
        p.x = -p.x   // Express4D mirror
        let base = (i * 1220 + j) * 3
        out[base] = p.x
        out[base + 1] = p.y
        out[base + 2] = p.z
    }
}

// Write NPY (N, 1220, 3) float32, C-order.
// Minimal .npy writer (v1.0 format, little-endian).
let n = frames.count
let shape = "\(n),1220,3"
var header = "\u{93}NUMPY\u{01}\u{00}v\u{00}\u{00}"
header += "{'descr': '<f4', 'fortran_order': False, 'shape': (\(shape),), }"
// Pad header to 64-byte alignment
let total = 10 + header.utf8.count
let pad = (64 - (total % 64)) % 64
header += String(repeating: " ", count: pad)
header += "\n"

var np = Data()
np.append(contentsOf: header.utf8)
var floats = out
let bytes = floats.withUnsafeBytes { Data($0) }
np.append(bytes)
try np.write(to: URL(fileURLWithPath: outputPath))
print("wrote \(n) meshes to \(outputPath)")
