//
//  FluidSimulator.swift
//  Owns Metal device, textures, pipelines; runs one sim step per frame and
//  raymarches the result into an MTKView drawable.
//
//  NOTE: this is an uncompiled skeleton. Wire it into an Xcode app:
//    - add Fluid3D.metal to the target
//    - create an MTKView, set view.device, attach a Renderer as delegate
//    - call Renderer in draw(in:)
//  Verify struct layouts against the .metal structs if you change fields.
//

import Metal
import MetalKit
import simd

// MARK: - Uniform structs (must match Fluid3D.metal layouts)

struct SimParams {
    var size: SIMD3<UInt32>
    var dt: Float
    var buoyancy: Float
    var weight: Float
    var ambientT: Float
    var vorticity: Float
    var dx: Float
    var damping: Float
}

struct Emitter {
    var pos: SIMD3<Float>
    var vel: SIMD3<Float>
    var radius: Float
    var density: Float
    var temperature: Float
    var pad: Float = 0
}

struct CameraUniforms {
    var invViewProj: float4x4
    var camPos: SIMD3<Float>
    var boxMin: SIMD3<Float>
    var boxMax: SIMD3<Float>
    var lightDir: SIMD3<Float>
    var lightColor: SIMD3<Float>
    var absorption: Float
    var scatter: Float
    var emission: Float
    var tempScale: Float
    var steps: Int32
    var shadowSteps: Int32
    var stepSize: Float
    var pad: Float = 0
}

// MARK: - Double-buffered 3D field

final class Field {
    var tex: [MTLTexture]
    var cur = 0
    var read:  MTLTexture { tex[cur] }
    var write: MTLTexture { tex[1 - cur] }
    func swap() { cur = 1 - cur }

    init(_ device: MTLDevice, _ n: Int, _ format: MTLPixelFormat, count: Int = 2) {
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = format
        d.width = n; d.height = n; d.depth = n
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        tex = (0..<count).map { _ in device.makeTexture(descriptor: d)! }
    }
}

// MARK: - Simulator

final class FluidSimulator {
    let device: MTLDevice
    let n: Int

    let velocity: Field      // rgba16Float
    let density: Field       // r16Float
    let temperature: Field   // r16Float
    let pressure: Field      // r16Float
    let divergence: Field    // r16Float (single)
    let vorticity: Field     // rgba16Float (single)

    private let pso: [String: MTLComputePipelineState]
    private var params: SimParams
    private var emitter: Emitter

    private let paramsBuf: MTLBuffer
    private let emitterBuf: MTLBuffer
    private let densDissBuf: MTLBuffer
    private let tempDissBuf: MTLBuffer

    let pressureIters = 40

    init(device: MTLDevice, library: MTLLibrary, n: Int = 128) {
        self.device = device
        self.n = n
        velocity    = Field(device, n, .rgba16Float)
        density     = Field(device, n, .r16Float)
        temperature = Field(device, n, .r16Float)
        pressure    = Field(device, n, .r16Float)
        divergence  = Field(device, n, .r16Float, count: 1)
        vorticity   = Field(device, n, .rgba16Float, count: 1)

        let names = ["clearTex", "inject", "buoyancy", "curl", "vorticityForce",
                     "advectVelocity", "advectScalar",
                     "divergence", "jacobi", "project"]
        var table = [String: MTLComputePipelineState]()
        for nm in names {
            let fn = library.makeFunction(name: nm)!
            table[nm] = try! device.makeComputePipelineState(function: fn)
        }
        pso = table

        paramsBuf   = device.makeBuffer(length: MemoryLayout<SimParams>.stride, options: .storageModeShared)!
        emitterBuf  = device.makeBuffer(length: MemoryLayout<Emitter>.stride,   options: .storageModeShared)!
        densDissBuf = device.makeBuffer(length: MemoryLayout<Float>.stride,     options: .storageModeShared)!
        tempDissBuf = device.makeBuffer(length: MemoryLayout<Float>.stride,     options: .storageModeShared)!
        densDissBuf.contents().storeBytes(of: 0.995, as: Float.self)
        tempDissBuf.contents().storeBytes(of: 0.95, as: Float.self)

        params = SimParams(size: SIMD3(UInt32(n), UInt32(n), UInt32(n)),
                           dt: 0.10, buoyancy: 1.0, weight: 0.05,
                           ambientT: 0.0, vorticity: 0.8, dx: 1.0, damping: 0.97)

        // Bottom-center emitter, blowing straight up.
        emitter = Emitter(pos: SIMD3(0.5, 0.08, 0.5),
                          vel: SIMD3(0, 9, 0),
                          radius: 0.06, density: 28, temperature: 22)
    }

    // Dispatch helper.
    private func grid() -> (MTLSize, MTLSize) {
        let tg = MTLSize(width: 8, height: 8, depth: 8)
        let g  = MTLSize(width:  (n + 7) / 8,
                         height: (n + 7) / 8,
                         depth:  (n + 7) / 8)
        return (g, tg)
    }

    private func encode(_ enc: MTLComputeCommandEncoder,
                        _ name: String,
                        textures: [MTLTexture?],
                        buffers: [(MTLBuffer, Int)] = []) {
        enc.setComputePipelineState(pso[name]!)
        for (i, t) in textures.enumerated() { enc.setTexture(t, index: i) }
        for (buf, idx) in buffers { enc.setBuffer(buf, offset: 0, index: idx) }
        let (g, tg) = grid()
        enc.dispatchThreadgroups(g, threadsPerThreadgroup: tg)
    }

    /// Zero every field. Call once before the first step — private textures
    /// start with undefined contents.
    func clearAll(_ cb: MTLCommandBuffer) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso["clearTex"]!)
        let (g, tg) = grid()
        let all = velocity.tex + density.tex + temperature.tex
                + pressure.tex + divergence.tex + vorticity.tex
        for t in all {
            enc.setTexture(t, index: 0)
            enc.dispatchThreadgroups(g, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    private func updateUniforms() {
        withUnsafeBytes(of: params)  { paramsBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        withUnsafeBytes(of: emitter) { emitterBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    }

    /// Advance one step. dt/emitter can be animated by mutating params/emitter.
    func step(_ cb: MTLCommandBuffer) {
        let enc = cb.makeComputeCommandEncoder()!
        updateUniforms()
        let pb = (paramsBuf, 0)

        // 1. inject density/temp/velocity
        encode(enc, "inject",
               textures: [density.read, temperature.read, velocity.read,
                          density.write, temperature.write, velocity.write],
               buffers: [pb, (emitterBuf, 1)])
        density.swap(); temperature.swap(); velocity.swap()

        // 2. buoyancy
        encode(enc, "buoyancy",
               textures: [velocity.read, density.read, temperature.read, velocity.write],
               buffers: [pb])
        velocity.swap()

        // 3. vorticity confinement
        encode(enc, "curl",
               textures: [velocity.read, vorticity.read], buffers: [pb])
        encode(enc, "vorticityForce",
               textures: [velocity.read, vorticity.read, velocity.write], buffers: [pb])
        velocity.swap()

        // 4. advect velocity (self)
        encode(enc, "advectVelocity",
               textures: [velocity.read, velocity.write], buffers: [pb])
        velocity.swap()

        // 5. projection: divergence -> jacobi xN (warm-started) -> subtract grad
        encode(enc, "divergence",
               textures: [velocity.read, divergence.read], buffers: [pb])
        for _ in 0..<pressureIters {
            encode(enc, "jacobi",
                   textures: [pressure.read, divergence.read, pressure.write],
                   buffers: [pb])
            pressure.swap()
        }
        encode(enc, "project",
               textures: [pressure.read, velocity.read, velocity.write], buffers: [pb])
        velocity.swap()

        // 6. advect density (dissipates) and temperature (cools)
        encode(enc, "advectScalar",
               textures: [velocity.read, density.read, density.write],
               buffers: [pb, (densDissBuf, 1)])
        density.swap()
        encode(enc, "advectScalar",
               textures: [velocity.read, temperature.read, temperature.write],
               buffers: [pb, (tempDissBuf, 1)])
        temperature.swap()

        enc.endEncoding()
    }
}

// MARK: - Renderer (MTKViewDelegate)

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let sim: FluidSimulator
    let renderPSO: MTLRenderPipelineState

    var aspect: Float = 1
    var time: Float = 0
    var cam = CameraUniforms(invViewProj: matrix_identity_float4x4,
                             camPos: .zero, boxMin: SIMD3(repeating: -0.5),
                             boxMax: SIMD3(repeating: 0.5),
                             lightDir: normalize(SIMD3(0.4, 1.0, 0.3)),
                             lightColor: SIMD3(1.0, 0.95, 0.9),
                             absorption: 4.0, scatter: 1.4,
                             emission: 3.5, tempScale: 0.05,
                             steps: 256, shadowSteps: 16, stepSize: 0.006)

    init?(view: MTKView, gridN: Int = 128) {
        guard let dev = view.device ?? MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue(),
              let lib = dev.makeDefaultLibrary() else { return nil }
        device = dev; queue = q
        sim = FluidSimulator(device: dev, library: lib, n: gridN)

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction   = lib.makeFunction(name: "raymarch_vertex")
        desc.fragmentFunction = lib.makeFunction(name: "raymarch_fragment")
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        renderPSO = try! dev.makeRenderPipelineState(descriptor: desc)
        super.init()

        // Private textures are not zero-initialized — clear all fields once
        // before the first frame, otherwise the solver reads garbage and blows
        // up to NaN (black screen).
        if let cb = queue.makeCommandBuffer() {
            sim.clearAll(cb)
            cb.commit()
            cb.waitUntilCompleted()
        }
    }

    func mtkView(_ v: MTKView, drawableSizeWillChange s: CGSize) {
        aspect = Float(s.width / max(s.height, 1))
    }

    private func updateCamera() {
        // Orbit camera around the box.
        let r: Float = 1.6
        let a = time * 0.3
        let eye = SIMD3(cos(a) * r, 0.35, sin(a) * r)
        let view = lookAt(eye: eye, center: .zero, up: SIMD3(0,1,0))
        let proj = perspective(fovY: 0.9, aspect: aspect, near: 0.05, far: 10)
        cam.invViewProj = (proj * view).inverse
        cam.camPos = eye
    }

    func draw(in view: MTKView) {
        guard let cb = queue.makeCommandBuffer() else { return }
        time += 1.0 / 60.0
        updateCamera()

        sim.step(cb)                                   // compute pass

        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else {
            cb.commit(); return
        }
        enc.setRenderPipelineState(renderPSO)
        enc.setFragmentTexture(sim.density.read, index: 0)
        enc.setFragmentTexture(sim.temperature.read, index: 1)
        withUnsafeBytes(of: cam) {
            enc.setFragmentBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}

// MARK: - Tiny matrix helpers

func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
    let f = normalize(center - eye)
    let s = normalize(cross(f, up))
    let u = cross(s, f)
    return float4x4(columns: (
        SIMD4(s.x, u.x, -f.x, 0),
        SIMD4(s.y, u.y, -f.y, 0),
        SIMD4(s.z, u.z, -f.z, 0),
        SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)
    ))
}

func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
    let y = 1 / tan(fovY * 0.5)
    let x = y / aspect
    let z = far / (near - far)
    return float4x4(columns: (
        SIMD4(x, 0, 0, 0),
        SIMD4(0, y, 0, 0),
        SIMD4(0, 0, z, -1),
        SIMD4(0, 0, z * near, 0)
    ))
}
