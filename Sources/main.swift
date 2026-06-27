//
//  main.swift
//  Minimal AppKit host: opens a window with an MTKView driven by Renderer.
//

import Cocoa
import MetalKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var renderer: Renderer!

    func applicationDidFinishLaunching(_ note: Notification) {
        let rect = NSRect(x: 0, y: 0, width: 900, height: 900)
        window = NSWindow(contentRect: rect,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "Eulerian Smoke & Fire"
        window.center()

        let view = MTKView(frame: rect, device: MTLCreateSystemDefaultDevice())
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60

        guard let r = Renderer(view: view, gridN: 128) else {
            fatalError("Failed to create Metal renderer")
        }
        renderer = r
        view.delegate = r

        window.contentView = view
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()