# Eulerian Smoke & Fire (Metal)

Real-time 3D fluid solver and volumetric renderer written from scratch in
**Swift + Metal**, with **no external dependencies** — no SceneKit, no third-party
math or physics libraries, only the platform `Metal`, `MetalKit`, and `simd`.

The simulation is a grid-based ("Eulerian") solver following Jos Stam's
*Stable Fluids*, extended with vorticity confinement, temperature-driven
buoyancy, and a single-scattering volumetric raymarcher with blackbody emission
for fire.

> Replace this line with a GIF/screenshot of the running simulation.

---

## What it does

A hot, buoyant plume is injected at the bottom of a cubic domain and rises,
rolls, and dissipates. Temperature lifts the fluid and drives blackbody
emission (the flame); smoke density absorbs light and casts soft self-shadows.
Everything is computed on the GPU every frame and raymarched directly from a 3D
texture — there is no mesh and no offline bake.

## Technical highlights

- **Pure GPU compute.** The entire solve runs in Metal compute kernels over 3D
  textures. The CPU only orchestrates passes and updates uniforms.
- **Semi-Lagrangian advection** via hardware trilinear sampling of 3D textures —
  the backtrace lookup is a single `sample()` call instead of a hand-rolled
  trilinear interpolation.
- **Vorticity confinement** restores the small-scale curling that numerical
  diffusion otherwise kills, giving the plume its characteristic licks.
- **Pressure projection** through a Jacobi Poisson solver with Neumann
  boundaries (clamp-to-edge reads), warm-started across frames.
- **Volumetric rendering** by ray-box marching with front-to-back emission–
  absorption compositing, blackbody color ramp from temperature, and a short
  shadow march toward the light for self-shadowing.
- **Gather-only, atomic-free.** Semi-Lagrangian advection reads the old field
  and writes the new one, so the solver maps cleanly to the GPU with no scatter
  and no atomics. All fields are double-buffered (ping-pong), which also keeps
  the half-float texture formats portable.

## Method

Each step, in order:

1. **Inject** density, temperature, and velocity at the emitter.
2. **Buoyancy** — add `(buoyancy·(T − Tₐ) − weight·ρ)·ŷ` to the velocity.
3. **Vorticity confinement** — compute curl `ω`, then add `ε·(N × ω)` where
   `N` is the normalized gradient of `|ω|`.
4. **Advect velocity** (self-advection, semi-Lagrangian).
5. **Project** — compute divergence, solve `∇²p = ∇·u` (Jacobi), subtract `∇p`
   to make the field divergence-free.
6. **Advect** density and temperature with the divergence-free velocity, with
   per-field dissipation/cooling.

The result is rendered by casting one ray per pixel, intersecting the domain
AABB, and marching the density/temperature textures with emission–absorption
compositing.

The grid is **collocated** (all quantities at cell centers) and works in cell
units (`dx = 1`), which keeps the kernels compact. See *Roadmap* for the
staggered-grid upgrade.

## Requirements

- macOS with a Metal-capable GPU (Apple Silicon recommended)
- Xcode 15+

## Build & run

This repository ships the solver and renderer; wire them into a minimal app:

1. Create a new macOS (or iOS) app target in Xcode.
2. Add `Fluid3D.metal` and `FluidSimulator.swift` to the target.
3. Add an `MTKView` to your window, set its `device`, and attach a `Renderer`
   as its delegate:

   ```swift
   let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
   view.colorPixelFormat = .bgra8Unorm
   let renderer = Renderer(view: view, gridN: 128)
   view.delegate = renderer
   ```

4. Build and run.

## Tuning

The look is driven by a handful of parameters in `FluidSimulator` (sim) and
`Renderer` (render):

| Parameter        | Effect                                              |
|------------------|-----------------------------------------------------|
| `buoyancy`       | Upward thermal lift                                 |
| `weight`         | How much smoke density sinks                        |
| `vorticity`      | Strength of small-scale curling (raise for flames)  |
| `dt`             | Time step — keep `velocity·dt < 1` cell for stability |
| density/temp dissipation | How fast smoke fades and the plume cools    |
| `absorption`     | Smoke opacity                                       |
| `emission` / `tempScale` | Flame brightness and temperature-to-color mapping |
| `steps` / `shadowSteps`  | Raymarch and self-shadow quality            |

## Performance

`128³` runs comfortably in real time on Apple Silicon. `256³` is feasible, but
the Jacobi pressure solve becomes the bottleneck — reduce the iteration count or
switch to a red–black Gauss–Seidel / multigrid solver (see *Roadmap*).

## Project structure

```
Fluid3D.metal         Compute kernels (sim) + raymarch vertex/fragment (render)
FluidSimulator.swift  Field double-buffering, pipeline setup, per-frame step,
                      MTKView renderer, camera/matrix helpers
```

## Roadmap

- **Combustion model** — convert fuel to heat and smoke above an ignition
  threshold for true fire rather than merely hot smoke.
- **MAC (staggered) grid** for a more accurate projection and less numerical
  diffusion.
- **Faster pressure solve** — red–black Gauss–Seidel or a multigrid V-cycle.
- **Closed-box boundary kernel** with proper no-slip / free-slip walls.
- **Temporal reprojection** to denoise the shadow march.

## License

MIT — see `LICENSE`.