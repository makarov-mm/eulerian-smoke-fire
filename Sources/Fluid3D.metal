//
//  Fluid3D.metal
//  3D Eulerian smoke/fire — Stam "Stable Fluids" + vorticity confinement,
//  Jacobi pressure projection, volumetric raymarch with blackbody emission.
//
//  Conventions:
//   - Collocated grid, all quantities at cell centers.
//   - Work in CELL units: dx = 1, velocity in cells/sec, dt in sec.
//   - Everything ping-pongs (read src -> write dst). No access::read_write,
//     so half-float formats (r16f / rgba16f) stay portable.
//

#include <metal_stdlib>
using namespace metal;

constant float3 kUp = float3(0.0, 1.0, 0.0);

struct SimParams {
    uint3  size;        // grid dims (Nx,Ny,Nz)
    float  dt;
    float  buoyancy;    // temperature buoyancy coeff
    float  weight;      // smoke weight (density pulls down)
    float  ambientT;    // ambient temperature
    float  vorticity;   // confinement epsilon
    float  dx;          // cell size (1.0)
    float  damping;     // velocity drag per step (kills runaway mean flow)
};

struct Emitter {
    float3 pos;         // emitter center in [0,1] domain coords
    float3 vel;         // injected velocity (cells/sec)
    float  radius;      // in [0,1] of Nx
    float  density;
    float  temperature;
    float  pad;
};

// ---- helpers ---------------------------------------------------------------

static inline float3 readVel(texture3d<float, access::read> v, int3 c, uint3 n) {
    c = clamp(c, int3(0), int3(n) - 1);
    return v.read(uint3(c)).xyz;
}

static inline float readScalar(texture3d<float, access::read> s, int3 c, uint3 n) {
    c = clamp(c, int3(0), int3(n) - 1);
    return s.read(uint3(c)).x;
}

// ---- clear (private textures are NOT zero-initialized) ---------------------

kernel void clearTex(texture3d<float, access::write> t [[texture(0)]],
                     uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= t.get_width() || gid.y >= t.get_height() || gid.z >= t.get_depth())
        return;
    t.write(float4(0.0), gid);
}

// ---- sources / injection ---------------------------------------------------

kernel void inject(texture3d<float, access::read>  densIn  [[texture(0)]],
                   texture3d<float, access::read>  tempIn  [[texture(1)]],
                   texture3d<float, access::read>  velIn   [[texture(2)]],
                   texture3d<float, access::write> densOut [[texture(3)]],
                   texture3d<float, access::write> tempOut [[texture(4)]],
                   texture3d<float, access::write> velOut  [[texture(5)]],
                   constant SimParams& P [[buffer(0)]],
                   constant Emitter&   E [[buffer(1)]],
                   uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;

    float  d = densIn.read(gid).x;
    float  t = tempIn.read(gid).x;
    float3 v = velIn.read(gid).xyz;

    float3 p = float3(gid) + 0.5;
    float3 c = E.pos * float3(P.size);
    float  r = E.radius * float(P.size.x);
    float  fall = smoothstep(r, 0.0, length(p - c));   // 1 at center -> 0 at edge

    d += E.density     * fall * P.dt;
    t += E.temperature * fall * P.dt;
    v += E.vel         * fall * P.dt;

    densOut.write(float4(d), gid);
    tempOut.write(float4(t), gid);
    velOut.write(float4(v, 0.0), gid);
}

// ---- buoyancy (temperature lifts, smoke sinks) -----------------------------

kernel void buoyancy(texture3d<float, access::read>  velIn   [[texture(0)]],
                     texture3d<float, access::read>  density [[texture(1)]],
                     texture3d<float, access::read>  temp    [[texture(2)]],
                     texture3d<float, access::write> velOut  [[texture(3)]],
                     constant SimParams& P [[buffer(0)]],
                     uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;
    float3 v = velIn.read(gid).xyz * P.damping;   // drag removes runaway mean flow
    float  d = density.read(gid).x;
    float  t = temp.read(gid).x;
    float3 f = (P.buoyancy * (t - P.ambientT) - P.weight * d) * kUp;
    velOut.write(float4(v + P.dt * f, 0.0), gid);
}

// ---- vorticity confinement -------------------------------------------------

kernel void curl(texture3d<float, access::read>  vel [[texture(0)]],
                 texture3d<float, access::write> out [[texture(1)]], // xyz=curl, w=|curl|
                 constant SimParams& P [[buffer(0)]],
                 uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;
    int3 c = int3(gid);
    float3 xp = readVel(vel, c + int3(1,0,0), P.size);
    float3 xm = readVel(vel, c - int3(1,0,0), P.size);
    float3 yp = readVel(vel, c + int3(0,1,0), P.size);
    float3 ym = readVel(vel, c - int3(0,1,0), P.size);
    float3 zp = readVel(vel, c + int3(0,0,1), P.size);
    float3 zm = readVel(vel, c - int3(0,0,1), P.size);

    float3 w;
    w.x = (yp.z - ym.z) - (zp.y - zm.y);
    w.y = (zp.x - zm.x) - (xp.z - xm.z);
    w.z = (xp.y - xm.y) - (yp.x - ym.x);
    w *= 0.5;                                  // central diff / (2 dx)
    out.write(float4(w, length(w)), gid);
}

kernel void vorticityForce(texture3d<float, access::read>  velIn  [[texture(0)]],
                           texture3d<float, access::read>  w      [[texture(1)]],
                           texture3d<float, access::write> velOut [[texture(2)]],
                           constant SimParams& P [[buffer(0)]],
                           uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;
    int3 c = int3(gid);
    float mxp = readScalar(w, c + int3(1,0,0), P.size); // |curl| stored in .x? no -> .w
    // |curl| lives in .w; reuse readScalar on a single-channel view won't work, read .w manually:
    float wmxp = w.read(uint3(clamp(c + int3(1,0,0), int3(0), int3(P.size)-1))).w;
    float wmxm = w.read(uint3(clamp(c - int3(1,0,0), int3(0), int3(P.size)-1))).w;
    float wmyp = w.read(uint3(clamp(c + int3(0,1,0), int3(0), int3(P.size)-1))).w;
    float wmym = w.read(uint3(clamp(c - int3(0,1,0), int3(0), int3(P.size)-1))).w;
    float wmzp = w.read(uint3(clamp(c + int3(0,0,1), int3(0), int3(P.size)-1))).w;
    float wmzm = w.read(uint3(clamp(c - int3(0,0,1), int3(0), int3(P.size)-1))).w;
    (void)mxp;

    float3 grad = 0.5 * float3(wmxp - wmxm, wmyp - wmym, wmzp - wmzm);
    float3 N = grad / (length(grad) + 1e-5);
    float3 omega = w.read(gid).xyz;
    float3 force = P.vorticity * cross(N, omega);      // * dx (=1)

    float3 v = velIn.read(gid).xyz + P.dt * force;
    velOut.write(float4(v, 0.0), gid);
}

// ---- semi-Lagrangian advection (gather, GPU-friendly) ----------------------

kernel void advectVelocity(texture3d<float, access::sample> vel    [[texture(0)]],
                           texture3d<float, access::write>  velOut [[texture(1)]],
                           constant SimParams& P [[buffer(0)]],
                           uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;
    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
    float3 n   = float3(P.size);
    float3 pos = float3(gid) + 0.5;
    float3 v   = vel.sample(smp, pos / n).xyz;
    float3 back = pos - P.dt * v;
    float3 vb  = vel.sample(smp, back / n).xyz;
    velOut.write(float4(vb, 0.0), gid);
}

kernel void advectScalar(texture3d<float, access::sample> vel    [[texture(0)]],
                         texture3d<float, access::sample> src    [[texture(1)]],
                         texture3d<float, access::write>  dst    [[texture(2)]],
                         constant SimParams& P  [[buffer(0)]],
                         constant float&     dissipation [[buffer(1)]],
                         uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;
    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
    float3 n   = float3(P.size);
    float3 pos = float3(gid) + 0.5;
    float3 v   = vel.sample(smp, pos / n).xyz;
    float3 back = pos - P.dt * v;
    float  s   = src.sample(smp, back / n).x;
    dst.write(float4(s * dissipation), gid);
}

// ---- pressure projection ---------------------------------------------------

kernel void divergence(texture3d<float, access::read>  vel [[texture(0)]],
                       texture3d<float, access::write> div [[texture(1)]],
                       constant SimParams& P [[buffer(0)]],
                       uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;
    int3 c = int3(gid);
    float xp = readVel(vel, c + int3(1,0,0), P.size).x;
    float xm = readVel(vel, c - int3(1,0,0), P.size).x;
    float yp = readVel(vel, c + int3(0,1,0), P.size).y;
    float ym = readVel(vel, c - int3(0,1,0), P.size).y;
    float zp = readVel(vel, c + int3(0,0,1), P.size).z;
    float zm = readVel(vel, c - int3(0,0,1), P.size).z;
    float d = 0.5 * ((xp - xm) + (yp - ym) + (zp - zm));
    div.write(float4(d), gid);
}

// Solve  Laplacian(p) = div.  Discrete (dx=1): sum_neighbors - 6p = div
// => p = (sum_neighbors - div) / 6.  Clamped reads => Neumann BC.
kernel void jacobi(texture3d<float, access::read>  x    [[texture(0)]], // pressure in
                   texture3d<float, access::read>  b    [[texture(1)]], // divergence
                   texture3d<float, access::write> xOut [[texture(2)]],
                   constant SimParams& P [[buffer(0)]],
                   uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;
    int3 c = int3(gid);
    float pxp = readScalar(x, c + int3(1,0,0), P.size);
    float pxm = readScalar(x, c - int3(1,0,0), P.size);
    float pyp = readScalar(x, c + int3(0,1,0), P.size);
    float pym = readScalar(x, c - int3(0,1,0), P.size);
    float pzp = readScalar(x, c + int3(0,0,1), P.size);
    float pzm = readScalar(x, c - int3(0,0,1), P.size);
    float div = b.read(gid).x;
    float p = (pxp + pxm + pyp + pym + pzp + pzm - div) / 6.0;
    xOut.write(float4(p), gid);
}

kernel void project(texture3d<float, access::read>  p      [[texture(0)]],
                    texture3d<float, access::read>  velIn  [[texture(1)]],
                    texture3d<float, access::write> velOut [[texture(2)]],
                    constant SimParams& P [[buffer(0)]],
                    uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= P.size)) return;
    int3 c = int3(gid);
    float pxp = readScalar(p, c + int3(1,0,0), P.size);
    float pxm = readScalar(p, c - int3(1,0,0), P.size);
    float pyp = readScalar(p, c + int3(0,1,0), P.size);
    float pym = readScalar(p, c - int3(0,1,0), P.size);
    float pzp = readScalar(p, c + int3(0,0,1), P.size);
    float pzm = readScalar(p, c - int3(0,0,1), P.size);
    float3 grad = 0.5 * float3(pxp - pxm, pyp - pym, pzp - pzm);
    float3 v = velIn.read(gid).xyz - grad;
    velOut.write(float4(v, 0.0), gid);
}

// ===========================================================================
//  Volumetric raymarch render
// ===========================================================================

struct Camera {
    float4x4 invViewProj;
    float3   camPos;
    float3   boxMin;
    float3   boxMax;
    float3   lightDir;     // toward light, normalized
    float3   lightColor;
    float    absorption;   // density -> extinction
    float    scatter;      // in-scatter strength
    float    emission;     // fire emission scale
    float    tempScale;    // temperature -> [0,1] for blackbody
    int      steps;
    int      shadowSteps;
    float    stepSize;     // world units per step
    float    pad;
};

struct VSOut {
    float4 pos [[position]];
    float2 uv;
};

// Fullscreen triangle, no vertex buffer.
vertex VSOut raymarch_vertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);   // (0,0) (2,0) (0,2)
    VSOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv  = p;
    return o;
}

static inline bool intersectBox(float3 ro, float3 rd, float3 bmin, float3 bmax,
                                thread float& t0, thread float& t1) {
    float3 inv = 1.0 / rd;
    float3 a = (bmin - ro) * inv;
    float3 b = (bmax - ro) * inv;
    float3 tmin = min(a, b), tmax = max(a, b);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    return t1 > max(t0, 0.0);
}

// Flame ramp: dim red -> orange -> yellow -> white as t in [0,1].
static inline float3 blackbody(float t) {
    t = clamp(t, 0.0, 1.0);
    float3 c = float3(1.0, 0.20, 0.05);
    c = mix(c, float3(1.0, 0.55, 0.15), smoothstep(0.25, 0.55, t));
    c = mix(c, float3(1.0, 0.85, 0.55), smoothstep(0.55, 0.80, t));
    c = mix(c, float3(1.0, 1.0,  0.95), smoothstep(0.80, 1.0,  t));
    return c * t;
}

fragment float4 raymarch_fragment(VSOut in [[stage_in]],
                                  texture3d<float, access::sample> density [[texture(0)]],
                                  texture3d<float, access::sample> temp    [[texture(1)]],
                                  constant Camera& C [[buffer(0)]])
{
    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);

    // Reconstruct world ray from NDC.
    float2 ndc = in.uv * 2.0 - 1.0;
    float4 nearH = C.invViewProj * float4(ndc, 0.0, 1.0);
    float4 farH  = C.invViewProj * float4(ndc, 1.0, 1.0);
    float3 nearP = nearH.xyz / nearH.w;
    float3 farP  = farH.xyz  / farH.w;
    float3 ro = C.camPos;
    float3 rd = normalize(farP - nearP);

    float t0, t1;
    if (!intersectBox(ro, rd, C.boxMin, C.boxMax, t0, t1))
        return float4(0.0);
    t0 = max(t0, 0.0);

    float3 boxSize = C.boxMax - C.boxMin;
    float  ds = C.stepSize;
    float3 col = float3(0.0);
    float  trans = 1.0;

    float t = t0;
    for (int i = 0; i < C.steps && t < t1; ++i, t += ds) {
        float3 wp  = ro + rd * t;
        float3 uvw = (wp - C.boxMin) / boxSize;

        float d = density.sample(smp, uvw).x;
        if (d > 1e-4) {
            float sigma = d * C.absorption;
            float a = 1.0 - exp(-sigma * ds);

            // Emission (fire) from temperature.
            float  T = temp.sample(smp, uvw).x;
            float3 emit = blackbody(T * C.tempScale) * (C.emission * d);

            // Single-scatter shadow toward light.
            float lt = 1.0;
            float3 sp = wp;
            for (int j = 0; j < C.shadowSteps; ++j) {
                sp += C.lightDir * ds;
                float3 su = (sp - C.boxMin) / boxSize;
                if (any(su < 0.0) || any(su > 1.0)) break;
                float sd = density.sample(smp, su).x;
                lt *= exp(-sd * C.absorption * ds);
                if (lt < 0.02) break;
            }
            float3 scat = C.lightColor * lt * (C.scatter * d);

            col   += trans * (emit + scat) * a;
            trans *= (1.0 - a);
            if (trans < 0.01) break;
        }
    }
    return float4(col, 1.0 - trans);
}
