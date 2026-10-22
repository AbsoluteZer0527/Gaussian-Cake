// Shared by the sdf_*.comp passes (pasted in after their #version line).
// Same math as SDFMorph.cpp; the grid layout is x fastest, then y, then z.

uniform sampler3D gridA;      // signed distance of A, one texel per grid point
uniform sampler3D gridB;
uniform int   n;              // grid points per axis
uniform vec3  lo;             // position of grid point (0, 0, 0)
uniform float h;              // spacing between grid points
uniform float t;              // morph time in [0, 1]
uniform mat3  LtInv;          // inverse of L_t = (1 - t) I + t A
uniform vec3  shift;          // t b
uniform mat3  Amat;           // OT map T(x) = A x + b
uniform vec3  bvec;
uniform uint  vertCap;        // room in the vertex buffer
uniform uint  indexCap;       // room in the index buffer

struct Vertex {
    vec4 position;            // xyz used
    vec4 normal;              // xyz used
};

layout(std430, binding = 0) buffer Field    { float field[]; };      // phi_t per grid point
layout(std430, binding = 1) buffer CellVert { int cellVert[]; };     // vertex index per cell, -1 = none
layout(std430, binding = 2) buffer Verts    { Vertex verts[]; };
layout(std430, binding = 3) buffer Indices  { uint indices[]; };
layout(std430, binding = 4) buffer Counters {
    uint drawCount;           // the first five are the indirect draw command
    uint instanceCount;
    uint firstIndex;
    int  baseVertex;
    uint baseInstance;
    uint vertCount;           // atomic "tickets" for vertex slots
    uint indexCount;          // atomic tickets for index slots
    uint overflow;
};

// Trilinear lookup (done by the texture hardware). Outside the grid: the
// boundary value plus the distance to the grid box, like SDFGrid::Sample.
float SampleGrid(sampler3D g, vec3 p) {
    vec3 q  = (p - lo) / h;                       // in grid-point units
    vec3 qc = clamp(q, vec3(0.0), vec3(float(n - 1)));
    float outside = h * length(q - qc);
    // texel centers sit at (i + 0.5) / n
    return texture(g, (qc + 0.5) / float(n)).r + outside;
}

// phi_t(y) = (1 - t) phi_A(u) + t phi_B(T(u)),  u = M_t^-1(y)
float Phi(vec3 y) {
    vec3 u = LtInv * (y - shift);
    float fa = (t < 1.0) ? SampleGrid(gridA, u) : 0.0;
    float fb = (t > 0.0) ? SampleGrid(gridB, Amat * u + bvec) : 0.0;
    return (1.0 - t) * fa + t * fb;
}

uint PointId(ivec3 p) { return uint((p.z * n + p.y) * n + p.x); }
uint CellId(ivec3 c)  { int m = n - 1; return uint((c.z * m + c.y) * m + c.x); }
