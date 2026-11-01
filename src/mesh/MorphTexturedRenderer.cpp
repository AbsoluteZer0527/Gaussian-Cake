#include "MorphTexturedRenderer.h"
#include "Parallel.h"
#include "SurfaceQuery.h"

#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cstddef>
#include <numeric>

// ---- Bake: where each corner lands in each texture ----

namespace {

// One side's lookup for one corner
struct SideHit {
    glm::vec3 point{0.0f};   // closest point on the side's surface
    glm::vec2 uv{0.0f};
    glm::vec3 tint{1.0f};
    int material = -1;       // a textured material, or -1
};

SideHit LookUp(const SurfaceColor& color, const glm::vec3& p) {
    SideHit out;
    SurfaceBVH::Hit hit = color.Closest(p);
    if (hit.tri < 0) return out;

    const MeshData& mesh = color.Mesh();
    const Triangle& t = mesh.tris[hit.tri];
    glm::vec3 w = Barycentric(hit.point, t.v0, t.v1, t.v2);
    out.point = hit.point;
    out.uv    = w.x * t.uv0 + w.y * t.uv1 + w.z * t.uv2;
    out.tint  = w.x * t.c0 + w.y * t.c1 + w.z * t.c2;
    if (t.material >= 0 && t.material < (int)mesh.materials.size()) {
        const Material& m = mesh.materials[t.material];
        out.tint *= m.baseColor;
        if (m.texture) out.material = t.material;
    }
    return out;
}

float Longest(float a, float b, float c) { return std::max(a, std::max(b, c)); }

// Every corner's hit on one side, then the per-triangle checks
std::vector<SideHit> PullBackSide(const std::vector<glm::vec3>& pos,
                                  const std::vector<unsigned int>& indices,
                                  const SurfaceColor& color) {
    const int corners = (int)indices.size();
    const int faces = corners / 3;
    std::vector<SideHit> hits(corners);
    ParallelFor(corners, [&](int c) { hits[c] = LookUp(color, pos[indices[c]]); });

    // How stretched each triangle is in uv: its longest uv edge over its longest
    // edge on the surface. Inside one texture island that's about the same
    // everywhere; a triangle across a seam reaches over to another island, so
    // its uv edges are far longer than usual.
    std::vector<float> stretch(faces, -1.0f);
    for (int f = 0; f < faces; ++f) {
        const SideHit& h0 = hits[3 * f + 0];
        const SideHit& h1 = hits[3 * f + 1];
        const SideHit& h2 = hits[3 * f + 2];
        float uvLen = Longest(glm::length(h1.uv - h0.uv), glm::length(h2.uv - h1.uv), glm::length(h0.uv - h2.uv));
        float len   = Longest(glm::length(h1.point - h0.point), glm::length(h2.point - h1.point),
                              glm::length(h0.point - h2.point));
        if (len > 1e-6f) stretch[f] = uvLen / len;
    }
    std::vector<float> valid;
    for (float s : stretch) {
        if (s >= 0.0f) valid.push_back(s);
    }
    float typical = 0.0f;
    if (!valid.empty()) {
        std::nth_element(valid.begin(), valid.begin() + valid.size() / 2, valid.end());
        typical = valid[valid.size() / 2];
    }

    // Untextured on this side if its corners use different materials, or a
    // corner has none, or it straddles a seam
    const float kSeamStretch = 4.0f;
    for (int f = 0; f < faces; ++f) {
        int m0 = hits[3 * f + 0].material;
        int m1 = hits[3 * f + 1].material;
        int m2 = hits[3 * f + 2].material;
        bool mixed = m0 < 0 || m0 != m1 || m0 != m2;
        bool seam  = typical > 0.0f && stretch[f] > kSeamStretch * typical;
        if (mixed || seam) {
            for (int k = 0; k < 3; ++k) hits[3 * f + k].material = -1;
        }
    }
    return hits;
}

}   // namespace

std::vector<CornerUV> PullBackCorners(const std::vector<glm::vec3>& startPositions,
                                      const std::vector<glm::vec3>& endPositions,
                                      const std::vector<unsigned int>& indices,
                                      const SurfaceColor& colorA, const SurfaceColor& colorB) {
    std::vector<SideHit> a = PullBackSide(startPositions, indices, colorA);
    std::vector<SideHit> b = PullBackSide(endPositions, indices, colorB);
    std::vector<CornerUV> out(indices.size());
    for (size_t c = 0; c < indices.size(); ++c) {
        out[c].uvA   = a[c].uv;
        out[c].tintA = a[c].tint;
        out[c].matA  = a[c].material;
        out[c].uvB   = b[c].uv;
        out[c].tintB = b[c].tint;
        out[c].matB  = b[c].material;
    }
    return out;
}

// ---- Renderer ----

MorphTexturedRenderer::~MorphTexturedRenderer() { Release(); }

void MorphTexturedRenderer::Release() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    vao = vbo = 0;
    ranges.clear();
    texturesA.clear();
    texturesB.clear();
}

// One GL texture per material of `data` (0 = untextured)
static std::vector<GLuint> MaterialTextures(const MeshData& data, TextureCache& textures) {
    std::vector<GLuint> ids;
    for (const Material& m : data.materials) {
        ids.push_back(m.texture ? textures.Get(m.texture) : 0);
    }
    return ids;
}

// Area-weighted vertex normals (as MeshRenderer::ComputeNormals)
static std::vector<glm::vec3> VertexNormals(const std::vector<glm::vec3>& pos,
                                            const std::vector<unsigned int>& indices) {
    std::vector<glm::vec3> n(pos.size(), glm::vec3(0.0f));
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        unsigned int i0 = indices[t];
        unsigned int i1 = indices[t + 1];
        unsigned int i2 = indices[t + 2];
        glm::vec3 faceN = glm::cross(pos[i1] - pos[i0], pos[i2] - pos[i0]);   // length = 2 x area
        n[i0] += faceN;
        n[i1] += faceN;
        n[i2] += faceN;
    }
    for (glm::vec3& v : n) {
        float len = glm::length(v);
        v = (len > 1e-20f) ? v / len : glm::vec3(0.0f, 1.0f, 0.0f);
    }
    return n;
}

void MorphTexturedRenderer::Initialize(const std::vector<glm::vec3>& startPositions,
                                       const std::vector<glm::vec3>& endPositions,
                                       const std::vector<unsigned int>& indices,
                                       const std::vector<glm::vec3>& startColors,
                                       const std::vector<glm::vec3>& endColors,
                                       const std::vector<CornerUV>& corners,
                                       const MeshData& dataA, const MeshData& dataB,
                                       TextureCache& textures) {
    Release();
    if (indices.empty() || corners.size() != indices.size()) return;
    texturesA = MaterialTextures(dataA, textures);
    texturesB = MaterialTextures(dataB, textures);

    std::vector<glm::vec3> normalsStart = VertexNormals(startPositions, indices);
    std::vector<glm::vec3> normalsEnd   = VertexNormals(endPositions, indices);

    // Triangles sorted by their (A material, B material) pair, so each pair
    // is one draw call. All 3 corners of a triangle share its materials.
    const int faces = (int)indices.size() / 3;
    std::vector<int> order(faces);
    std::iota(order.begin(), order.end(), 0);
    auto materialsOf = [&](int f) { return std::make_pair(corners[3 * f].matA, corners[3 * f].matB); };
    std::stable_sort(order.begin(), order.end(), [&](int f, int g) { return materialsOf(f) < materialsOf(g); });

    std::vector<Vertex> verts;
    verts.reserve(indices.size());
    for (int f : order) {
        auto mats = materialsOf(f);
        if (ranges.empty() || ranges.back().matA != mats.first || ranges.back().matB != mats.second)
            ranges.push_back({ (int)verts.size(), 0, mats.first, mats.second });
        for (int k = 0; k < 3; ++k) {
            int c = 3 * f + k;
            unsigned int v = indices[c];
            const CornerUV& uv = corners[c];
            glm::vec3 vertexStart = startColors.empty() ? glm::vec3(1.0f) : startColors[v];
            glm::vec3 vertexEnd   = endColors.empty() ? glm::vec3(1.0f) : endColors[v];

            Vertex out;
            out.positionStart = startPositions[v];
            out.normalStart   = normalsStart[v];
            out.positionEnd   = endPositions[v];
            out.normalEnd     = normalsEnd[v];
            out.colorStart    = (uv.matA >= 0) ? uv.tintA : vertexStart;
            out.colorEnd      = (uv.matB >= 0) ? uv.tintB : vertexEnd;
            out.uvA           = uv.uvA;
            out.uvB           = uv.uvB;
            verts.push_back(out);
        }
        ranges.back().count += 3;
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
    auto attrib = [](GLuint loc, int size, size_t offset) {
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offset);
    };
    attrib(0, 3, offsetof(Vertex, positionStart));
    attrib(1, 3, offsetof(Vertex, normalStart));
    attrib(2, 3, offsetof(Vertex, positionEnd));
    attrib(3, 3, offsetof(Vertex, normalEnd));
    attrib(4, 3, offsetof(Vertex, colorStart));
    attrib(5, 3, offsetof(Vertex, colorEnd));
    attrib(6, 2, offsetof(Vertex, uvA));
    attrib(7, 2, offsetof(Vertex, uvB));
    glBindVertexArray(0);
}

void MorphTexturedRenderer::Draw(GLuint shaderProgram, const glm::mat4& viewProj, float morphT) {
    if (!vao) return;
    glUseProgram(shaderProgram);
    glm::mat4 model(1.0f);
    auto loc = [&](const char* name) { return glGetUniformLocation(shaderProgram, name); };
    glUniformMatrix4fv(loc("viewProj"), 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniformMatrix4fv(loc("model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform1f(loc("alpha"), 1.0f);
    glUniform1f(loc("morphT"), morphT);
    glUniform1i(loc("useMorphTexture"), 1);
    glUniform1i(loc("albedoTex"), 0);
    glUniform1i(loc("albedoTexB"), 1);

    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(vao);
    for (const Range& r : ranges) {
        GLuint texA = (r.matA >= 0 && r.matA < (int)texturesA.size()) ? texturesA[r.matA] : 0;
        GLuint texB = (r.matB >= 0 && r.matB < (int)texturesB.size()) ? texturesB[r.matB] : 0;
        glUniform1i(loc("useTexture"), texA ? 1 : 0);
        glUniform1i(loc("useTextureB"), texB ? 1 : 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texA);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texB);
        glDrawArrays(GL_TRIANGLES, r.first, r.count);
    }
    glBindTexture(GL_TEXTURE_2D, 0);   // unit 1
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    // the uniforms stay set on the program: reset them for MeshRenderer's draws
    glUniform1i(loc("useMorphTexture"), 0);
    glUniform1i(loc("useTexture"), 0);
    glUniform1i(loc("useTextureB"), 0);
    glBindVertexArray(0);
    glUseProgram(0);
}
