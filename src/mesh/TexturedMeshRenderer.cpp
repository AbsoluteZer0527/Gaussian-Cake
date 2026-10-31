#include "TexturedMeshRenderer.h"

#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>

namespace {

// One vertex as uploaded: matches the attribute locations in shader.vert
struct Vertex {
    glm::vec3 position;   // location 0
    glm::vec3 normal;     // location 1
    glm::vec3 color;      // location 4
    glm::vec2 uv;         // location 6
};

// Exact byte key for welding: vertices merge only if every value matches
template <typename T>
std::string Key(const T& value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(T));
}

}   // namespace

TexturedMeshRenderer::~TexturedMeshRenderer() { Release(); }

void TexturedMeshRenderer::Release() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    vao = vbo = ebo = 0;
    for (auto& entry : uploaded) glDeleteTextures(1, &entry.second);
    uploaded.clear();
    ranges.clear();
    materialColors.clear();
    materialTextures.clear();
}

// Texels are display values in [0, 1], row 0 = the image top. Uploading row 0
// first puts it at t = 0, which is where our uv (v = 0 at the top) looks.
GLuint TexturedMeshRenderer::Upload(const Texture& tex) {
    auto it = uploaded.find(&tex);
    if (it != uploaded.end()) return it->second;

    std::vector<unsigned char> rgb(tex.Texels().size() * 3);
    for (size_t i = 0; i < tex.Texels().size(); ++i)
        for (int ch = 0; ch < 3; ++ch)
            rgb[3 * i + ch] = (unsigned char)std::lround(glm::clamp(tex.Texels()[i][ch], 0.0f, 1.0f) * 255.0f);

    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);   // rows of 3-byte pixels aren't 4-byte aligned
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, tex.Width(), tex.Height(), 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    glGenerateMipmap(GL_TEXTURE_2D);         // smaller copies, so a far-away mesh doesn't shimmer
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
    uploaded[&tex] = id;
    return id;
}

void TexturedMeshRenderer::Initialize(const MeshData& mesh) {
    Release();
    if (mesh.tris.empty()) return;

    // Material slots: one per material, plus a last one for "no material"
    const int noMaterial = (int)mesh.materials.size();
    for (const Material& m : mesh.materials) {
        materialColors.push_back(m.baseColor);
        materialTextures.push_back(m.texture ? Upload(*m.texture) : 0);
    }
    materialColors.push_back(glm::vec3(1.0f));
    materialTextures.push_back(0);

    // Weld corners into vertices; collect each material's indices separately
    std::vector<Vertex> verts;
    std::vector<std::vector<unsigned int>> indicesOf(noMaterial + 1);
    std::unordered_map<std::string, unsigned int> weld;
    for (const Triangle& t : mesh.tris) {
        int mat = (t.material >= 0 && t.material < noMaterial) ? t.material : noMaterial;
        const glm::vec3 p[3]  = { t.v0, t.v1, t.v2 };
        const glm::vec3 c[3]  = { t.c0, t.c1, t.c2 };
        const glm::vec2 uv[3] = { t.uv0, t.uv1, t.uv2 };
        for (int k = 0; k < 3; ++k) {
            std::string key = Key(p[k]) + Key(uv[k]) + Key(c[k]) + Key(mat);
            auto it = weld.find(key);
            if (it == weld.end()) {
                it = weld.emplace(key, (unsigned int)verts.size()).first;
                verts.push_back({ p[k], glm::vec3(0.0f), c[k], uv[k] });
            }
            indicesOf[mat].push_back(it->second);
        }
    }

    // Smooth normals over every vertex at one position (across UV seams)
    std::unordered_map<std::string, glm::vec3> normalAt;
    for (const Triangle& t : mesh.tris) {
        glm::vec3 n = glm::cross(t.v1 - t.v0, t.v2 - t.v0);   // length = 2 x area: area-weighted
        for (const glm::vec3& p : { t.v0, t.v1, t.v2 }) normalAt[Key(p)] += n;
    }
    for (Vertex& v : verts) {
        glm::vec3 n = normalAt[Key(v.position)];
        float len = glm::length(n);
        v.normal = (len > 1e-20f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
    }

    std::vector<unsigned int> indices;
    for (int m = 0; m <= noMaterial; ++m) {
        if (indicesOf[m].empty()) continue;
        ranges.push_back({ (int)indices.size(), (int)indicesOf[m].size(), m });
        indices.insert(indices.end(), indicesOf[m].begin(), indicesOf[m].end());
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
    auto attrib = [](GLuint loc, int size, size_t offset) {
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offset);
    };
    attrib(0, 3, offsetof(Vertex, position));
    attrib(1, 3, offsetof(Vertex, normal));
    attrib(4, 3, offsetof(Vertex, color));
    attrib(6, 2, offsetof(Vertex, uv));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

void TexturedMeshRenderer::Draw(GLuint shaderProgram, const glm::mat4& viewProj) {
    if (!vao) return;
    glUseProgram(shaderProgram);
    glm::mat4 model(1.0f);
    auto loc = [&](const char* name) { return glGetUniformLocation(shaderProgram, name); };
    glUniformMatrix4fv(loc("viewProj"), 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniformMatrix4fv(loc("model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform1f(loc("alpha"), 1.0f);
    glUniform1i(loc("useVertexColor"), 1);
    glUniform1i(loc("albedoTex"), 0);

    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    for (const Range& r : ranges) {
        glUniform3fv(loc("MaterialColor"), 1, glm::value_ptr(materialColors[r.material]));
        GLuint tex = materialTextures[r.material];
        glUniform1i(loc("useTexture"), tex ? 1 : 0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawElements(GL_TRIANGLES, r.count, GL_UNSIGNED_INT, (void*)(r.first * sizeof(unsigned int)));
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glUniform1i(loc("useTexture"), 0);            // the uniforms stay set on the program:
    glUniform3f(loc("MaterialColor"), 1, 1, 1);   // reset them for MeshRenderer's draws
    glBindVertexArray(0);
    glUseProgram(0);
}
