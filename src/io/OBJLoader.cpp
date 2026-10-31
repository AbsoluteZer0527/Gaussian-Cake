#include "OBJLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <climits>
#include <cfloat>
#include <filesystem>
#include <glm/gtx/component_wise.hpp>

bool OBJLoader::Load(const std::string& filepath, std::vector<Triangle>& outTriangles) {
    MeshData mesh;
    if (!Load(filepath, mesh)) {
        outTriangles.clear();
        return false;
    }
    outTriangles = std::move(mesh.tris);
    return true;
}

// OBJ indices are 1-based; negative ones count back from the end. Returns -1
// for a missing or broken index.
static int ResolveIndex(const std::string& s, size_t count) {
    if (s.empty()) return -1;
    int idx;
    try {
        idx = std::stoi(s);
    } catch (const std::exception&) {
        return -1;
    }
    idx = (idx < 0) ? (int)count + idx : idx - 1;
    return (idx >= 0 && idx < (int)count) ? idx : -1;
}

//simple .obj file loader code
bool OBJLoader::Load(const std::string& filepath, MeshData& out) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[OBJLoader] Could not open: " << filepath << "\n";
        return false;
    }

    std::vector<glm::vec3> verts, colors;
    std::vector<glm::vec2> uvs;
    std::unordered_map<std::string, int> materialIndex;
    int currentMaterial = -1;
    out = MeshData();
    const std::filesystem::path dir = std::filesystem::path(filepath).parent_path();

    // one face corner: the position, and its uv if it has one
    struct Corner { int v, vt; };

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            // "v x y z" or, from many scanners, "v x y z r g b"
            float x, y, z;
            ss >> x >> y >> z;
            verts.push_back({x, y, z});
            float r, g, b;
            if (ss >> r >> g >> b) {
                colors.push_back({r, g, b});
                out.hasVertexColors = true;
            } else {
                colors.push_back(glm::vec3(1.0f));
            }
        }
        else if (token == "vt") {
            float u = 0.0f, v = 0.0f;
            ss >> u >> v;
            uvs.push_back({u, 1.0f - v});   // OBJ's v = 0 is the image bottom; ours is the top
        }
        else if (token == "mtllib") {
            std::string name;
            std::getline(ss >> std::ws, name);   // the rest of the line: names can have spaces
            if (!name.empty() && name.back() == '\r') name.pop_back();
            LoadMtl((dir / name).string(), out.materials, materialIndex);
        }
        else if (token == "usemtl") {
            std::string name;
            ss >> name;
            auto it = materialIndex.find(name);
            currentMaterial = (it != materialIndex.end()) ? it->second : -1;
        }
        else if (token == "f") {
            // Each face token may be: v, v/vt, v/vt/vn, or v//vn
            std::vector<Corner> corners;
            std::string faceToken;
            while (ss >> faceToken) {
                size_t slash = faceToken.find('/');
                Corner c;
                c.v = ResolveIndex(faceToken.substr(0, slash), verts.size());
                c.vt = -1;
                if (slash != std::string::npos) {
                    size_t slash2 = faceToken.find('/', slash + 1);
                    c.vt = ResolveIndex(faceToken.substr(slash + 1, slash2 - slash - 1), uvs.size());
                }
                if (c.v >= 0) corners.push_back(c);   // malformed index: skip it instead of crashing
            }

            // Fan triangulation: (0,1,2), (0,2,3), (0,3,4), ...
            auto uvOf = [&](const Corner& c) { return c.vt >= 0 ? uvs[c.vt] : glm::vec2(0.0f); };
            for (int i = 1; i + 1 < (int)corners.size(); i++) {
                const Corner& a = corners[0];
                const Corner& b = corners[i];
                const Corner& c = corners[i + 1];
                Triangle t{verts[a.v], verts[b.v], verts[c.v]};
                t.uv0 = uvOf(a); t.uv1 = uvOf(b); t.uv2 = uvOf(c);
                t.c0 = colors[a.v]; t.c1 = colors[b.v]; t.c2 = colors[c.v];
                t.material = currentMaterial;
                out.tris.push_back(t);
            }
        }
    }

    if (out.tris.empty()) {
        std::cerr << "[OBJLoader] No triangles found in: " << filepath << "\n";
        return false;
    }

    std::cout << "[OBJLoader] Loaded " << out.tris.size() << " triangles, "
              << out.materials.size() << " materials from " << filepath << "\n";
    return true;
}

// A .mtl file is line-based like the .obj: "newmtl name" starts a material,
// "Kd r g b" sets its colour, "map_Kd file" its texture (relative to the .mtl).
bool OBJLoader::LoadMtl(const std::string& path, std::vector<Material>& materials,
                        std::unordered_map<std::string, int>& index) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[OBJLoader] Could not open material file: " << path << "\n";
        return false;
    }
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "newmtl") {
            std::string name;
            ss >> name;
            index[name] = (int)materials.size();
            materials.push_back(Material());
        } else if (materials.empty()) {
            continue;   // anything before the first newmtl has no material to go to
        } else if (keyword == "Kd") {
            float r = 1.0f, g = 1.0f, b = 1.0f;
            ss >> r >> g >> b;
            // Kd is linear; our colours are display values
            materials.back().baseColor = glm::pow(glm::vec3(r, g, b), glm::vec3(1.0f / 2.2f));
        } else if (keyword == "map_Kd") {
            // options may come first ("map_Kd -s 1 1 1 file.png"): the file is the last word
            std::string word, image;
            while (ss >> word) image = word;
            if (image.empty()) continue;
            auto tex = std::make_shared<Texture>();
            if (tex->LoadFile((dir / image).string())) materials.back().texture = tex;
        }
    }
    return true;
}

void OBJLoader::Normalize(std::vector<Triangle>& tris) {
    if (tris.empty()) return;

    glm::vec3 minV(FLT_MAX), maxV(-FLT_MAX);
    for (const auto& t : tris) {
        for (const auto& v : {t.v0, t.v1, t.v2}) {
            minV = glm::min(minV, v);
            maxV = glm::max(maxV, v);
        }
    }

    glm::vec3 center = (minV + maxV) * 0.5f;
    float extent = glm::compMax(maxV - minV) * 0.5f;
    if (extent < 1e-6f) extent = 1.0f;

    for (auto& t : tris) {
        t.v0 = (t.v0 - center) / extent;
        t.v1 = (t.v1 - center) / extent;
        t.v2 = (t.v2 - center) / extent;
    }
}
