#include "OBJLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <climits>
#include <glm/gtx/component_wise.hpp>

//simple .obj file loader code
bool OBJLoader::Load(const std::string& filepath, std::vector<Triangle>& outTriangles) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[OBJLoader] Could not open: " << filepath << "\n";
        return false;
    }

    std::vector<glm::vec3> verts;
    outTriangles.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            verts.push_back({x, y, z});
        }
        else if (token == "f") {
            // Each face token may be: v, v/vt, v/vt/vn, or v//vn
            // We only care about the vertex index (first number).
            std::vector<int> indices;
            std::string faceToken;
            while (ss >> faceToken) {
                // Take only the part before the first '/'
                std::string idxStr = faceToken.substr(0, faceToken.find('/'));
                int idx;
                try {
                    idx = std::stoi(idxStr);
                } catch (const std::exception&) {
                    continue;   // malformed index: skip it instead of crashing
                }
                // OBJ is 1-based; negative means relative to end
                if (idx < 0)
                    idx = (int)verts.size() + idx;
                else
                    idx -= 1;
                indices.push_back(idx);
            }

            // Fan triangulation: (0,1,2), (0,2,3), (0,3,4), ...
            for (int i = 1; i + 1 < (int)indices.size(); i++) {
                int a = indices[0], b = indices[i], c = indices[i + 1];
                if (a < 0 || a >= (int)verts.size()) continue;
                if (b < 0 || b >= (int)verts.size()) continue;
                if (c < 0 || c >= (int)verts.size()) continue;
                outTriangles.push_back({verts[a], verts[b], verts[c]});
            }
        }
    }

    if (outTriangles.empty()) {
        std::cerr << "[OBJLoader] No triangles found in: " << filepath << "\n";
        return false;
    }

    std::cout << "[OBJLoader] Loaded " << outTriangles.size()
              << " triangles from " << filepath << "\n";
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
