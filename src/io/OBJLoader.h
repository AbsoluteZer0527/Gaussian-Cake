#pragma once
#include "core.h"
#include <vector>
#include <string>

struct Triangle {
    glm::vec3 v0, v1, v2;
};

class OBJLoader {
public:
    // Parse a .obj file into a flat list of triangles.
    // Handles v/vt/vn face formats. Fan-triangulates polygonal faces.
    // Returns false if the file could not be opened or contains no valid triangles.
    static bool Load(const std::string& filepath, std::vector<Triangle>& outTriangles);

    // Center and scale all triangles to fit within [-1, 1]^3.
    // Prevents scale mismatches between different meshes.
    static void Normalize(std::vector<Triangle>& tris);
};
