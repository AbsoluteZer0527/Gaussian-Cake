#pragma once
#include "core.h"
#include "MeshData.h"
#include <string>
#include <unordered_map>
#include <vector>

class OBJLoader {
public:
    // Parse a .obj file into a flat list of triangles.
    // Handles v/vt/vn face formats. Fan-triangulates polygonal faces.
    // Returns false if the file could not be opened or contains no valid triangles.
    static bool Load(const std::string& filepath, std::vector<Triangle>& outTriangles);

    // Same, keeping the appearance: texture coordinates, "v x y z r g b"
    // vertex colours, and the materials of the .mtl files it names.
    static bool Load(const std::string& filepath, MeshData& out);

    // Center and scale all triangles to fit within [-1, 1]^3.
    // Prevents scale mismatches between different meshes.
    static void Normalize(std::vector<Triangle>& tris);

private:
    // Reads the materials of one .mtl file into `materials`; `index` maps each
    // material's name to its position there (for "usemtl name" in the .obj).
    static bool LoadMtl(const std::string& path, std::vector<Material>& materials,
                        std::unordered_map<std::string, int>& index);
};
