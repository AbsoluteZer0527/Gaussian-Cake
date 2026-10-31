#pragma once
#include "MeshData.h"
#include <string>

// Loads a .ply triangle mesh, the format scanners and photogrammetry tools
// write: "element vertex" with x y z and optional red green blue, then
// "element face" with a vertex_indices list. ASCII and binary (either byte
// order), any property types; polygons are fan-triangulated.
// (.ply files without faces are 3DGS splat captures: see PlyLoader.)
class PlyMeshLoader {
public:
    // Does the header declare faces? Reads the header only.
    static bool IsMesh(const std::string& path);

    static bool Load(const std::string& path, MeshData& out);
};
