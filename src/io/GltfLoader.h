#pragma once
#include "MeshData.h"
#include <string>

// Loads a glTF 2.0 model (.gltf + its files, or a single .glb): every
// triangle primitive in the default scene with its node transform applied,
// plus TEXCOORD_0, COLOR_0 and the base colour (factor and texture) of each
// material. Animation, skinning and the other PBR maps are ignored.
class GltfLoader {
public:
    static bool Load(const std::string& path, MeshData& out);
};
