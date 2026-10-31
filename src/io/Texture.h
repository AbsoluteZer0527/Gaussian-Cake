#pragma once
#include "core.h"
#include <string>
#include <vector>

// An image kept on the CPU, for looking up a mesh's colour at a texture coordinate.
class Texture {
public:
    bool LoadFile(const std::string& path);
    bool LoadMemory(const unsigned char* bytes, size_t size);   // e.g. an image inside a .glb

    // Colour at uv, blending the 4 nearest texels (bilinear). The image
    // repeats outside [0, 1] like GL_REPEAT: u = 1.25 reads the same as 0.25.
    glm::vec3 Sample(const glm::vec2& uv) const;

    int Width() const { return width; }
    int Height() const { return height; }
    const std::vector<glm::vec3>& Texels() const { return texels; }

private:
    int width = 0, height = 0;
    std::vector<glm::vec3> texels;   // row-major, row 0 = the image's top, values in [0, 1]

    glm::vec3 Texel(int x, int y) const { return texels[(size_t)y * width + x]; }
    void FromRGBA8(const unsigned char* pixels, int w, int h);
};
