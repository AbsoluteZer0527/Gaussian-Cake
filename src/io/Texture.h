#pragma once
#include "core.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// An image kept on the CPU, for looking up a mesh's colour at a texture coordinate.
class Texture {
public:
    bool LoadFile(const std::string& path);
    bool LoadMemory(const unsigned char* bytes, size_t size);   // e.g. an image inside a .glb

    // Colour at uv, blending the 4 nearest texels (bilinear). The image
    // repeats outside [0, 1] like GL_REPEAT: u = 1.25 reads the same as 0.25.
    glm::vec3 Sample(const glm::vec2& uv) const;

    // A new GL texture with this image (mipmapped, repeating). The caller owns it
    // and frees it with glDeleteTextures.
    GLuint CreateGLTexture() const;

    int Width() const { return width; }
    int Height() const { return height; }
    const std::vector<glm::vec3>& Texels() const { return texels; }

private:
    int width = 0, height = 0;
    std::vector<glm::vec3> texels;   // row-major, row 0 = the image's top, values in [0, 1]

    glm::vec3 Texel(int x, int y) const { return texels[(size_t)y * width + x]; }
    void FromRGBA8(const unsigned char* pixels, int w, int h);
};

// GL copies of images, one per image, shared by every renderer that draws them
// (so the rest mesh and the textured morph don't each upload the same image).
// Clear() frees them all; call it before the GL context goes away.
class TextureCache {
public:
    GLuint Get(const std::shared_ptr<Texture>& image);   // uploads on first use
    void Clear();

private:
    std::unordered_map<const Texture*, GLuint> ids;
    // Holding the images keeps their addresses (the keys above) from being
    // reused by a new image while the cache still maps them
    std::vector<std::shared_ptr<Texture>> images;
};
