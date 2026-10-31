#include "Texture.h"

#include <cmath>
#include <iostream>

// stb_image is a single header: exactly one .cpp defines this before
// including it, which compiles the implementation in.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool Texture::LoadFile(const std::string& path) {
    int w, h, channels;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);   // force RGBA
    if (!pixels) {
        std::cerr << "[Texture] cannot load " << path << ": " << stbi_failure_reason() << "\n";
        return false;
    }
    FromRGBA8(pixels, w, h);
    stbi_image_free(pixels);
    return true;
}

bool Texture::LoadMemory(const unsigned char* bytes, size_t size) {
    int w, h, channels;
    unsigned char* pixels = stbi_load_from_memory(bytes, (int)size, &w, &h, &channels, 4);
    if (!pixels) {
        std::cerr << "[Texture] cannot decode embedded image: " << stbi_failure_reason() << "\n";
        return false;
    }
    FromRGBA8(pixels, w, h);
    stbi_image_free(pixels);
    return true;
}

void Texture::FromRGBA8(const unsigned char* pixels, int w, int h) {
    width = w;
    height = h;
    texels.resize((size_t)w * h);
    for (size_t i = 0; i < texels.size(); ++i)
        texels[i] = glm::vec3(pixels[4 * i], pixels[4 * i + 1], pixels[4 * i + 2]) / 255.0f;
}

// Bilinear: blend the 4 texels around uv, weighted by how close each one is.
// The image repeats outside [0, 1] (like GL_REPEAT).
glm::vec3 Texture::Sample(const glm::vec2& uv) const {
    if (width == 0 || height == 0) return glm::vec3(1.0f);

    // texel centres sit at half-integers, so shift by half a texel
    float x = uv.x * width  - 0.5f;
    float y = uv.y * height - 0.5f;
    int   x0 = (int)std::floor(x);
    int   y0 = (int)std::floor(y);
    float fx = x - (float)x0;   // 0 = on texel x0, 1 = on texel x0 + 1
    float fy = y - (float)y0;

    // % keeps the sign in C++ (-1 % 5 == -1), so fix negatives up
    auto wrap = [](int i, int n) { int r = i % n; return r < 0 ? r + n : r; };
    int xa = wrap(x0, width),  xb = wrap(x0 + 1, width);
    int ya = wrap(y0, height), yb = wrap(y0 + 1, height);

    glm::vec3 top    = glm::mix(Texel(xa, ya), Texel(xb, ya), fx);
    glm::vec3 bottom = glm::mix(Texel(xa, yb), Texel(xb, yb), fx);
    return glm::mix(top, bottom, fy);
}
