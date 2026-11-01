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

// Texels are display values in [0, 1], row 0 = the image top. Uploading row 0
// first puts it at t = 0, which is where our uv (v = 0 at the top) looks.
GLuint Texture::CreateGLTexture() const {
    std::vector<unsigned char> rgb(texels.size() * 3);
    for (size_t i = 0; i < texels.size(); ++i)
        for (int ch = 0; ch < 3; ++ch)
            rgb[3 * i + ch] = (unsigned char)std::lround(glm::clamp(texels[i][ch], 0.0f, 1.0f) * 255.0f);

    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);   // rows of 3-byte pixels aren't 4-byte aligned
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    glGenerateMipmap(GL_TEXTURE_2D);         // smaller copies, so a far-away mesh doesn't shimmer
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
    return id;
}

GLuint TextureCache::Get(const std::shared_ptr<Texture>& image) {
    auto it = ids.find(image.get());
    if (it != ids.end()) return it->second;
    GLuint id = image->CreateGLTexture();
    ids[image.get()] = id;
    images.push_back(image);
    return id;
}

void TextureCache::Clear() {
    for (auto& entry : ids) glDeleteTextures(1, &entry.second);
    ids.clear();
    images.clear();
}
