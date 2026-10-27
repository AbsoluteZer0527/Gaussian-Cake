#include "PlyLoader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <cfloat>
#include <glm/gtc/constants.hpp>

bool PlyLoader::Load(const std::string& path, std::vector<Splat>& outSplats,
                     const PlyLoadOptions& options) {
    outSplats.clear();

    // std::ios::binary: on Windows, text mode would turn "\r\n" into "\n" and
    // corrupt the binary floats after the header.
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[PlyLoader] cannot open " << path << "\n";
        return false;
    }

    Header header;
    std::string error;
    if (!ReadHeader(in, header, error)) {
        std::cerr << "[PlyLoader] " << path << ": " << error << "\n";
        return false;
    }

    // Read all records in one go: vertexCount x stride bytes.
    std::vector<char> data(header.vertexCount * header.stride);
    if (!in.read(data.data(), (std::streamsize)data.size())) {
        std::cerr << "[PlyLoader] " << path << ": file ends before all "
                  << header.vertexCount << " splats were read\n";
        return false;
    }

    outSplats.resize(header.vertexCount);
    for (size_t i = 0; i < header.vertexCount; ++i) {
        outSplats[i] = Convert(data.data() + i * header.stride, header);
        MakeZThinnest(outSplats[i]);
    }

    size_t loaded = outSplats.size();
    CleanAndNormalize(outSplats, options);
    std::cout << "[PlyLoader] " << path << ": " << loaded << " splats, "
              << outSplats.size() << " kept after cleaning\n";
    return !outSplats.empty();
}


size_t PlyLoader::CountSplats(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    Header header;
    std::string error;
    return (in && ReadHeader(in, header, error)) ? header.vertexCount : 0;
}

const char* const PlyLoader::kRequired[PlyLoader::kFieldCount] = {
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
};

bool PlyLoader::ReadHeader(std::ifstream& in, Header& header, std::string& error) {
    // one header line, without the '\r' some exporters leave at the end
    auto readLine = [&](std::string& line) {
        if (!std::getline(in, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    };

    std::string line;
    if (!readLine(line) || line != "ply") {
        error = "not a .ply file (first line is not 'ply')";
        return false;
    }

    bool inVertexElement = false;
    bool seenVertex = false;
    bool ended = false;
    while (readLine(line)) {
        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "end_header") {
            ended = true;
            break;
        } else if (keyword == "format") {
            std::string format;
            ss >> format;
            if (format != "binary_little_endian") {
                error = "format '" + format + "' is not supported (only binary_little_endian)";
                return false;
            }
        } else if (keyword == "element") {
            std::string name;
            size_t count = 0;
            ss >> name >> count;
            inVertexElement = (name == "vertex");
            if (inVertexElement) {
                header.vertexCount = count;
                seenVertex = true;
            } else if (count > 0 && !seenVertex) {
                // its data would sit between the header and ours (compressed exports do this);
                // an element declared after vertex is stored after our data, so it's harmless
                error = "extra element '" + name + "' before vertex is not supported (compressed .ply?)";
                return false;
            }
        } else if (keyword == "property" && inVertexElement) {
            std::string type, name;
            ss >> type >> name;
            if (type != "float" && type != "float32") {
                error = "property '" + name + "' has type '" + type + "' (only float is supported)";
                return false;
            }
            header.offset[name] = header.stride;
            header.stride += sizeof(float);
        }
    }

    if (!ended) {
        error = "the header never ends (no end_header)";
        return false;
    }
    if (header.vertexCount == 0) {
        error = "no splats (element vertex is missing or empty)";
        return false;
    }
    for (int f = 0; f < kFieldCount; ++f) {
        auto it = header.offset.find(kRequired[f]);
        if (it == header.offset.end()) {
            error = std::string("missing property '") + kRequired[f] + "' (not a 3DGS capture?)";
            return false;
        }
        header.field[f] = it->second;
    }
    return true;
}

// 3DGS stores values before their activation functions: log scales, logit
// opacity, and colour as its degree-0 spherical-harmonic coefficient.
static const float kSH0 = 0.28209479f;

Splat PlyLoader::Convert(const char* record, const Header& header) {
    // memcpy, not a cast: the bytes in the buffer aren't guaranteed to be
    // aligned like a float, and memcpy is the portable way to reinterpret them
    auto get = [&](Field f) {
        float v;
        std::memcpy(&v, record + header.field[f], sizeof(float));
        return v;
    };

    Splat s;
    s.mean    = glm::vec3(get(X), get(Y), get(Z));
    s.color   = glm::clamp(0.5f + kSH0 * glm::vec3(get(DC0), get(DC1), get(DC2)), 0.0f, 1.0f);
    s.opacity = 1.0f / (1.0f + std::exp(-get(Opacity)));
    s.scale   = glm::exp(glm::vec3(get(Scale0), get(Scale1), get(Scale2)));

    glm::quat q(get(Rot0), get(Rot1), get(Rot2), get(Rot3));   // glm order: (w, x, y, z)
    float len = glm::length(q);
    s.rot = (len > 1e-12f) ? q / len : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    return s;
}

void PlyLoader::MakeZThinnest(Splat& s) {
    int k = 0;
    if (s.scale[1] < s.scale[k]) k = 1;
    if (s.scale[2] < s.scale[k]) k = 2;
    if (k == 2) return;

    // Relabel the axes cyclically so axis k comes last: (x,y,z) -> (y,z,x) or
    // (z,x,y). A cyclic shift keeps the frame right-handed, so it stays a rotation.
    glm::mat3 R = glm::mat3_cast(s.rot);   // columns = the splat's axes in world space
    int a = (k + 1) % 3, b = (k + 2) % 3;
    s.rot   = glm::normalize(glm::quat_cast(glm::mat3(R[a], R[b], R[k])));
    s.scale = glm::vec3(s.scale[a], s.scale[b], s.scale[k]);
}

static bool IsFinite(const glm::vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// k-th smallest value (k in [0, size)); takes a copy because nth_element reorders
static float KthValue(std::vector<float> v, size_t k) {
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

void PlyLoader::CleanAndNormalize(std::vector<Splat>& splats, const PlyLoadOptions& options) {
    // 1. invisible or broken splats. A NaN opacity fails every comparison, so
    //    "opacity < 1/255" alone would keep it: check isfinite first.
    splats.erase(std::remove_if(splats.begin(), splats.end(), [](const Splat& s) {
        return !IsFinite(s.mean) || !IsFinite(s.scale) || !std::isfinite(s.opacity) ||
               s.opacity < 1.0f / 255.0f;
    }), splats.end());
    if (splats.empty()) return;

    // 2. up axis. FlipY turns the capture upside down by a 180-degree turn about
    //    X (y and z both flip): a mirror would make every splat frame left-handed.
    glm::quat up(1.0f, 0.0f, 0.0f, 0.0f);
    const glm::vec3 xAxis(1.0f, 0.0f, 0.0f);
    switch (options.up) {
        case PlyLoadOptions::Up::FlipY:        up = glm::angleAxis(glm::pi<float>(), xAxis); break;
        case PlyLoadOptions::Up::RotateXPos90: up = glm::angleAxis(glm::half_pi<float>(), xAxis); break;
        case PlyLoadOptions::Up::RotateXNeg90: up = glm::angleAxis(-glm::half_pi<float>(), xAxis); break;
        case PlyLoadOptions::Up::AsIs:         break;
    }
    if (options.up != PlyLoadOptions::Up::AsIs) {
        for (Splat& s : splats) {
            s.mean = up * s.mean;
            s.rot  = up * s.rot;
        }
    }

    // 3. crop. Median centre and 90th-percentile radius, so the background and
    //    far floaters don't get to decide where "the object" is.
    const size_t n = splats.size();
    std::vector<float> xs(n), ys(n), zs(n);
    for (size_t i = 0; i < n; ++i) {
        xs[i] = splats[i].mean.x;
        ys[i] = splats[i].mean.y;
        zs[i] = splats[i].mean.z;
    }
    glm::vec3 centre(KthValue(xs, n / 2), KthValue(ys, n / 2), KthValue(zs, n / 2));

    std::vector<float> dist(n);
    for (size_t i = 0; i < n; ++i) dist[i] = glm::length(splats[i].mean - centre);
    float keep = options.cropRadius * KthValue(dist, n * 9 / 10);

    splats.erase(std::remove_if(splats.begin(), splats.end(), [&](const Splat& s) {
        return glm::length(s.mean - centre) > keep;
    }), splats.end());
    if (splats.empty()) return;

    // 4. into [-1, 1]^3, the same way OBJLoader::Normalize places meshes
    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    for (const Splat& s : splats) {
        lo = glm::min(lo, s.mean);
        hi = glm::max(hi, s.mean);
    }
    glm::vec3 mid = 0.5f * (lo + hi);
    glm::vec3 half = 0.5f * (hi - lo);
    float extent = std::max(half.x, std::max(half.y, half.z));
    if (extent < 1e-6f) extent = 1.0f;
    for (Splat& s : splats) {
        s.mean   = (s.mean - mid) / extent;
        s.scale /= extent;
    }
}
