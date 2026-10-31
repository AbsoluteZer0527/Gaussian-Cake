#include "PlyMeshLoader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

struct Property {
    std::string name, type;   // type of the value, or of the list's items
    bool list = false;
    std::string countType;    // type of a list's leading item count
};

struct Element {
    std::string name;
    size_t count = 0;
    std::vector<Property> props;
};

enum class Format { Ascii, BinaryLE, BinaryBE };

// The general header: every element with its properties, in file order.
bool ReadHeader(std::istream& in, Format& format, std::vector<Element>& elements, std::string& error) {
    std::string line;
    auto readLine = [&]() {
        if (!std::getline(in, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    };
    if (!readLine() || line != "ply") { error = "not a .ply file"; return false; }

    while (readLine()) {
        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;
        if (keyword == "end_header") return true;
        if (keyword == "format") {
            std::string f;
            ss >> f;
            if      (f == "ascii")                format = Format::Ascii;
            else if (f == "binary_little_endian") format = Format::BinaryLE;
            else if (f == "binary_big_endian")    format = Format::BinaryBE;
            else { error = "unknown format '" + f + "'"; return false; }
        } else if (keyword == "element") {
            Element e;
            ss >> e.name >> e.count;
            elements.push_back(e);
        } else if (keyword == "property" && !elements.empty()) {
            Property p;
            ss >> p.type;
            if (p.type == "list") {
                p.list = true;
                ss >> p.countType >> p.type;
            }
            ss >> p.name;
            elements.back().props.push_back(p);
        }
    }
    error = "the header never ends (no end_header)";
    return false;
}

size_t TypeSize(const std::string& t) {
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "int" || t == "uint" || t == "int32" || t == "uint32" ||
        t == "float" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    return 0;
}

bool IsInteger(const std::string& t) {
    return t != "float" && t != "float32" && t != "double" && t != "float64";
}

// Reads one value at a time, whatever the format, as a double.
class ValueReader {
public:
    ValueReader(std::istream& in, Format format) : in(in), format(format) {}

    bool Read(const std::string& type, double& out) {
        if (format == Format::Ascii) return (bool)(in >> out);

        size_t size = TypeSize(type);
        unsigned char bytes[8];
        if (size == 0 || !in.read((char*)bytes, (std::streamsize)size)) return false;
        if (format == Format::BinaryBE) std::reverse(bytes, bytes + size);   // to this machine's order

        // memcpy into the right type, like PlyLoader::Convert does
        auto as = [&](auto value) { std::memcpy(&value, bytes, sizeof(value)); return (double)value; };
        if      (type == "char"   || type == "int8")    out = as(int8_t());
        else if (type == "uchar"  || type == "uint8")   out = as(uint8_t());
        else if (type == "short"  || type == "int16")   out = as(int16_t());
        else if (type == "ushort" || type == "uint16")  out = as(uint16_t());
        else if (type == "int"    || type == "int32")   out = as(int32_t());
        else if (type == "uint"   || type == "uint32")  out = as(uint32_t());
        else if (type == "float"  || type == "float32") out = as(float());
        else                                            out = as(double());
        return true;
    }

private:
    std::istream& in;
    Format format;
};

bool Open(const std::string& path, std::ifstream& in, Format& format,
          std::vector<Element>& elements, std::string& error) {
    in.open(path, std::ios::binary);
    if (!in) { error = "cannot open"; return false; }
    return ReadHeader(in, format, elements, error);
}

// Reads every element's data in file order, keeping the vertices (position,
// colour) and the faces. False if the file ends early.
bool ReadElements(std::istream& in, Format format, const std::vector<Element>& elements,
                  std::vector<glm::vec3>& pos, std::vector<glm::vec3>& col,
                  std::vector<std::vector<int>>& faces, bool& hasVertexColors) {
    ValueReader reader(in, format);
    for (const Element& e : elements) {
        // which property feeds which value (-1 = not present)
        int px = -1, py = -1, pz = -1, pr = -1, pg = -1, pb = -1, pface = -1;
        for (int k = 0; k < (int)e.props.size(); ++k) {
            const std::string& n = e.props[k].name;
            if (n == "x") px = k;
            else if (n == "y") py = k;
            else if (n == "z") pz = k;
            else if (n == "red"   || n == "diffuse_red")   pr = k;
            else if (n == "green" || n == "diffuse_green") pg = k;
            else if (n == "blue"  || n == "diffuse_blue")  pb = k;
            else if (n == "vertex_indices" || n == "vertex_index") pface = k;
        }
        const bool isVertex = (e.name == "vertex");
        const bool isFace   = (e.name == "face");
        const bool hasColor = isVertex && pr >= 0 && pg >= 0 && pb >= 0;
        if (hasColor) hasVertexColors = true;

        // other elements (edges, materials, ...) are read and dropped: their
        // data sits between ours, so it has to be stepped over either way
        std::vector<double> values(e.props.size());
        std::vector<int> polygon;
        for (size_t i = 0; i < e.count; ++i) {
            polygon.clear();
            for (int k = 0; k < (int)e.props.size(); ++k) {
                const Property& p = e.props[k];
                if (!p.list) {
                    if (!reader.Read(p.type, values[k])) return false;
                    continue;
                }
                double count;
                if (!reader.Read(p.countType, count)) return false;
                for (int c = 0; c < (int)count; ++c) {
                    double v;
                    if (!reader.Read(p.type, v)) return false;
                    if (isFace && k == pface) polygon.push_back((int)v);
                }
            }
            if (isVertex) {
                pos.push_back(glm::vec3(px >= 0 ? values[px] : 0.0, py >= 0 ? values[py] : 0.0,
                                        pz >= 0 ? values[pz] : 0.0));
                glm::vec3 c(1.0f);
                if (hasColor) {
                    // 0..255 for integer types, 0..1 for floats; either way a display colour
                    float scale = IsInteger(e.props[pr].type) ? 1.0f / 255.0f : 1.0f;
                    c = glm::vec3(values[pr], values[pg], values[pb]) * scale;
                }
                col.push_back(c);
            } else if (isFace && polygon.size() >= 3) {
                faces.push_back(polygon);
            }
        }
    }
    return true;
}

}   // namespace

bool PlyMeshLoader::IsMesh(const std::string& path) {
    std::ifstream in;
    Format format = Format::Ascii;
    std::vector<Element> elements;
    std::string error;
    if (!Open(path, in, format, elements, error)) return false;
    for (const Element& e : elements)
        if (e.name == "face" && e.count > 0) return true;
    return false;
}

bool PlyMeshLoader::Load(const std::string& path, MeshData& out) {
    out = MeshData();
    std::ifstream in;
    Format format = Format::Ascii;
    std::vector<Element> elements;
    std::string error;
    if (!Open(path, in, format, elements, error)) {
        std::cerr << "[PlyMeshLoader] " << path << ": " << error << "\n";
        return false;
    }

    std::vector<glm::vec3> pos, col;
    std::vector<std::vector<int>> faces;
    if (!ReadElements(in, format, elements, pos, col, faces, out.hasVertexColors)) {
        std::cerr << "[PlyMeshLoader] " << path << ": the file ends early\n";
        return false;
    }

    for (const std::vector<int>& f : faces) {
        for (size_t k = 1; k + 1 < f.size(); ++k) {   // fan: (0,1,2), (0,2,3), ...
            int a = f[0], b = f[k], c = f[k + 1];
            if (a < 0 || b < 0 || c < 0 || a >= (int)pos.size() || b >= (int)pos.size() || c >= (int)pos.size())
                continue;
            Triangle t{pos[a], pos[b], pos[c]};
            t.c0 = col[a]; t.c1 = col[b]; t.c2 = col[c];
            out.tris.push_back(t);
        }
    }
    if (out.tris.empty()) {
        std::cerr << "[PlyMeshLoader] no triangles in " << path << "\n";
        return false;
    }
    std::cout << "[PlyMeshLoader] Loaded " << out.tris.size() << " triangles"
              << (out.hasVertexColors ? " with vertex colours" : "") << " from " << path << "\n";
    return true;
}
