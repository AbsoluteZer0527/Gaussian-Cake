#include "PlyLoader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

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

// ---------------------------------------------------------------------------
// YOUR TURN: ReadHeader
//
// A 3DGS .ply header looks like this (plain text, one item per line):
//
//   ply
//   format binary_little_endian 1.0
//   element vertex 1063091
//   property float x
//   property float y
//   property float z
//   property float nx
//   ...
//   property float rot_3
//   end_header
//
// Fill in `header`:
//   - header.vertexCount  from the "element vertex N" line
//   - header.offset[name] the byte offset of each property inside one record,
//                         in the order they appear (x -> 0, y -> 4, z -> 8, ...)
//   - header.stride       total bytes of one record (the sum of all sizes)
//
// Return false with a helpful `error` if:
//   - the first line isn't "ply"
//   - the format isn't binary_little_endian
//   - a property type isn't "float" (keep it simple: 3DGS files are all float)
//   - a required property is missing: x y z f_dc_0 f_dc_1 f_dc_2 opacity
//     scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
//
// Hints:
//   - std::getline(in, line) reads one line; it stops after "end_header",
//     so the stream is then exactly at the binary data.
//   - Some exporters write "\r\n" line endings: strip a trailing '\r'
//     (if (!line.empty() && line.back() == '\r') line.pop_back();).
//   - std::istringstream ss(line); ss >> word1 >> word2 >> word3; splits a
//     line on spaces.
//   - header.offset.count("x") tells you whether a name is in the map.
//   - Only properties of the "vertex" element count. (3DGS files have no
//     other elements, but checking is good practice.)
// ---------------------------------------------------------------------------
bool PlyLoader::ReadHeader(std::ifstream& in, Header& header, std::string& error) {
    error = "ReadHeader is not written yet";
    return false;
}

// ---- Below: stubs for now, we'll do these together next ----

Splat PlyLoader::Convert(const char* record, const Header& header) {
    return Splat();
}

void PlyLoader::MakeZThinnest(Splat& s) {
}

void PlyLoader::CleanAndNormalize(std::vector<Splat>& splats, const PlyLoadOptions& options) {
}
