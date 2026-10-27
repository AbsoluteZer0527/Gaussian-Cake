#pragma once
#include "core.h"
#include "Splat.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

// Settings for turning a raw capture into something the morph can use.
struct PlyLoadOptions {
    // Keep splats within cropRadius x (the 90th-percentile distance from the
    // centre); everything further out is background or floaters.
    float cropRadius = 1.5f;

    // Captures from COLMAP are usually upside down (+Y points down).
    enum class Up { AsIs, FlipY, RotateXPos90, RotateXNeg90 };
    Up up = Up::FlipY;
};

// Loads a trained 3D Gaussian Splatting capture (.ply in the layout the
// original 3DGS code writes: binary little-endian, one "vertex" per splat with
// x y z, f_dc_0..2, opacity, scale_0..2, rot_0..3 and optional extras).
class PlyLoader {
public:
    // Returns false (and prints why) if the file can't be read.
    static bool Load(const std::string& path, std::vector<Splat>& outSplats,
                     const PlyLoadOptions& options = {});

    // Splats in the file before cleaning, from the header alone (0 if unreadable).
    static size_t CountSplats(const std::string& path);

private:
    // The properties every 3DGS capture has, in the order kRequired lists them.
    enum Field { X, Y, Z, DC0, DC1, DC2, Opacity, Scale0, Scale1, Scale2,
                 Rot0, Rot1, Rot2, Rot3, kFieldCount };
    static const char* const kRequired[kFieldCount];

    // What the header tells us about the binary data that follows it.
    struct Header {
        size_t vertexCount = 0;                         // number of splats
        size_t stride = 0;                              // bytes per splat record
        std::unordered_map<std::string, size_t> offset; // property name -> byte offset in a record
        size_t field[kFieldCount] = {};                 // offset of each Field, looked up once
    };

    // Reads the text header up to and including "end_header". Afterwards the
    // stream is positioned at the first byte of binary data.
    static bool ReadHeader(std::ifstream& in, Header& header, std::string& error);

    // One binary record -> one Splat (3DGS activations applied).
    static Splat Convert(const char* record, const Header& header);

    // Reorder the splat's axes so local z is the thinnest one (our convention).
    static void MakeZThinnest(Splat& s);

    // Drop invisible splats and floaters, apply the up-axis preset, then
    // centre and scale into [-1, 1]^3 like OBJLoader::Normalize.
    static void CleanAndNormalize(std::vector<Splat>& splats, const PlyLoadOptions& options);
};
