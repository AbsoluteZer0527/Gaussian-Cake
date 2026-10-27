#pragma once
#include "core.h"
#include "OBJLoader.h"
#include <Eigen/Sparse>
#include <array>
#include <vector>
#include <string>

// Shared-vertex triangle mesh: verts + 3 indices per triangle.
struct IndexedMesh {
    std::vector<glm::vec3>    verts;
    std::vector<unsigned int> indices;

    int FaceCount() const { return (int)indices.size() / 3; }
};

// Connectivity summary used to decide whether two meshes can share one topology.
struct TopologyInfo {
    int verts = 0, faces = 0;
    int parts = 0;              // connected components
    int euler = 0;              // V - E + F
    int boundaryEdges = 0;      // edges used by one face (open holes)
    int boundaryLoops = 0;
    int nonManifoldEdges = 0;   // edges used by 3+ faces
    int genus = 0;              // (2 parts - euler - boundaryLoops) / 2
    bool orientable = true;     // faces can be wound consistently

    std::string Describe() const;
};

// Discrete exterior derivatives on a triangle mesh (0-forms live on vertices,
// 1-forms on edges, 2-forms on faces). d1 * d0 = 0 is the discrete d o d = 0.
struct DiscreteForms {
    std::vector<std::array<int, 2>> edges;   // edge e = (i, j) with i < j
    Eigen::SparseMatrix<double> d0;          // |E| x |V|: (d0 f)_e = f_j - f_i
    Eigen::SparseMatrix<double> d1;          // |F| x |E|: +1 if the face walks e as i -> j, -1 if j -> i
};

class MeshTopology {
public:
    // Weld a triangle soup into shared vertices. Vertices closer than weldEps
    // (after Normalize) are merged; triangles that collapse are dropped.
    static void Build(const std::vector<Triangle>& tris, IndexedMesh& out,
                      float weldEps = 1e-5f);

    static std::vector<Triangle> ToTriangles(const IndexedMesh& m);

    // Midpoint subdivision: each pass splits every triangle into 4.
    static void Subdivide(std::vector<Triangle>& tris, int levels);

    // Close every boundary loop with at most maxLoopEdges edges by a fan
    // around the loop centroid. Returns the number of holes filled.
    static int FillSmallHoles(IndexedMesh& m, int maxLoopEdges);

    // Flip all faces if the mesh's signed volume is negative, so normals
    // point outward. Returns true if the mesh was flipped.
    static bool OrientOutward(IndexedMesh& m);

    // Flip faces so every manifold edge is walked in opposite directions by
    // its two faces. Returns false if that's impossible (non-orientable).
    static bool OrientConsistently(IndexedMesh& m);

    static TopologyInfo Analyze(const IndexedMesh& m);

    static DiscreteForms BuildForms(const IndexedMesh& m);

    // Neighbouring vertex indices of every vertex.
    static std::vector<std::vector<int>> OneRings(const IndexedMesh& m);

    // Area-weighted vertex normals for the given positions (m's connectivity).
    static std::vector<glm::vec3> VertexNormals(const IndexedMesh& m,
                                                const std::vector<glm::vec3>& positions);

private:
    // Boundary loops as vertex sequences, following the faces' edge direction.
    static std::vector<std::vector<int>> BoundaryLoops(const IndexedMesh& m);
};
