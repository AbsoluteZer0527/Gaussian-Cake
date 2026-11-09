#pragma once
#include "core.h"
#include "MeshTopology.h"
#include <Eigen/Sparse>
#include <cstdint>
#include <unordered_map>
#include <vector>

// Discrete Laplacians shared by the template fit (MeshFitter) and the sphere
// map (SphereMap).

using SpMat = Eigen::SparseMatrix<double>;

inline Eigen::Vector3d ToE(const glm::vec3& v) { return Eigen::Vector3d(v.x, v.y, v.z); }

// Key of an undirected edge (i, j).
inline uint64_t PairKey(int i, int j) {
    if (i > j) std::swap(i, j);
    return ((uint64_t)i << 32) | (uint64_t)j;
}

// Cotan weight of every edge of forms.edges on the given positions (DEC's *1).
std::vector<double> CotanWeights(const IndexedMesh& m, const DiscreteForms& forms,
                                 const std::vector<glm::vec3>& pos);

// The cotan Laplacian d0^T W d0 (n x n, positive semidefinite: x^T L x is the
// Dirichlet energy of the piecewise-linear function x).
SpMat CotanMatrix(const IndexedMesh& m, const std::vector<glm::vec3>& pos);

// Lumped vertex areas (a third of every face around the vertex).
std::vector<double> VertexAreas(const IndexedMesh& m, const std::vector<glm::vec3>& pos);
