#pragma once
#include "core.h"
#include <Eigen/Sparse>
#include <glm/gtc/quaternion.hpp>
#include <vector>

// As-rigid-as-possible shape interpolation (Alexa, Cohen-Or & Levin 2000;
// Xu et al. 2005, "Poisson shape interpolation"): in-between shapes for two
// meshes with the same triangles.
//
// A straight line per vertex (start + t (end - start)) shrinks and folds
// parts that turn: an arm swinging 180 degrees passes through its own
// shoulder. Instead, every triangle's change from start to end is split into
// a rotation R and a stretch S (A = R S, polar decomposition), and at time t
// the triangle is turned by slerp(I, R, t) and stretched by (1 - t) I + t S.
// The triangles no longer fit together exactly, so the vertices are the
// least-squares fit to all of them (one sparse solve, factored once).
// Exact at both ends; each triangle can run on its own time (the Morph
// Director's per-vertex times).
class RigidInterp {
public:
    // false if the meshes don't match or the system can't be factored.
    bool Build(const std::vector<glm::vec3>& start, const std::vector<glm::vec3>& end,
               const std::vector<unsigned int>& indices);
    bool Ready() const { return ready; }
    void Clear() { ready = false; }

    // The shape when vertex i is at time u[i] (a triangle takes the mean of
    // its corners'). Times may leave [0, 1] (anticipation / overshoot).
    void Evaluate(const std::vector<float>& u, std::vector<glm::vec3>& out) const;
    // Every vertex at time t.
    void Evaluate(float t, std::vector<glm::vec3>& out) const;

    mutable double lastMs = 0.0;   // the last Evaluate

private:
    bool ready = false;
    int n = 0;
    std::vector<glm::vec3> start, end;
    std::vector<unsigned int> indices;
    std::vector<glm::quat> rot;      // per face: R
    std::vector<glm::mat3> stretch;  // per face: S
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
};
