#pragma once
#include "core.h"
#include <Eigen/Sparse>
#include <vector>

// Heat flow on the splats' k-nearest-neighbour graph, built like DEC on a
// mesh (CSE 270, Discrete Exterior Calculus): the graph's edges are the 1-cells,
// so L = d0^T *1 d0 with edge conductivities *1, and each splat's mass is *0.
// Heat then spreads as  *0 du/dt = -L u.

// The k nearest other points of every point (k entries each, -1 = none),
// searched within the 3x3x3 grid cells of size cellSize around it.
std::vector<int> NearestNeighbors(const std::vector<glm::vec3>& pos, int k, float cellSize);

// The graph Laplacian and masses for points `pos` and their neighbour lists
// `knn` (k entries per point, -1 = none). Edge weights w_ij = exp(-|xi - xj|^2 / r^2),
// r^2 = mean squared edge length. Mass m_i = 1/4 sum_j w_ij |xi - xj|^2 makes
// M^-1 L approximate the surface Laplacian at any splat spacing, so one heat
// time spreads over the same distance at 5k splats as at 500k.
void SplatGraphLaplacian(const std::vector<glm::vec3>& pos, const std::vector<int>& knn, int k,
                         Eigen::SparseMatrix<double>& L, Eigen::VectorXd& mass);

// One backward-Euler heat step, (M + tau L) x = M b, for every column of b.
// Stable for any tau; heat spreads about sigma = sqrt(2 tau) in that time.
Eigen::MatrixXd HeatStep(const Eigen::SparseMatrix<double>& L, const Eigen::VectorXd& mass,
                         const Eigen::MatrixXd& b, double tau);

// Each point's departure delay in [0, 1] for a wave of heat spreading from the
// lowest seedFraction of the points (lowest y): the warmest point leaves at 0,
// the coldest at 1, evenly spaced by rank in between. The heat is solved on
// at most maxSolve of the points (a long heat time on many points is slow);
// the rest take the delay of their nearest solved point. cellSize: the
// neighbour search cell for all n points (it grows for the sparser subset).
std::vector<float> HeatWaveDelay(const std::vector<glm::vec3>& pos, float cellSize,
                                 double sigma, double seedFraction, int maxSolve = 20000);

// A splat's own time on the timeline e when it leaves after `delay` of a wave
// of width `wave` (0 = everyone leaves together, then this returns e). Exact
// at the ends (e = 0 -> 0, e = 1 -> 1); easing overshoot past either end is kept.
float WaveTime(float e, float wave, float delay);

// Everything one splat needs from the timeline this frame. The CPU loop in
// Window::idleCallback uses it, and splat_prep.comp repeats it line for line.
struct SplatClock {
    float e;            // its own eased time (WaveTime)
    float g;            // cohesion: how much of its own correction is applied, e^cohesionExp
    float grow;         // mid-flight size factor, 1 + midGrow 4e(1-e)
    float stretchExp;   // exponent of its area stretch, 4e(1-e)
};
SplatClock SplatClockAt(float e, float wave, float delay, float cohesionExp, float midGrow);
