#include "SplatHeat.h"
#include "SurfaceQuery.h"

#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <execution>
#include <cmath>
#include <iostream>
#include <numeric>
#include <utility>

using SpMat = Eigen::SparseMatrix<double>;

static const double kHeatTolerance = 1e-4;   // CG relative residual

std::vector<int> NearestNeighbors(const std::vector<glm::vec3>& pos, int k, float cellSize) {
    const int n = (int)pos.size();
    std::vector<int> knn((size_t)n * k, -1);
    PointGrid grid;
    grid.Build(pos, cellSize);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
        std::vector<std::pair<float, int>> cand;
        grid.ForEachNear(pos[i], [&](int j) {
            if (j != i) cand.push_back({ Dist2(pos[i], pos[j]), j });
        });
        int m = std::min(k, (int)cand.size());
        std::partial_sort(cand.begin(), cand.begin() + m, cand.end());
        for (int s = 0; s < m; ++s) knn[(size_t)i * k + s] = cand[s].second;
    });
    return knn;
}

void SplatGraphLaplacian(const std::vector<glm::vec3>& pos, const std::vector<int>& knn, int k,
                         SpMat& L, Eigen::VectorXd& mass) {
    const int n = (int)pos.size();

    auto lists = [&](int i, int j) {   // does i have j as a neighbour?
        for (int s = 0; s < k; ++s) {
            if (knn[(size_t)i * k + s] == j) return true;
        }
        return false;
    };

    // Each undirected edge once. kNN isn't symmetric: j may list i or not.
    // A pair that lists each other is kept from the lower index only.
    std::vector<std::pair<int, int>> edges;
    edges.reserve((size_t)n * k);
    double sumD2 = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int s = 0; s < k; ++s) {
            int j = knn[(size_t)i * k + s];
            if (j < 0 || j == i) continue;
            if (j < i && lists(j, i)) continue;   // already added from j
            edges.push_back({ i, j });
            sumD2 += Dist2(pos[i], pos[j]);
        }
    }
    double r2 = edges.empty() ? 1.0 : sumD2 / (double)edges.size();

    // L = d0^T *1 d0: +w on both diagonals, -w off the diagonal, per edge
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(edges.size() * 4);
    mass = Eigen::VectorXd::Zero(n);
    for (const auto& edge : edges) {
        int i = edge.first;
        int j = edge.second;
        double d2 = Dist2(pos[i], pos[j]);
        double w = std::exp(-d2 / r2);
        trip.emplace_back(i, i, w);
        trip.emplace_back(j, j, w);
        trip.emplace_back(i, j, -w);
        trip.emplace_back(j, i, -w);
        mass[i] += 0.25 * w * d2;
        mass[j] += 0.25 * w * d2;
    }
    L.resize(n, n);
    L.setFromTriplets(trip.begin(), trip.end());   // duplicate entries are summed

    // A point without neighbours still needs some mass, or M + tau L is singular
    double meanMass = (n > 0) ? mass.mean() : 0.0;
    double floor = std::max(1e-6 * meanMass, 1e-12);
    mass = mass.cwiseMax(floor);
}

Eigen::MatrixXd HeatStep(const SpMat& L, const Eigen::VectorXd& mass,
                         const Eigen::MatrixXd& b, double tau) {
    const int n = (int)mass.size();

    // A = M + tau L, symmetric positive definite
    SpMat A = tau * L;
    std::vector<Eigen::Triplet<double>> diag;
    diag.reserve(n);
    for (int i = 0; i < n; ++i) diag.emplace_back(i, i, mass[i]);
    SpMat M(n, n);
    M.setFromTriplets(diag.begin(), diag.end());
    A += M;

    // Conjugate gradients, one column per thread. b itself is a good first
    // guess: heat only smooths it. A loose tolerance is plenty for smoothing.
    Eigen::MatrixXd rhs = mass.asDiagonal() * b;
    Eigen::MatrixXd x(n, b.cols());
    std::vector<int> cols(b.cols());
    std::iota(cols.begin(), cols.end(), 0);
    std::for_each(std::execution::par, cols.begin(), cols.end(), [&](int c) {
        // Each thread needs its own solver: solving writes the iteration count
        // and error into it. Setting one up is cheap (the preconditioner is
        // just A's diagonal).
        Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> cg;
        cg.setTolerance(kHeatTolerance);
        cg.compute(A);
        Eigen::VectorXd guess = b.col(c);
        x.col(c) = cg.solveWithGuess(rhs.col(c), guess);
        if (cg.info() != Eigen::Success)
            std::cerr << "[SplatHeat] heat step did not converge (error " << cg.error() << ")\n";
    });
    return x;
}

// Delay by heat on exactly these points: rank of the heat from the bottom
static std::vector<float> SolvedWaveDelay(const std::vector<glm::vec3>& pos, float cellSize,
                                          double sigma, double seedFraction) {
    const int n = (int)pos.size();
    const int k = 8;
    std::vector<int> knn = NearestNeighbors(pos, k, cellSize);
    SpMat L;
    Eigen::VectorXd mass;
    SplatGraphLaplacian(pos, knn, k, L, mass);

    // Order the points from the bottom up
    std::vector<int> byHeight(n);
    std::iota(byHeight.begin(), byHeight.end(), 0);
    std::sort(byHeight.begin(), byHeight.end(), [&](int a, int b) { return pos[a].y < pos[b].y; });

    // Heat source: the lowest points (the "pan")
    int seeds = std::max(1, (int)std::lround(seedFraction * n));
    Eigen::MatrixXd source = Eigen::MatrixXd::Zero(n, 1);
    for (int r = 0; r < seeds; ++r) source(byHeight[r], 0) = 1.0;

    Eigen::MatrixXd u = HeatStep(L, mass, source, 0.5 * sigma * sigma);

    // Warmest first. Heat below the solver's error is noise, so it counts as
    // none; ties (e.g. the far top, or a part the heat never reaches) go bottom-up.
    double maxHeat = u.col(0).maxCoeff();
    double noise = 10.0 * kHeatTolerance * maxHeat;
    std::vector<float> heat(n);
    for (int i = 0; i < n; ++i) {
        double h = u(i, 0);
        heat[i] = (h > noise) ? (float)h : 0.0f;
    }
    std::vector<int> heightRank(n);
    for (int r = 0; r < n; ++r) heightRank[byHeight[r]] = r;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (heat[a] != heat[b]) return heat[a] > heat[b];
        return heightRank[a] < heightRank[b];
    });
    std::vector<float> delay(n);
    for (int r = 0; r < n; ++r) delay[order[r]] = (float)r / (float)(n - 1);
    return delay;
}

std::vector<float> HeatWaveDelay(const std::vector<glm::vec3>& pos, float cellSize,
                                 double sigma, double seedFraction, int maxSolve) {
    const int n = (int)pos.size();
    if (n < 2) return std::vector<float>(n, 0.0f);
    if (n <= maxSolve) return SolvedWaveDelay(pos, cellSize, sigma, seedFraction);

    // Every stride-th point; they're spaced sqrt(stride) times wider apart
    const int stride = (n + maxSolve - 1) / maxSolve;
    std::vector<glm::vec3> sub;
    for (int i = 0; i < n; i += stride) sub.push_back(pos[i]);
    float subCell = cellSize * std::sqrt((float)stride);
    std::vector<float> subDelay = SolvedWaveDelay(sub, subCell, sigma, seedFraction);

    // Each point takes its nearest solved point's delay...
    PointGrid grid;
    grid.Build(sub, subCell);
    std::vector<float> coarse(n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
        coarse[i] = subDelay[grid.Nearest(pos[i])];
    });

    // ...then re-rank, so delays are evenly spread again (ties: bottom first)
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (coarse[a] != coarse[b]) return coarse[a] < coarse[b];
        return pos[a].y < pos[b].y;
    });
    std::vector<float> delay(n);
    for (int r = 0; r < n; ++r) delay[order[r]] = (float)r / (float)(n - 1);
    return delay;
}

float WaveTime(float e, float wave, float delay) {
    if (wave <= 0.0f) return e;
    float inside = std::clamp(e, 0.0f, 1.0f);
    float local = (inside - wave * delay) / (1.0f - wave);
    float overshoot = e - inside;   // easing past either end, kept as is
    return std::clamp(local, 0.0f, 1.0f) + overshoot;
}

SplatClock SplatClockAt(float e, float wave, float delay, float cohesionExp, float midGrow) {
    SplatClock clock;
    clock.e = WaveTime(e, wave, delay);
    clock.g = std::pow(std::max(clock.e, 0.0f), cohesionExp);
    float t = std::clamp(clock.e, 0.0f, 1.0f);
    float bump = 4.0f * t * (1.0f - t);   // 0 at both ends, 1 halfway
    clock.grow = 1.0f + midGrow * bump;
    clock.stretchExp = bump;
    return clock;
}
