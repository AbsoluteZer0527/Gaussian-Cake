#include "MeshFitter.h"
#include "MeshSampler.h"
#include "Parallel.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_map>

using SpMat = Eigen::SparseMatrix<double>;

static Eigen::Vector3d ToE(const glm::vec3& v) { return Eigen::Vector3d(v.x, v.y, v.z); }

static uint64_t PairKey(int i, int j) {
    if (i > j) std::swap(i, j);
    return ((uint64_t)i << 32) | (uint64_t)j;
}

// Cotan weight of every edge of forms.edges on the given positions:
// w_ij = (cot alpha + cot beta) / 2, alpha and beta the angles opposite the edge.
// This is the diagonal Hodge star on 1-forms (DEC's *1). Obtuse angles can make
// a weight negative; that's fine and kept as is: d0^T W d0 is the Dirichlet
// energy of piecewise-linear functions, so it stays positive semidefinite, and
// clamping would cost the exact "linear functions have zero Laplacian" property.
// Slivers are the real danger: a near-zero-area triangle has a cot near +-1e8,
// so those faces are skipped (their edges keep the weight from the other side).
static std::vector<double> CotanWeights(const IndexedMesh& m, const DiscreteForms& forms,
                                        const std::vector<glm::vec3>& pos) {
    const int E = (int)forms.edges.size();
    std::unordered_map<uint64_t, int> edgeId;
    edgeId.reserve(E);
    for (int e = 0; e < E; ++e)
        edgeId[PairKey(forms.edges[e][0], forms.edges[e][1])] = e;

    std::vector<double> w(E, 0.0);
    for (int f = 0; f < m.FaceCount(); ++f) {
        // shape quality: 1 for equilateral, 0 for a sliver (2 area / sum of squared edges, scaled)
        glm::vec3 p0 = pos[m.indices[3 * f + 0]];
        glm::vec3 p1 = pos[m.indices[3 * f + 1]];
        glm::vec3 p2 = pos[m.indices[3 * f + 2]];
        double twiceArea = ToE(p1 - p0).cross(ToE(p2 - p0)).norm();
        double edgeSq = ToE(p1 - p0).squaredNorm() + ToE(p2 - p1).squaredNorm() + ToE(p0 - p2).squaredNorm();
        if (edgeSq <= 0.0 || 2.0 * std::sqrt(3.0) * twiceArea / edgeSq < 1e-4) continue;

        for (int k = 0; k < 3; ++k) {
            // corner c, and the edge (a, b) across from it
            int c = (int)m.indices[3 * f + k];
            int a = (int)m.indices[3 * f + (k + 1) % 3];
            int b = (int)m.indices[3 * f + (k + 2) % 3];

            Eigen::Vector3d u = ToE(pos[a] - pos[c]);
            Eigen::Vector3d v = ToE(pos[b] - pos[c]);
            double sinArea = u.cross(v).norm();   // |u||v| sin(angle), > 0 after the check above
            double cot = u.dot(v) / sinArea;      // |u||v| cos / |u||v| sin

            w[edgeId[PairKey(a, b)]] += 0.5 * cot;
        }
    }
    return w;
}

// Row-normalized Laplacian L = D^-1 C, with C = d0^T W d0 and D = diag(C).
// W = I gives the uniform Laplacian (L_ii = 1, L_ij = -1/deg); W = cotan weights
// gives the cotan Laplacian (*d*d in exterior calculus). Normalizing keeps
// L * x in length units either way, so the stiffness schedule fits both.
static SpMat NormalizedLaplacian(const DiscreteForms& forms, const std::vector<double>& w, int n) {
    const int E = (int)forms.edges.size();
    SpMat W(E, E);
    std::vector<Eigen::Triplet<double>> diag;
    diag.reserve(E);
    for (int e = 0; e < E; ++e) diag.emplace_back(e, e, w[e]);
    W.setFromTriplets(diag.begin(), diag.end());

    SpMat C = SpMat(forms.d0.transpose()) * W * forms.d0;   // n x n

    // D_ii = sum of vertex i's edge weights
    Eigen::VectorXd D = C.diagonal();

    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(C.nonZeros() + n);
    for (int col = 0; col < C.outerSize(); ++col) {
        for (SpMat::InnerIterator it(C, col); it; ++it) {
            int row = (int)it.row();
            if (D(row) > 0.0) trip.emplace_back(row, col, it.value() / D(row));
        }
    }
    for (int i = 0; i < n; ++i)
        if (D(i) <= 0.0) trip.emplace_back(i, i, 1.0);   // vertex without edges: identity row
    SpMat L(n, n);
    L.setFromTriplets(trip.begin(), trip.end());
    return L;
}

// Best rotation taking vertex i's rest one-ring edges onto its current ones,
// each edge weighted by ringW (the Laplacian's edge weights; ARAP local step).
static Eigen::Matrix3d RingRotation(int i, const std::vector<int>& ring,
                                    const std::vector<double>& ringW,
                                    const std::vector<glm::vec3>& rest,
                                    const std::vector<glm::vec3>& cur) {
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    for (size_t r = 0; r < ring.size(); ++r) {
        int j = ring[r];
        S += ringW[r] * ToE(rest[j] - rest[i]) * ToE(cur[j] - cur[i]).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0.0) {
        U.col(2) *= -1.0;
        R = V * U.transpose();
    }
    return R;
}

std::vector<glm::vec3> MeshFitter::Fit(const IndexedMesh& tmpl,
                                       const std::vector<glm::vec3>& start,
                                       const std::vector<Triangle>& target,
                                       const FitParams& params,
                                       FitStats* stats) {
    auto t0 = std::chrono::high_resolution_clock::now();
    const int n = (int)start.size();
    std::vector<glm::vec3> X = start;
    if (n == 0 || target.empty()) return X;

    SurfaceBVH bvh;
    bvh.Build(target);

    // Target samples with their face normals, for the backward (target -> template) pull.
    std::vector<int> sampleTri;
    std::vector<glm::vec3> samples = MeshSampler::Sample(target, std::min(2 * n, 60000), 11u, sampleTri);

    auto rings = MeshTopology::OneRings(tmpl);

    // Laplacian L = D^-1 d0^T W d0 and the rest differential coordinates delta = L * start.
    // W is the cotan Hodge star on the rest shape, or all ones for the uniform Laplacian.
    DiscreteForms forms = MeshTopology::BuildForms(tmpl);
    std::vector<double> edgeW = params.cotanLaplacian
        ? CotanWeights(tmpl, forms, start)
        : std::vector<double>(forms.edges.size(), 1.0);
    SpMat L = NormalizedLaplacian(forms, edgeW, n);

    // the same weights per one-ring neighbour, for the rotation fit
    std::unordered_map<uint64_t, double> weightOf;
    weightOf.reserve(forms.edges.size());
    for (size_t e = 0; e < forms.edges.size(); ++e)
        weightOf[PairKey(forms.edges[e][0], forms.edges[e][1])] = edgeW[e];
    std::vector<std::vector<double>> ringW(n);
    for (int i = 0; i < n; ++i)
        // A negative weight would make the rotation fit pull an edge the wrong
        // way, so the local step uses only the positive part (L keeps the sign).
        for (int j : rings[i]) ringW[i].push_back(std::max(0.0, weightOf[PairKey(i, j)]));

    SpMat Lt = L.transpose();
    SpMat LtL = Lt * L;

    Eigen::MatrixXd rest(n, 3);
    for (int i = 0; i < n; ++i) rest.row(i) = ToE(start[i]).transpose();
    Eigen::MatrixXd delta = L * rest;

    Eigen::SimplicialLDLT<SpMat> solver;
    bool analyzed = false;

    std::vector<glm::vec3> fwdPoint(n);
    std::vector<float>     fwdW(n);
    std::vector<Eigen::Matrix3d> R(n);

    const float cosMin = params.normalCosMin;
    const float kReg = 1e-4f;   // keeps parts without matches from drifting

    for (int s = 0; s < params.stages; ++s) {
        float f = (params.stages > 1) ? (float)s / (float)(params.stages - 1) : 1.0f;
        double alpha = params.stiffnessScale * params.stiffnessStart *
                       std::pow(params.stiffnessEnd / params.stiffnessStart, f);

        for (int it = 0; it < params.itersPerStage; ++it) {
            std::vector<glm::vec3> nrm = MeshTopology::VertexNormals(tmpl, X);

            // forward: closest point on the target for every template vertex
            ParallelFor(n, [&](int i) {
                SurfaceBVH::Hit h = bvh.Closest(X[i]);
                fwdPoint[i] = h.point;
                float c = glm::dot(nrm[i], bvh.FaceNormal(h.tri));
                fwdW[i] = (c >= cosMin) ? 1.0f : 0.0f;
            });

            // backward: every target sample pulls its nearest template vertex
            PointGrid grid;
            float spacing = 2.0f * std::sqrt(MeshSampler::TotalArea(target) / (float)samples.size());
            grid.Build(X, spacing);
            std::vector<glm::vec3> bwdSum(n, glm::vec3(0.0f));
            std::vector<int>       bwdCount(n, 0);
            for (size_t j = 0; j < samples.size(); ++j) {
                int k = grid.Nearest(samples[j]);
                if (k < 0) continue;
                if (glm::dot(nrm[k], bvh.FaceNormal(sampleTri[j])) < cosMin) continue;
                bwdSum[k] += samples[j];
                bwdCount[k]++;
            }

            // local step: per-vertex rotations of the rest Laplacian coordinates
            ParallelFor(n, [&](int i) {
                R[i] = rings[i].empty() ? Eigen::Matrix3d::Identity()
                                        : RingRotation(i, rings[i], ringW[i], start, X);
            });
            Eigen::MatrixXd rotDelta(n, 3);
            for (int i = 0; i < n; ++i)
                rotDelta.row(i) = (R[i] * delta.row(i).transpose()).transpose();

            // global step: (alpha LtL + W) X = alpha Lt (R delta) + W C
            Eigen::VectorXd W(n);
            Eigen::MatrixXd rhs = alpha * (Lt * rotDelta);
            for (int i = 0; i < n; ++i) {
                double wb = bwdCount[i] > 0 ? params.backwardWeight : 0.0;
                double w = fwdW[i] + wb + kReg;
                Eigen::Vector3d c = fwdW[i] * ToE(fwdPoint[i]) + kReg * ToE(X[i]);
                if (bwdCount[i] > 0) c += wb * ToE(bwdSum[i] / (float)bwdCount[i]);
                W(i) = w;
                rhs.row(i) += c.transpose();
            }

            // LtL always stores its diagonal, so the pattern never changes
            SpMat A = alpha * LtL;
            for (int i = 0; i < n; ++i) A.coeffRef(i, i) += W(i);
            if (!analyzed) { solver.analyzePattern(A); analyzed = true; }
            solver.factorize(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "[MeshFitter] factorization failed at stage " << s << "\n";
                break;
            }
            Eigen::MatrixXd sol = solver.solve(rhs);
            for (int i = 0; i < n; ++i)
                X[i] = glm::vec3((float)sol(i, 0), (float)sol(i, 1), (float)sol(i, 2));
        }
    }

    // residual before locking onto the surface
    std::vector<float> dist(n);
    ParallelFor(n, [&](int i) { dist[i] = std::sqrt(bvh.Closest(X[i]).dist2); });

    // final lock: exactly onto the target, then even out the triangles by
    // tangential relaxation + reprojection
    ParallelFor(n, [&](int i) { X[i] = bvh.Closest(X[i]).point; });
    for (int r = 0; r < params.relaxRounds; ++r) {
        std::vector<glm::vec3> nrm = MeshTopology::VertexNormals(tmpl, X);
        std::vector<glm::vec3> next = X;
        ParallelFor(n, [&](int i) {
            if (rings[i].empty()) return;
            glm::vec3 c(0.0f);
            for (int j : rings[i]) c += X[j];
            c /= (float)rings[i].size();
            glm::vec3 d = c - X[i];
            d -= nrm[i] * glm::dot(nrm[i], d);
            next[i] = bvh.Closest(X[i] + 0.5f * d).point;
        });
        X.swap(next);
    }

    if (stats) {
        double sum = 0.0;
        float mx = 0.0f;
        for (float d : dist) { sum += d; mx = std::max(mx, d); }
        stats->meanDist = (float)(sum / n);
        stats->maxDist  = mx;

        int flipped = 0;
        for (size_t t = 0; t + 2 < tmpl.indices.size(); t += 3) {
            unsigned int a = tmpl.indices[t], b = tmpl.indices[t + 1], c = tmpl.indices[t + 2];
            glm::vec3 n0 = glm::cross(start[b] - start[a], start[c] - start[a]);
            glm::vec3 n1 = glm::cross(X[b] - X[a], X[c] - X[a]);
            if (glm::dot(n0, n1) < 0.0f) ++flipped;
        }
        stats->flipped = flipped;
        stats->ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
    }
    return X;
}
