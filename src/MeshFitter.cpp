#include "MeshFitter.h"
#include "MeshSampler.h"
#include "Parallel.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>

using SpMat = Eigen::SparseMatrix<double>;

static Eigen::Vector3d ToE(const glm::vec3& v) { return Eigen::Vector3d(v.x, v.y, v.z); }

// Best rotation taking vertex i's rest one-ring edges onto its current ones.
static Eigen::Matrix3d RingRotation(int i, const std::vector<int>& ring,
                                    const std::vector<glm::vec3>& rest,
                                    const std::vector<glm::vec3>& cur) {
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    for (int j : ring)
        S += ToE(rest[j] - rest[i]) * ToE(cur[j] - cur[i]).transpose();
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

    // Uniform Laplacian L and the rest differential coordinates delta = L * start.
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(n * 8);
    for (int i = 0; i < n; ++i) {
        trip.emplace_back(i, i, 1.0);
        double w = rings[i].empty() ? 0.0 : 1.0 / (double)rings[i].size();
        for (int j : rings[i]) trip.emplace_back(i, j, -w);
    }
    SpMat L(n, n);
    L.setFromTriplets(trip.begin(), trip.end());
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
                                        : RingRotation(i, rings[i], start, X);
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
