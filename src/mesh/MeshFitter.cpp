#include "MeshFitter.h"
#include "MeshSampler.h"
#include "Parallel.h"
#include "Laplacian.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_map>

using SpMat = Eigen::SparseMatrix<double>;

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

// Faces whose normal turned more than 90 degrees from the rest shape's: folded over.
int MeshFitter::CountFlips(const IndexedMesh& tmpl, const std::vector<glm::vec3>& rest,
                           const std::vector<glm::vec3>& X, std::vector<int>* flippedFaces) {
    int flipped = 0;
    for (size_t t = 0; t + 2 < tmpl.indices.size(); t += 3) {
        unsigned int a = tmpl.indices[t], b = tmpl.indices[t + 1], c = tmpl.indices[t + 2];
        glm::vec3 n0 = glm::cross(rest[b] - rest[a], rest[c] - rest[a]);
        glm::vec3 n1 = glm::cross(X[b] - X[a], X[c] - X[a]);
        if (glm::dot(n0, n1) < 0.0f) {
            ++flipped;
            if (flippedFaces) flippedFaces->push_back((int)(t / 3));
        }
    }
    return flipped;
}

// Faces that face against the target surface they lie on (dot of the face
// normal with the target's normal at the face centre < 0): visible folds.
int MeshFitter::CountInverted(const IndexedMesh& tmpl, const std::vector<glm::vec3>& X, const SurfaceBVH& bvh,
                              std::vector<int>* faces) {
    const int F = tmpl.FaceCount();
    std::vector<char> bad(F, 0);
    ParallelFor(F, [&](int f) {
        glm::vec3 a = X[tmpl.indices[3 * f]], b = X[tmpl.indices[3 * f + 1]], c = X[tmpl.indices[3 * f + 2]];
        glm::vec3 n = glm::cross(b - a, c - a);
        if (glm::dot(n, n) < 1e-20f) return;
        SurfaceBVH::Hit h = bvh.Closest((a + b + c) / 3.0f);
        if (h.tri >= 0 && glm::dot(n, bvh.FaceNormal(h.tri)) < 0.0f) bad[f] = 1;
    });
    int count = 0;
    for (int f = 0; f < F; ++f) {
        if (!bad[f]) continue;
        ++count;
        if (faces) faces->push_back(f);
    }
    return count;
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

    // pinned vertices: pulled to their target only (not to the closest point)
    std::vector<int> pinOf(n, -1);
    for (size_t k = 0; k < params.pins.size(); ++k) {
        int v = params.pins[k].first;
        if (v >= 0 && v < n) pinOf[v] = (int)k;
    }

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

            // far matches (beyond farFactor x the median match distance) pull across gaps: drop them
            if (params.farFactor > 0.0f) {
                std::vector<float> d2(n);
                for (int i = 0; i < n; ++i) d2[i] = Dist2(X[i], fwdPoint[i]);
                std::vector<float> sorted = d2;
                std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.end());
                float lim = params.farFactor * params.farFactor * std::max(sorted[n / 2], 1e-10f);
                for (int i = 0; i < n; ++i) if (d2[i] > lim) fwdW[i] = 0.0f;
            }

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
            // constantWeights: every vertex keeps the same weights all stage
            // long (a match that's dropped pulls it to where it already is),
            // so the matrix only changes with alpha: one factorization per
            // stage instead of one per iteration.
            Eigen::VectorXd W(n);
            Eigen::MatrixXd rhs = alpha * (Lt * rotDelta);
            for (int i = 0; i < n; ++i) {
                double w;
                Eigen::Vector3d c;
                if (pinOf[i] >= 0) {
                    w = params.pinWeight + kReg;
                    c = params.pinWeight * ToE(params.pins[pinOf[i]].second) + kReg * ToE(X[i]);
                } else if (params.constantWeights) {
                    double wb = params.backwardWeight;
                    w = 1.0 + wb + kReg;
                    c = ToE(fwdW[i] > 0.0f ? fwdPoint[i] : X[i]) + kReg * ToE(X[i]);
                    c += wb * ToE(bwdCount[i] > 0 ? bwdSum[i] / (float)bwdCount[i] : X[i]);
                } else {
                    double wb = bwdCount[i] > 0 ? params.backwardWeight : 0.0;
                    w = fwdW[i] + wb + kReg;
                    c = fwdW[i] * ToE(fwdPoint[i]) + kReg * ToE(X[i]);
                    if (bwdCount[i] > 0) c += wb * ToE(bwdSum[i] / (float)bwdCount[i]);
                }
                W(i) = w;
                rhs.row(i) += c.transpose();
            }

            // LtL always stores its diagonal, so the pattern never changes
            if (!params.constantWeights || it == 0) {
                SpMat A = alpha * LtL;
                for (int i = 0; i < n; ++i) A.coeffRef(i, i) += W(i);
                if (!analyzed) { solver.analyzePattern(A); analyzed = true; }
                solver.factorize(A);
                if (solver.info() != Eigen::Success) {
                    std::cerr << "[MeshFitter] factorization failed at stage " << s << "\n";
                    break;
                }
            }
            Eigen::MatrixXd sol = solver.solve(rhs);
            for (int i = 0; i < n; ++i)
                X[i] = glm::vec3((float)sol(i, 0), (float)sol(i, 1), (float)sol(i, 2));
        }
    }

    if (stats) stats->flippedBeforeLock = CountFlips(tmpl, start, X);

    // residual before locking onto the surface
    std::vector<float> dist(n);
    ParallelFor(n, [&](int i) { dist[i] = std::sqrt(bvh.Closest(X[i]).dist2); });

    float pinSum = 0.0f;
    int pinCount = 0;
    for (int i = 0; i < n; ++i) {
        if (pinOf[i] < 0) continue;
        pinSum += glm::length(X[i] - params.pins[pinOf[i]].second);
        ++pinCount;
    }

    // final lock: exactly onto the target, then even out the triangles by
    // tangential relaxation + reprojection (pins land exactly on their points)
    ParallelFor(n, [&](int i) {
        X[i] = pinOf[i] >= 0 ? params.pins[pinOf[i]].second : bvh.Closest(X[i]).point;
    });
    if (stats) stats->flippedAfterLock = CountFlips(tmpl, start, X);
    for (int r = 0; r < params.relaxRounds; ++r) {
        std::vector<glm::vec3> nrm = MeshTopology::VertexNormals(tmpl, X);
        std::vector<glm::vec3> next = X;
        ParallelFor(n, [&](int i) {
            if (rings[i].empty() || pinOf[i] >= 0) return;
            glm::vec3 c(0.0f);
            for (int j : rings[i]) c += X[j];
            c /= (float)rings[i].size();
            glm::vec3 d = c - X[i];
            d -= nrm[i] * glm::dot(nrm[i], d);
            next[i] = bvh.Closest(X[i] + 0.5f * d).point;
        });
        X.swap(next);
    }

    // Untangle: smooth the folded regions over the surface (each vertex
    // toward its one-ring centre, within the target's tangent plane, then
    // back onto the target). A round is kept only if it leaves fewer folds;
    // otherwise the next one tries a wider region with a smaller step.
    if (stats) stats->invertedBeforeUntangle = CountInverted(tmpl, X, bvh);
    {
        std::vector<int> faces;
        int bad = CountInverted(tmpl, X, bvh, &faces);
        int grow = 0;
        float step = 1.0f;
        for (int r = 0; r < params.untangleRounds && bad > 0; ++r) {
            std::vector<char> move(n, 0);
            for (int f : faces)
                for (int k = 0; k < 3; ++k) move[tmpl.indices[3 * f + k]] = 1;
            for (int g = 0; g < grow; ++g) {   // the ring around them too
                std::vector<char> wider = move;
                for (int i = 0; i < n; ++i)
                    if (move[i]) for (int j : rings[i]) wider[j] = 1;
                move.swap(wider);
            }
            std::vector<glm::vec3> next = X;
            ParallelFor(n, [&](int i) {
                if (!move[i] || rings[i].empty() || pinOf[i] >= 0) return;
                glm::vec3 c(0.0f);
                for (int j : rings[i]) c += X[j];
                c /= (float)rings[i].size();
                SurfaceBVH::Hit h = bvh.Closest(X[i]);
                glm::vec3 nt = bvh.FaceNormal(h.tri);
                glm::vec3 d = c - X[i];
                d -= nt * glm::dot(nt, d);   // stay on the surface's tangent plane
                next[i] = bvh.Closest(X[i] + step * d).point;
            });
            std::vector<int> nextFaces;
            int nextBad = CountInverted(tmpl, next, bvh, &nextFaces);
            if (nextBad < bad) {
                X.swap(next);
                faces.swap(nextFaces);
                bad = nextBad;
            } else {   // no better: look wider, move less
                grow = std::min(grow + 1, 3);
                step *= 0.6f;
                if (step < 0.1f) break;
            }
        }
    }

    if (stats) {
        stats->inverted = CountInverted(tmpl, X, bvh);
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
        stats->pinError = pinCount ? pinSum / pinCount : 0.0f;
        stats->ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
    }
    return X;
}
