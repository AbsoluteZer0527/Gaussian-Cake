#include "RigidInterp.h"
#include "Parallel.h"
#include <Eigen/Dense>
#include <chrono>

// The triangle's frame: two edges and a normal scaled like them, so the map
// between two frames is the triangle's own affine change (Sumner & Popovic).
static glm::mat3 Frame(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
    glm::vec3 e1 = b - a, e2 = c - a;
    glm::vec3 nrm = glm::cross(e1, e2);
    float len = glm::length(nrm);
    nrm = len > 1e-20f ? nrm / std::sqrt(len) : glm::vec3(0.0f);
    return glm::mat3(e1, e2, nrm);
}

bool RigidInterp::Build(const std::vector<glm::vec3>& s, const std::vector<glm::vec3>& e,
                        const std::vector<unsigned int>& idx) {
    ready = false;
    if (s.size() != e.size() || s.empty() || idx.size() < 3 || idx.size() % 3) return false;
    start = s;
    end = e;
    indices = idx;
    n = (int)s.size();
    const int F = (int)idx.size() / 3;
    rot.assign(F, glm::quat(1, 0, 0, 0));
    stretch.assign(F, glm::mat3(1.0f));

    ParallelFor(F, [&](int f) {
        glm::vec3 a0 = s[idx[3 * f]], b0 = s[idx[3 * f + 1]], c0 = s[idx[3 * f + 2]];
        glm::vec3 a1 = e[idx[3 * f]], b1 = e[idx[3 * f + 1]], c1 = e[idx[3 * f + 2]];
        glm::mat3 F0 = Frame(a0, b0, c0), F1 = Frame(a1, b1, c1);
        if (std::abs(glm::determinant(F0)) < 1e-24f) return;   // a sliver: leave it at identity
        glm::mat3 A = F1 * glm::inverse(F0);
        // polar decomposition A = R S via the SVD A = U D V^T: R = U V^T, S = V D V^T
        Eigen::Matrix3d M;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) M(r, c) = A[c][r];
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
        Eigen::Vector3d D = svd.singularValues();
        if ((U * V.transpose()).determinant() < 0.0) {   // a reflection: fold it into the stretch
            U.col(2) *= -1.0;
            D(2) = -D(2);
        }
        Eigen::Matrix3d R = U * V.transpose();
        Eigen::Matrix3d S = V * D.asDiagonal() * V.transpose();
        glm::mat3 Rg, Sg;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) { Rg[c][r] = (float)R(r, c); Sg[c][r] = (float)S(r, c); }
        rot[f] = glm::normalize(glm::quat_cast(Rg));
        stretch[f] = Sg;
    });

    // Normal equations of "every triangle's edges as wanted": the edge-vertex
    // incidence E (3 edges per face), M = E^T E, plus a tiny pull to the
    // origin so translation is fixed (the centroid is set after the solve).
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(12 * F + n);
    for (int f = 0; f < F; ++f)
        for (int k = 0; k < 3; ++k) {
            int i = (int)idx[3 * f + k], j = (int)idx[3 * f + (k + 1) % 3];
            trip.emplace_back(i, i, 1.0);
            trip.emplace_back(j, j, 1.0);
            trip.emplace_back(i, j, -1.0);
            trip.emplace_back(j, i, -1.0);
        }
    for (int i = 0; i < n; ++i) trip.emplace_back(i, i, 1e-8);
    Eigen::SparseMatrix<double> M(n, n);
    M.setFromTriplets(trip.begin(), trip.end());
    solver.compute(M);
    ready = solver.info() == Eigen::Success;
    return ready;
}

void RigidInterp::Evaluate(float t, std::vector<glm::vec3>& out) const {
    Evaluate(std::vector<float>(n, t), out);
}

void RigidInterp::Evaluate(const std::vector<float>& u, std::vector<glm::vec3>& out) const {
    if (!ready || (int)u.size() != n) return;
    auto t0 = std::chrono::high_resolution_clock::now();
    const int F = (int)indices.size() / 3;
    const glm::quat id(1, 0, 0, 0);

    // each face's wanted edges at its own time
    std::vector<glm::vec3> want(3 * (size_t)F);
    ParallelFor(F, [&](int f) {
        unsigned int v[3] = { indices[3 * f], indices[3 * f + 1], indices[3 * f + 2] };
        float t = (u[v[0]] + u[v[1]] + u[v[2]]) / 3.0f;
        glm::mat3 R = glm::mat3_cast(glm::slerp(id, rot[f], t));
        glm::mat3 A = R * ((1.0f - t) * glm::mat3(1.0f) + t * stretch[f]);
        for (int k = 0; k < 3; ++k)
            want[3 * f + k] = A * (start[v[(k + 1) % 3]] - start[v[k]]);
    });

    // right-hand side E^T d
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n, 3);
    for (int f = 0; f < F; ++f)
        for (int k = 0; k < 3; ++k) {
            int i = (int)indices[3 * f + k], j = (int)indices[3 * f + (k + 1) % 3];
            const glm::vec3& d = want[3 * f + k];
            rhs.row(j) += Eigen::RowVector3d(d.x, d.y, d.z);
            rhs.row(i) -= Eigen::RowVector3d(d.x, d.y, d.z);
        }
    Eigen::MatrixXd x = solver.solve(rhs);

    // put the centroid where the straight-line paths would
    glm::dvec3 cWant(0.0), cGot(0.0);
    for (int i = 0; i < n; ++i) {
        cWant += glm::dvec3(start[i] + u[i] * (end[i] - start[i]));
        cGot  += glm::dvec3(x(i, 0), x(i, 1), x(i, 2));
    }
    glm::vec3 shift = glm::vec3((cWant - cGot) / (double)n);
    out.resize(n);
    for (int i = 0; i < n; ++i)
        out[i] = glm::vec3((float)x(i, 0), (float)x(i, 1), (float)x(i, 2)) + shift;
    lastMs = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}
