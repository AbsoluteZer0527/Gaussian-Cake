#pragma once
#include "core.h"
#include <array>
#include <cmath>
#include <Eigen/Dense>

// View-dependent colour of 3D Gaussian Splatting captures: real spherical
// harmonics up to degree 3, with the constants and sign conventions of the
// original 3DGS code (so the coefficients in a .ply mean the same thing here).
//   colour(d) = 0.5 + C0 * dc + sum_{k=1..15} Y_k(d) * sh_k,   d = unit view direction
// The DC term already lives in Splat::color; Splat::sh holds k = 1..15.
namespace SH {

constexpr int kCoeffs = 15;                       // degrees 1..3, per colour channel
using Coeffs = std::array<glm::vec3, kCoeffs>;

// Number of non-DC coefficients per channel for a degree (0..3).
constexpr int CountForDegree(int degree) { return (degree + 1) * (degree + 1) - 1; }

inline Coeffs Zero() {
    Coeffs c;
    c.fill(glm::vec3(0.0f));
    return c;
}

// Y_1..Y_15 at unit direction d (degree limits how many are non-zero).
inline void Basis(const glm::vec3& d, int degree, float out[kCoeffs]) {
    const float C1 = 0.4886025119029199f;
    const float C2[5] = { 1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                          -1.0925484305920792f, 0.5462742152960396f };
    const float C3[7] = { -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                          0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                          -0.5900435899266435f };
    for (int k = 0; k < kCoeffs; ++k) out[k] = 0.0f;
    if (degree < 1) return;
    const float x = d.x, y = d.y, z = d.z;
    out[0] = -C1 * y;
    out[1] =  C1 * z;
    out[2] = -C1 * x;
    if (degree < 2) return;
    const float xx = x * x, yy = y * y, zz = z * z;
    out[3] = C2[0] * x * y;
    out[4] = C2[1] * y * z;
    out[5] = C2[2] * (2.0f * zz - xx - yy);
    out[6] = C2[3] * x * z;
    out[7] = C2[4] * (xx - yy);
    if (degree < 3) return;
    out[8]  = C3[0] * y * (3.0f * xx - yy);
    out[9]  = C3[1] * x * y * z;
    out[10] = C3[2] * y * (4.0f * zz - xx - yy);
    out[11] = C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    out[12] = C3[4] * x * (4.0f * zz - xx - yy);
    out[13] = C3[5] * z * (xx - yy);
    out[14] = C3[6] * x * (xx - 3.0f * yy);
}

// The view-dependent part of the colour (add it to the DC colour).
inline glm::vec3 Eval(const Coeffs& sh, int degree, const glm::vec3& d) {
    float y[kCoeffs];
    Basis(d, degree, y);
    glm::vec3 c(0.0f);
    const int n = CountForDegree(degree);
    for (int k = 0; k < n; ++k) c += y[k] * sh[k];
    return c;
}

inline bool IsZero(const Coeffs& sh) {
    for (const glm::vec3& c : sh)
        if (c.x != 0.0f || c.y != 0.0f || c.z != 0.0f) return false;
    return true;
}

// Coefficient matrix of a rotation R: if f(d) = sum Y_k(d) c_k, the turned
// function f'(d) = f(R^T d) has coefficients M c. Each degree is closed
// under rotations, so M is exact; it's found by least squares on a few
// dozen directions instead of Wigner-D formulas.
inline Eigen::Matrix<float, kCoeffs, kCoeffs> RotationMatrix(const glm::mat3& R) {
    const int N = 96;
    Eigen::MatrixXf Y(N, kCoeffs), Yr(N, kCoeffs);
    const float golden = 2.39996323f;   // Fibonacci sphere: evenly spread directions
    for (int i = 0; i < N; ++i) {
        float z = 1.0f - 2.0f * (i + 0.5f) / N;
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        glm::vec3 d(r * std::cos(golden * i), r * std::sin(golden * i), z);
        float a[kCoeffs], b[kCoeffs];
        Basis(d, 3, a);
        Basis(glm::transpose(R) * d, 3, b);
        for (int k = 0; k < kCoeffs; ++k) { Y(i, k) = a[k]; Yr(i, k) = b[k]; }
    }
    // Y M = Yr (as functions of d), solved column by column
    Eigen::MatrixXf M = Y.colPivHouseholderQr().solve(Yr);
    return M;
}

inline void Rotate(Coeffs& sh, const Eigen::Matrix<float, kCoeffs, kCoeffs>& M) {
    for (int ch = 0; ch < 3; ++ch) {
        Eigen::Matrix<float, kCoeffs, 1> c;
        for (int k = 0; k < kCoeffs; ++k) c[k] = sh[k][ch];
        c = M * c;
        for (int k = 0; k < kCoeffs; ++k) sh[k][ch] = c[k];
    }
}

}  // namespace SH
