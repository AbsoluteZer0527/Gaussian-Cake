#include "OptimalTransport.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>


//public

OTMap OptimalTransport::Compute(const Gaussian& gaussianA, const Gaussian& gaussianB){
    const Eigen::Matrix3d& SigmaA = gaussianA.covariance;
    const Eigen::Matrix3d& SigmaB = gaussianB.covariance;

    //get Sigma_A^{1/2} and Sigma_A^{-1/2} using helper functions
    Eigen::Matrix3d SigmaA_sqrt    = MatrixSqrt(SigmaA);
    Eigen::Matrix3d SigmaA_invsqrt = MatrixInvSqrt(SigmaA);

    //M = Sigma_A^{1/2} * Sigma_B * Sigma_A^{1/2}
    Eigen::Matrix3d M = SigmaA_sqrt * SigmaB * SigmaA_sqrt;

    //M^{1/2}
    Eigen::Matrix3d M_sqrt = MatrixSqrt(M);

    //transpose A
    Eigen::Matrix3d A = SigmaA_invsqrt * M_sqrt * SigmaA_invsqrt;

    OTMap map;
    map.muA = gaussianA.mean;
    map.muB = gaussianB.mean;
    map.A = A;

    return map;
}

//T(x) = μ_B + A · (x − μ_A)
glm::vec3 OptimalTransport::Apply(const OTMap& map, const glm::vec3& x){
    Eigen::Vector3d xe(x.x, x.y, x.z);
    Eigen::Vector3d result = map.muB + map.A * (xe - map.muA);

    return glm::vec3((float)result(0), (float)result(1), (float)result(2));
}


double OptimalTransport::BuresW2Sq(const Gaussian& a, const Gaussian& b){
    Eigen::Matrix3d Sa_sqrt = MatrixSqrt(a.covariance);
    Eigen::Matrix3d cross   = MatrixSqrt(Sa_sqrt * b.covariance * Sa_sqrt);
    double meanTerm = (a.mean - b.mean).squaredNorm();
    double covTerm  = (a.covariance + b.covariance - 2.0 * cross).trace();
    return meanTerm + std::max(0.0, covTerm);
}

//log-sum-exp over a vector
static double LogSumExp(const Eigen::VectorXd& v){
    double m = v.maxCoeff();
    if (!std::isfinite(m)) return m;
    return m + std::log((v.array() - m).exp().sum());
}

Eigen::MatrixXd OptimalTransport::Sinkhorn(const Eigen::MatrixXd& C,
                                           const Eigen::VectorXd& a,
                                           const Eigen::VectorXd& b,
                                           double eps, int maxIters){
    const int K = (int)C.rows(), L = (int)C.cols();

    //scale eps by the median cost so the slider is shape-independent
    std::vector<double> vals(C.data(), C.data() + C.size());
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    double epsAbs = std::max(1e-12, eps * vals[vals.size() / 2]);

    Eigen::VectorXd logA = a.array().log();
    Eigen::VectorXd logB = b.array().log();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(K);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(L);
    Eigen::VectorXd tmpK(K), tmpL(L);

    auto plan = [&]() {
        Eigen::MatrixXd P(K, L);
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < L; ++j)
                P(i, j) = std::exp((f(i) + g(j) - C(i, j)) / epsAbs);
        return P;
    };

    double err = 0.0;
    int it = 0;
    for (; it < maxIters; ++it) {
        //f_i = eps log a_i - eps LSE_j((g_j - C_ij) / eps)
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < L; ++j) tmpL(j) = (g(j) - C(i, j)) / epsAbs;
            f(i) = epsAbs * (logA(i) - LogSumExp(tmpL));
        }
        //g_j = eps log b_j - eps LSE_i((f_i - C_ij) / eps)
        for (int j = 0; j < L; ++j) {
            for (int i = 0; i < K; ++i) tmpK(i) = (f(i) - C(i, j)) / epsAbs;
            g(j) = epsAbs * (logB(j) - LogSumExp(tmpK));
        }
        //after the g-update columns are exact; check the row marginal
        if (it % 10 == 9) {
            err = (plan().rowwise().sum() - a).cwiseAbs().sum();
            if (err < 1e-6) break;
        }
    }

    Eigen::MatrixXd P = plan();
    err = (P.rowwise().sum() - a).cwiseAbs().sum();
    std::cout << "[Sinkhorn] " << K << "x" << L << ", iters " << std::min(it + 1, maxIters)
              << ", marginal error " << err << "\n";
    return P;
}

MixtureOTMap OptimalTransport::ComputeMixture(const GaussianMixture& A,
                                              const GaussianMixture& B,
                                              double eps, float temperature){
    MixtureOTMap m;
    m.source = A;
    m.temperature = temperature;

    const int K = A.K(), L = B.K();
    if (K == 0 || L == 0) return m;

    Eigen::MatrixXd P;
    if (K == 1 && L == 1) {
        P = Eigen::MatrixXd::Ones(1, 1);   // single Gaussian: the original map
    } else {
        Eigen::MatrixXd C(K, L);
        for (int k = 0; k < K; ++k)
            for (int l = 0; l < L; ++l)
                C(k, l) = BuresW2Sq(A.components[k], B.components[l]);

        Eigen::VectorXd a = Eigen::Map<const Eigen::VectorXd>(A.weights.data(), K);
        Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(B.weights.data(), L);
        P = Sinkhorn(C, a, b, eps);
    }

    //keep pairs that carry mass; ApplyMixture renormalizes per component
    double pruneBelow = 1e-4 * P.maxCoeff();
    for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
            if (P(k, l) < pruneBelow) continue;
            m.srcIdx.push_back(k);
            m.dstIdx.push_back(l);
            m.coupling.push_back(P(k, l));
            m.maps.push_back(Compute(A.components[k], B.components[l]));
        }
    }
    std::cout << "[MixtureOT] K=" << K << " L=" << L
              << ", " << m.maps.size() << " pairs kept\n";
    return m;
}

void OptimalTransport::ApplyMixture(const MixtureOTMap& map, const glm::vec3& x,
                                    glm::vec3& outY, Eigen::Matrix3d& outJ){
    std::vector<double> post;
    map.source.Posterior(x, map.temperature, post);

    Eigen::Vector3d xe(x.x, x.y, x.z);
    Eigen::Vector3d y = Eigen::Vector3d::Zero();
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    double wsum = 0.0;

    for (size_t e = 0; e < map.maps.size(); ++e) {
        int k = map.srcIdx[e];
        //p_k(x) * P_kl / pi_k: share of x's mass sent from component k to l
        double w = post[k] * map.coupling[e] / map.source.weights[k];
        if (w < 1e-12) continue;
        const OTMap& T = map.maps[e];
        y    += w * (T.muB + T.A * (xe - T.muA));
        J    += w * T.A;
        wsum += w;
    }

    if (wsum < 1e-12) {   // shouldn't happen; fall back to identity
        outY = x;
        outJ = Eigen::Matrix3d::Identity();
        return;
    }
    y /= wsum;
    J /= wsum;
    outY = glm::vec3((float)y(0), (float)y(1), (float)y(2));
    outJ = J;
}

//private helpers
Eigen::Matrix3d OptimalTransport::MatrixSqrt(const Eigen::Matrix3d& M) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
    // Clamp negative eigenvalues to 0 (can occur due to float noise)
    Eigen::Vector3d sqrtVals = solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    return solver.eigenvectors() * sqrtVals.asDiagonal() * solver.eigenvectors().transpose();
}

Eigen::Matrix3d OptimalTransport::MatrixInvSqrt(const Eigen::Matrix3d& M, double epsilon) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
    Eigen::Vector3d sqrtVals = solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    // Clamp to epsilon before inverting to avoid division by near-zero
    Eigen::Vector3d invSqrtVals = sqrtVals.cwiseMax(epsilon).cwiseInverse();
    return solver.eigenvectors() * invSqrtVals.asDiagonal() * solver.eigenvectors().transpose();
}