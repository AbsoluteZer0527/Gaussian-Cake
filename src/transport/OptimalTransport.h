#pragma once
#include "core.h"
#include "Gaussian.h"
#include "GaussianMixture.h"
#include <vector>
#include <Eigen/Dense>


//OT map struct T(x) = muB + A * (x - muA)
struct OTMap {
    Eigen::Vector3d muA;
    Eigen::Vector3d muB;
    Eigen::Matrix3d A;    // transport matrix
};

//Mixture OT map (Delon & Desolneux 2020): a sparse K x L plan between the
//components of two Gaussian mixtures, plus the closed-form Gaussian map for
//every pair that carries mass.
struct MixtureOTMap {
    GaussianMixture source;
    std::vector<int>    srcIdx;     // component k in source
    std::vector<int>    dstIdx;     // component l in target
    std::vector<double> coupling;   // plan weight P_kl
    std::vector<OTMap>  maps;       // T_kl
    float temperature = 1.0f;       // posterior softness used by ApplyMixture
};

class OptimalTransport {
public:
    //function that computes the closed-form OT map between two Gaussians.
    //A = SigmaA^{-1/2} * (SigmaA^{1/2} * SigmaB * SigmaA^{1/2})^{1/2} * SigmaA^{-1/2}

    static OTMap Compute(const Gaussian& gaussianA, const Gaussian& gaussianB);

    //apply the map to a glm::vec3 particle position
    static glm::vec3 Apply(const OTMap& map, const glm::vec3& x);

    //squared 2-Wasserstein (Bures) distance between two Gaussians:
    //|muA - muB|^2 + tr(SigmaA + SigmaB - 2 (SigmaA^{1/2} SigmaB SigmaA^{1/2})^{1/2})
    static double BuresW2Sq(const Gaussian& a, const Gaussian& b);

    //entropic OT between histograms a (K) and b (L) with cost C (K x L), log-domain.
    //eps is relative to the median cost. Returns the K x L transport plan.
    static Eigen::MatrixXd Sinkhorn(const Eigen::MatrixXd& C,
                                    const Eigen::VectorXd& a,
                                    const Eigen::VectorXd& b,
                                    double eps, int maxIters = 2000);

    //build the mixture plan: cost = BuresW2Sq between components, plan = Sinkhorn,
    //then keep the pairs with non-negligible mass and compute their maps.
    static MixtureOTMap ComputeMixture(const GaussianMixture& A,
                                       const GaussianMixture& B,
                                       double eps, float temperature);

    //T(x) = sum_k p_k(x) sum_l (P_kl / pi_k) T_kl(x)   (Delon & Desolneux "T_mean")
    //outJ is the matching blend of the linear parts, used to carry covariances.
    static void ApplyMixture(const MixtureOTMap& map, const glm::vec3& x,
                             glm::vec3& outY, Eigen::Matrix3d& outJ);

private:
    //decompose covariance matrices since they are symmetric
    //(Av = \lambda v)
    //M = V · D · Vᵀ
    //clamps negative values to 0
    static Eigen::Matrix3d MatrixSqrt(const Eigen::Matrix3d& M);

    //symmetric matrix inverse square root
    // V * (1/sqrt(D)) * V^T
    //clamps denominator to epsilon so no /0
    static Eigen::Matrix3d MatrixInvSqrt(const Eigen::Matrix3d& M,
                                          double epsilon = 1e-9);
};