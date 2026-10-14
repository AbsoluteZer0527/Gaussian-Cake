#include "Gaussian.h"
#include <random>
#include <iostream>

//posterchild guassian :)
Gaussian::Gaussian() {
    mean = Eigen::Vector3d::Zero();
    covariance = Eigen::Matrix3d::Identity();
}

void Gaussian::FitFromPoints(const std::vector<glm::vec3>& points, bool verbose){
    if(points.empty()) return;

    int N = (int)points.size();

    //get mean
    Eigen::Vector3d mu = Eigen::Vector3d::Zero();
    for (const auto& p: points){
        mu += Eigen::Vector3d(p.x, p.y, p.z);
    }
    
    mean = mu /= N;

    //get covariance

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : points) {
        Eigen::Vector3d d = Eigen::Vector3d(p.x, p.y, p.z) - mu;
        cov += d * d.transpose();   // outer product
    }

    cov /= N;

    // Ridge: prevents singular covariance for flat/degenerate point clouds
    cov += 1e-6 * Eigen::Matrix3d::Identity();
    covariance = cov;

    if (!verbose) return;

    //testing
    std::cout << "[Gaussian] Mean: ("
              << mean.x() << ", " << mean.y() << ", " << mean.z() << ")\n";
    std::cout << "[Gaussian] Cov trace: " << covariance.trace() << "\n";
}

std::vector<glm::vec3> Gaussian::Sample(int N, unsigned int seed) const {
    std::vector<glm::vec3> result;
    result.reserve(N);

    // Cholesky decomposition: covariance = L * L^T
    // LDLT handles positive semi-definite (not just positive definite)

    Eigen::LDLT<Eigen::Matrix3d> ldlt(covariance);

    //edge case
    if (ldlt.info() != Eigen::Success) {
        std::cerr << "[Gaussian] LDLT decomposition failed.\n";
        return result;
    }

    //get L from LDLT factors: P^T * L * D * L^T * P = covariance
    //L * sqrt(D)
    Eigen::Matrix3d L = ldlt.matrixL();
    Eigen::Vector3d d = ldlt.vectorD();
    
    //edge cae: tiny negative value from floating point
    for (int i = 0; i < 3; i++){
        d(i) = std::sqrt(std::max(0.0, d(i)));
    }
    
    Eigen::Matrix3d sqrtCov = ldlt.transpositionsP().transpose()* (L * d.asDiagonal());
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    for (int i = 0; i < N; i++){
        Eigen::Vector3d z(norm(rng), norm(rng), norm(rng));
        Eigen::Vector3d x = mean + sqrtCov * z;

        result.push_back(glm::vec3((float)x(0), (float)x(1), (float)x(2)));
    }

    return result;
}