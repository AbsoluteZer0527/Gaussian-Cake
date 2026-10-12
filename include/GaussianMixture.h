#pragma once
#include "core.h"
#include "Gaussian.h"
#include <vector>
#include <Eigen/Dense>

// A K-component Gaussian mixture fit to a point cloud.
// K = 1 is exactly the single Gaussian used by the original OT map.
class GaussianMixture {
public:
    std::vector<Gaussian> components;
    std::vector<double>   weights;      // pi_k, sums to 1

    // k-means++ seeding + Lloyd iterations, then one Gaussian per cluster.
    // For K > 1 each component gets an extra ridge so thin surface patches
    // don't produce near-singular covariances (which blow up the OT maps).
    void Fit(const std::vector<glm::vec3>& points, int K,
             int iters = 20, unsigned int seed = 42);

    // Wrap a single Gaussian as a 1-component mixture.
    void FromSingle(const Gaussian& g);

    // Soft responsibilities p_k(x) = softmax_k(log(pi_k N(x | mu_k, Sigma_k)) / temperature).
    // temperature > 1 blends more smoothly across component borders.
    void Posterior(const glm::vec3& x, float temperature, std::vector<double>& out) const;

    int K() const { return (int)components.size(); }

private:
    // cached for Posterior
    std::vector<Eigen::Matrix3d> invCov;
    std::vector<double>          logNorm;   // log(pi_k) - 0.5 * log det Sigma_k

    void CacheDensityTerms();
};
