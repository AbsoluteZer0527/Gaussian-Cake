#include "GaussianMixture.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>

static float Dist2(const glm::vec3& a, const glm::vec3& b) {
    glm::vec3 d = a - b;
    return glm::dot(d, d);
}

void GaussianMixture::FromSingle(const Gaussian& g) {
    components = { g };
    weights    = { 1.0 };
    CacheDensityTerms();
}

void GaussianMixture::Fit(const std::vector<glm::vec3>& points, int K,
                          int iters, unsigned int seed) {
    components.clear();
    weights.clear();
    const int N = (int)points.size();
    if (N == 0) return;
    K = std::max(1, std::min(K, N));

    if (K == 1) {
        Gaussian g;
        g.FitFromPoints(points, false);
        FromSingle(g);
        return;
    }

    std::mt19937 rng(seed);

    //k-means++ seeding: next center picked with probability ~ squared distance
    std::vector<glm::vec3> centers;
    centers.reserve(K);
    centers.push_back(points[std::uniform_int_distribution<int>(0, N - 1)(rng)]);

    std::vector<float> minD2(N, std::numeric_limits<float>::max());
    while ((int)centers.size() < K) {
        double total = 0.0;
        for (int i = 0; i < N; ++i) {
            minD2[i] = std::min(minD2[i], Dist2(points[i], centers.back()));
            total += minD2[i];
        }
        double r = std::uniform_real_distribution<double>(0.0, total)(rng);
        int pick = N - 1;
        for (int i = 0; i < N; ++i) {
            r -= minD2[i];
            if (r <= 0.0) { pick = i; break; }
        }
        centers.push_back(points[pick]);
    }

    //Lloyd iterations
    std::vector<int> label(N, 0);
    for (int it = 0; it < iters; ++it) {
        bool changed = false;
        for (int i = 0; i < N; ++i) {
            int best = 0;
            float bestD = std::numeric_limits<float>::max();
            for (int k = 0; k < K; ++k) {
                float d = Dist2(points[i], centers[k]);
                if (d < bestD) { bestD = d; best = k; }
            }
            if (label[i] != best) { label[i] = best; changed = true; }
        }

        std::vector<glm::vec3> sum(K, glm::vec3(0.0f));
        std::vector<int> count(K, 0);
        for (int i = 0; i < N; ++i) { sum[label[i]] += points[i]; count[label[i]]++; }
        for (int k = 0; k < K; ++k) {
            if (count[k] > 0) {
                centers[k] = sum[k] / (float)count[k];
            } else {
                //empty cluster: re-seed at the point farthest from its center
                int far = 0; float farD = -1.0f;
                for (int i = 0; i < N; ++i) {
                    float d = Dist2(points[i], centers[label[i]]);
                    if (d > farD) { farD = d; far = i; }
                }
                centers[k] = points[far];
                changed = true;
            }
        }
        if (!changed && it > 0) break;
    }

    //one Gaussian per non-empty cluster
    std::vector<std::vector<glm::vec3>> clusters(K);
    for (int i = 0; i < N; ++i) clusters[label[i]].push_back(points[i]);

    for (int k = 0; k < K; ++k) {
        if (clusters[k].size() < 4) continue;   // too few points for a covariance
        Gaussian g;
        g.FitFromPoints(clusters[k], false);
        //ridge relative to the cluster's own size, so flat patches stay invertible
        double ridge = std::max(1e-4, 0.01 * g.covariance.trace() / 3.0);
        g.covariance += ridge * Eigen::Matrix3d::Identity();
        components.push_back(g);
        weights.push_back((double)clusters[k].size());
    }

    double wsum = 0.0;
    for (double w : weights) wsum += w;
    for (double& w : weights) w /= wsum;

    CacheDensityTerms();
}

void GaussianMixture::CacheDensityTerms() {
    invCov.resize(components.size());
    logNorm.resize(components.size());
    for (size_t k = 0; k < components.size(); ++k) {
        const Eigen::Matrix3d& S = components[k].covariance;
        invCov[k]  = S.inverse();
        logNorm[k] = std::log(weights[k]) - 0.5 * std::log(std::max(S.determinant(), 1e-300));
    }
}

void GaussianMixture::Posterior(const glm::vec3& x, float temperature,
                                std::vector<double>& out) const {
    const int K = (int)components.size();
    out.resize(K);
    if (K == 1) { out[0] = 1.0; return; }

    Eigen::Vector3d xe(x.x, x.y, x.z);
    double maxLog = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < K; ++k) {
        Eigen::Vector3d d = xe - components[k].mean;
        double lp = (logNorm[k] - 0.5 * d.dot(invCov[k] * d)) / temperature;
        out[k] = lp;
        maxLog = std::max(maxLog, lp);
    }

    //log-sum-exp normalization
    double sum = 0.0;
    for (int k = 0; k < K; ++k) { out[k] = std::exp(out[k] - maxLog); sum += out[k]; }
    for (int k = 0; k < K; ++k) out[k] /= sum;
}
