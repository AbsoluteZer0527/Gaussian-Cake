#include "MeshSampler.h"
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

//get triangle Area
float MeshSampler:: TriangleArea(const Triangle& t){
    glm::vec3 edge1 = t.v1 - t.v0;
    glm::vec3 edge2 = t.v2 - t.v0;
    return 0.5f * glm::length(glm::cross(edge1, edge2));
}

float MeshSampler::TotalArea(const std::vector<Triangle>& triangles){
    float total = 0.0f;
    for (const auto& t : triangles) total += TriangleArea(t);
    return total;
}

std::vector<glm::vec3> MeshSampler::Sample(const std::vector<Triangle>& triangles, int N, unsigned int seed){
    std::vector<int> unused;
    return Sample(triangles, N, seed, unused);
}

std::vector<glm::vec3> MeshSampler::Sample(const std::vector<Triangle>& triangles, int N, unsigned int seed,
                                           std::vector<int>& outTriIndices){
    std::vector<glm::vec3> result;
    outTriIndices.clear();
    if (triangles.empty() || N <= 0) return result;

    //get all the triangle areas.
    std::vector<float> areas(triangles.size());
    for (int i = 0; i < (int)triangles.size(); i++){
        areas[i] = TriangleArea(triangles[i]);
    }
        
    //get cumultive sum then normamlize
    std:: vector<float> cdf(triangles.size());
    std::partial_sum(areas.begin(), areas.end(), cdf.begin());
    float totalArea = cdf.back();
    if (totalArea < 1e-9f) {
        std::cerr << "[MeshSampler] Mesh has near-zero total area.\n";
        return result;
    }
    for (float& v : cdf) v /= totalArea;

    //sample N points
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    result.reserve(N);
    outTriIndices.reserve(N);
    for (int i = 0; i < N; i++) {
        // Pick a triangle via CDF
        float r = dist(rng);
        int triIdx = (int)(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
        triIdx = std::min(triIdx, (int)triangles.size() - 1);
        const Triangle& tri = triangles[triIdx];

        // Uniform point inside triangle using the barycentric method
        //point = v0 + s*(v1 - v0) + t*(v2 - v0)
        float u1 = dist(rng);
        float u2 = dist(rng);
        float s = 1.0f - std::sqrt(1.0f - u1);
        float t = u2 * std::sqrt(1.0f - u1);
        glm::vec3 point = (1.0f - s - t) * tri.v0 + s * tri.v1 + t * tri.v2;
        result.push_back(point);
        outTriIndices.push_back(triIdx);
    }
    return result;
}