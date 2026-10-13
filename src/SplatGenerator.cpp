#include "SplatGenerator.h"
#include "MeshSampler.h"
#include <cmath>

// same gray as the triangle-morph meshes
glm::vec3 SplatGenerator::baseColor = glm::vec3(0.72f, 0.72f, 0.74f);

float SplatGenerator::BaseSigma(const std::vector<Triangle>& tris, int N) {
    if (N <= 0) return 0.01f;
    return std::sqrt(MeshSampler::TotalArea(tris) / (float)N);
}

glm::vec3 SplatGenerator::TriangleNormal(const Triangle& t) {
    glm::vec3 n = glm::cross(t.v1 - t.v0, t.v2 - t.v0);
    float len = glm::length(n);
    return (len > 1e-12f) ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
}

glm::quat SplatGenerator::TriangleFrame(const Triangle& t) {
    glm::vec3 n = TriangleNormal(t);

    //tangent = first edge, re-orthogonalized against the normal
    glm::vec3 t1 = t.v1 - t.v0;
    t1 -= n * glm::dot(n, t1);
    if (glm::length(t1) < 1e-12f) {
        //degenerate edge: pick any vector perpendicular to n
        t1 = (std::abs(n.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        t1 -= n * glm::dot(n, t1);
    }
    t1 = glm::normalize(t1);
    glm::vec3 t2 = glm::cross(n, t1);   // right-handed: t1 x t2 = n

    return glm::normalize(glm::quat_cast(glm::mat3(t1, t2, n)));
}

glm::vec3 SplatGenerator::ShadeNormal(const glm::vec3& n) {
    //matches shaders/shader.frag: ambient 0.2, white light, sqrt as cheap gamma
    const glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 5.0f, 2.0f));
    float irradiance = 0.2f + std::abs(glm::dot(lightDir, n));
    return glm::sqrt(irradiance * baseColor);
}

std::vector<Splat> SplatGenerator::FromTriangles(const std::vector<Triangle>& tris,
                                                 int N, float sizeMul, unsigned int seed) {
    std::vector<Splat> result;
    if (tris.empty() || N <= 0) return result;

    std::vector<int> triIdx;
    auto pts = MeshSampler::Sample(tris, N, seed, triIdx);

    float sigma = sizeMul * BaseSigma(tris, N);

    result.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        const Triangle& t = tris[triIdx[i]];
        Splat& s = result[i];
        s.mean    = pts[i];
        s.rot     = TriangleFrame(t);
        s.scale   = glm::vec3(sigma, sigma, 0.05f * sigma);  // flat disk
        s.color   = ShadeNormal(TriangleNormal(t));
        s.opacity = 0.9f;
    }
    return result;
}
