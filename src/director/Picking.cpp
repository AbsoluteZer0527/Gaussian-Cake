#include "Picking.h"
#include <cmath>

Ray ScreenRay(float mouseX, float mouseY, float width, float height,
              const glm::mat4& view, const glm::mat4& proj) {
    float nx = 2.0f * mouseX / width - 1.0f;
    float ny = 1.0f - 2.0f * mouseY / height;
    glm::mat4 inv = glm::inverse(proj * view);
    glm::vec4 a = inv * glm::vec4(nx, ny, -1.0f, 1.0f);
    glm::vec4 b = inv * glm::vec4(nx, ny, 1.0f, 1.0f);
    glm::vec3 pa = glm::vec3(a) / a.w, pb = glm::vec3(b) / b.w;
    return { pa, glm::normalize(pb - pa) };
}

int PickPoint(const std::vector<glm::vec3>& points, const Ray& ray, float maxDist) {
    // Among points close to the ray, take the frontmost. "Close" grows a
    // little with depth isn't needed here: the shapes span ~2 units.
    int best = -1;
    float bestDepth = 1e30f;
    float max2 = maxDist * maxDist;
    for (size_t i = 0; i < points.size(); ++i) {
        glm::vec3 d = points[i] - ray.origin;
        float depth = glm::dot(d, ray.dir);
        if (depth <= 0.0f) continue;
        glm::vec3 perp = d - depth * ray.dir;
        if (glm::dot(perp, perp) > max2) continue;
        if (depth < bestDepth) { bestDepth = depth; best = (int)i; }
    }
    return best;
}
