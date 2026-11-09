#pragma once
#include <glm/glm.hpp>
#include <vector>

struct Ray {
    glm::vec3 origin;
    glm::vec3 dir;   // unit length
};

// The ray from the camera through a window pixel (x right, y down).
Ray ScreenRay(float mouseX, float mouseY, float width, float height,
              const glm::mat4& view, const glm::mat4& proj);

// The point nearest the camera among those within maxDist of the ray (in
// front of it), or -1 if none are. Points stand for a surface (mesh vertices,
// splat centres), so the nearest one along the ray is the one you see.
int PickPoint(const std::vector<glm::vec3>& points, const Ray& ray, float maxDist);
