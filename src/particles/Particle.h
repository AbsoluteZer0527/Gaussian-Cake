#pragma once
#include "core.h"

class Particle {
public:
    glm::vec3 position;
    float radius;

    Particle(float radius = 0.1f);
};
