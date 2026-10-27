#include "ParticleSystem.h"
#include <glm/gtc/type_ptr.hpp>

ParticleSystem::ParticleSystem() {
    particleRadius = 0.03f;
    useOTPositions = false;
}

void ParticleSystem::SpawnAt(const std::vector<glm::vec3>& points) {
    particles.clear();
    for (const auto& pos : points) {
        Particle p(particleRadius);
        p.position = pos;
        particles.push_back(p);
    }
    transportPositions = points;
    useOTPositions = false;
}

void ParticleSystem::SpawnFromGaussian(const Gaussian& g, int N, unsigned int seed) {
    SpawnAt(g.Sample(N, seed));
}

void ParticleSystem::Update(float deltaTime) {
    if (!useOTPositions) return;
    for (size_t i = 0; i < particles.size() && i < transportPositions.size(); i++) {
        particles[i].position = transportPositions[i];
    }
}

// Fixed-function points; the projection matrix stays identity, so the whole
// view-projection goes on the modelview stack.
void ParticleSystem::Draw(glm::mat4 viewProjection) {
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(glm::value_ptr(viewProjection));

    glColor3f(0.0f, 0.0f, 0.0f);

    // world-space radius -> pixel point size (rough, independent of distance)
    glPointSize(particleRadius * 100.0f);

    glBegin(GL_POINTS);
    for (const Particle& p : particles) {
        glVertex3f(p.position.x, p.position.y, p.position.z);
    }
    glEnd();
}