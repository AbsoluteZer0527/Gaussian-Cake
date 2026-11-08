#pragma once
#include "core.h"
#include <vector>
#include "Particle.h"
#include "Gaussian.h"

class ParticleSystem
{
public:
    ParticleSystem();

    std::vector<Particle> particles;
    float particleRadius;

    //OT transport mode — positions driven by TransportAnimator
    bool useOTPositions;
    std::vector<glm::vec3> transportPositions;

    //Replace all particles with one per point (clears OT mode)
    void SpawnAt(const std::vector<glm::vec3>& points);

    //Spawn N particles sampled from a Gaussian (clears existing particles)
    void SpawnFromGaussian(const Gaussian& g, int N, unsigned int seed = 42);

    void Update(float deltaTime);
    void Draw(glm::mat4 viewProjection);

    // Frees the GL objects (needs the context; safe to call twice).
    void ReleaseGL();

private:
    GLuint program = 0, vao = 0, vbo = 0;
    size_t vboCapacity = 0;          // particles the VBO has room for
    bool needsPointSprite = false;   // compatibility profile: see Draw
    std::vector<glm::vec3> staging;  // positions uploaded each frame
};
