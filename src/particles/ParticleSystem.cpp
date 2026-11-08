#include "ParticleSystem.h"
#include "Shader.h"
#include <glm/gtc/type_ptr.hpp>

static const GLenum kPointSprite = 0x8861;   // GL_POINT_SPRITE (not in glad's core header)

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

// Round points with a tiny shader: positions are streamed into one VBO per
// frame (particle counts are small, and they move every frame anyway).
void ParticleSystem::Draw(glm::mat4 viewProjection) {
    if (particles.empty()) return;
    if (!program) {
        program = LoadShaders("shaders/points.vert", "shaders/points.frag");
        if (!program) return;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glBindVertexArray(0);
        // A compatibility profile (Windows, Linux) only fills gl_PointCoord
        // while point sprites are on; without it points.frag discards every
        // pixel. A core profile (macOS) always fills it and rejects the enum.
        GLint profile = 0;
        glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile);
        needsPointSprite = !(profile & GL_CONTEXT_CORE_PROFILE_BIT);
    }

    staging.resize(particles.size());
    for (size_t i = 0; i < particles.size(); ++i) staging[i] = particles[i].position;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    if (staging.size() > vboCapacity) {
        vboCapacity = staging.size();
        glBufferData(GL_ARRAY_BUFFER, vboCapacity * sizeof(glm::vec3), nullptr, GL_STREAM_DRAW);
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, staging.size() * sizeof(glm::vec3), staging.data());

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    if (needsPointSprite) glEnable(kPointSprite);
    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "viewProj"), 1, GL_FALSE, glm::value_ptr(viewProjection));
    // world-space radius -> pixel point size (rough, independent of distance)
    glUniform1f(glGetUniformLocation(program, "pointSize"), particleRadius * 100.0f);
    glUniform3f(glGetUniformLocation(program, "color"), 0.0f, 0.0f, 0.0f);

    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, (GLsizei)staging.size());
    glBindVertexArray(0);
    glDisable(GL_PROGRAM_POINT_SIZE);
    if (needsPointSprite) glDisable(kPointSprite);
}

void ParticleSystem::ReleaseGL() {
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    vbo = vao = program = 0;
    vboCapacity = 0;
}
