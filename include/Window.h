#pragma once

#include "Camera.h"
#include "Cube.h"
#include "Shader.h"
#include "core.h"
#include "ParticleSystem.h"
#include "Gaussian.h"
#include "OptimalTransport.h"
#include "TransportAnimator.h"
#include "MeshRenderer.h"

#include <string>
#include <vector>

enum class RenderMode {
    TriangleMorph = 0,  // default: deforming triangle mesh
    Particles     = 1,  // legacy: floating dot cloud
};

// How each source vertex's endpoint is refined after the OT map.
enum class SnapMode {
    None            = 0,  // Phase 1: raw OT-warped source
    NearestSample   = 1,  // Phase 2: snap to nearest sampled point on target
    NearestTriangle = 2,  // Phase 3 (default): snap to nearest point on any target triangle
};

class Window {
public:
    // Window Properties
    static int width;
    static int height;
    static const char* windowTitle;

    // Objects to render
    static ParticleSystem* particleSystem;
    static MeshRenderer*   meshRenderer;         // source-topology mesh
    static MeshRenderer*   meshRendererTarget;   // target-topology mesh (Option B)

    // Render mode: triangle-mesh morph (default) or legacy particles
    static RenderMode renderMode;

    // Triangle-morph state: source-mesh vertex positions and OT-mapped destinations.
    // These are lerp'd each frame by idleCallback and pushed to meshRenderer.
    static std::vector<glm::vec3>    meshStartPositions;   // source per-frame buffer
    static std::vector<glm::vec3>    meshEndPositions;     // source OT-mapped destinations
    static std::vector<unsigned int> meshIndices;
    static bool                      meshMorphActive;

    // Option B (dual mesh): parallel target-topology state.
    // Target mesh animates from OT^-1(target_verts) at t=0 → target_verts at t=1,
    // in sync with the source mesh's animation timeline.
    static std::vector<glm::vec3>    meshTargetStart;      // T^-1 mapped target verts
    static std::vector<glm::vec3>    meshTargetEnd;        // actual target vertices
    static std::vector<glm::vec3>    meshTargetCurrent;    // per-frame lerp buffer
    static std::vector<unsigned int> meshTargetIndices;
    // Post-OT refinement: pull endpoints onto the actual target surface.
    // Default = NearestTriangle (Phase 3, highest quality).
    static SnapMode                  snapMode;

    // Gaussian OT pipeline
    static Gaussian* gaussianA;
    static Gaussian* gaussianB;
    static OTMap* otMap;
    static TransportAnimator* animator;
    static bool otMapComputed;
    static int particleCount;

    // Mesh selection: index 0 = blob, indices 1..N = entries in meshPaths
    static std::vector<std::string> meshNames;  // display names (filenames)
    static std::vector<std::string> meshPaths;  // full relative paths
    static int meshIndexA;
    static int meshIndexB;
    static int sampleCount;       // points sampled from mesh surface
    static int subdivisionLevel;  // midpoint subdivision passes on source mesh (0 = none)

    // Shader Program
    static GLuint shaderProgram;

    // Act as Constructors and desctructors
    static bool initializeProgram();
    static bool initializeObjects();
    static void cleanUp();

    // for the Window
    static GLFWwindow* createWindow(int width, int height);
    static void resizeCallback(GLFWwindow* window, int width, int height);

    // update and draw functions
    static void idleCallback();
    static void displayCallback(GLFWwindow*);

    // helper to reset the camera
    static void resetCamera();

    // callbacks - for interaction
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_callback(GLFWwindow* window, double currX, double currY);

    // OT pipeline
    static void ComputeOTAndSpawnParticles();
    static void DrawMainGUI();
};
