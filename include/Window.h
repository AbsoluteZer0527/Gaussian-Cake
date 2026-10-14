#pragma once

#include "Camera.h"
#include "Shader.h"
#include "core.h"
#include "ParticleSystem.h"
#include "Gaussian.h"
#include "OptimalTransport.h"
#include "TransportAnimator.h"
#include "MeshRenderer.h"
#include "SplatRenderer.h"

#include <string>
#include <vector>

enum class RenderMode {
    TriangleMorph = 0,  // one surface morphing A -> B (shared topology or SDF)
    Particles     = 1,  // point cloud carried by Gaussian / greedy discrete OT
    Splats        = 2,  // surface-aligned 3D Gaussian splats + mixture OT
};

// How TriangleMorph builds the in-between surfaces.
enum class MorphMethod {
    Auto           = 0,  // shared topology when both meshes allow it, else SDF
    SharedTopology = 1,  // fit the denser mesh onto the other, lerp vertices
    SDF            = 2,  // OT-aligned distance-field blend, re-meshed per frame
};


class Window {
public:
    // Window Properties
    static int width;
    static int height;
    static const char* windowTitle;

    // Objects to render
    static ParticleSystem* particleSystem;
    static MeshRenderer*   meshRenderer;

    static RenderMode renderMode;

    // True morph (TriangleMorph mode): one surface, A at t = 0, B at t = 1.
    static MorphMethod morphMethod;
    static int         sdfResolution;    // SDF grid points per axis
    static float       fitStiffness;     // multiplier on the template-fit stiffness schedule
    static bool        trueMorphActive;
    static std::string morphStatus;      // which method ran, and why

    // Splat mode: one splat set travels from A's surface to B's surface.
    // splatStart[i] (on A) is matched one-to-one with splatEnd[i] (on B), so
    // both ends are fully covered without a crossfade.
    static SplatRenderer*     splatRenderer;
    static std::vector<Splat> splatStart, splatEnd, splatCurrent;
    static std::vector<glm::vec3> splatSmoothDisp;   // neighbour-averaged displacement per splat
    static float splatCohesion;      // 0 = straight lines, 1 = neighbours move as chunks
    static bool  splatMorphActive;
    static bool  splatBaking;        // a background bake is running
    static float lastBakeMs;         // duration of the last finished bake
    static int   splatCount;
    static float splatSizeMul;
    static int   mixtureK;           // Gaussian-mixture components (1 = original single map)
    static float sinkhornEps;        // relative entropic regularization
    static float posteriorTemp;      // softness of the per-splat component blend

    // Gaussian OT pipeline
    static Gaussian* gaussianA;
    static Gaussian* gaussianB;
    static OTMap* otMap;
    static TransportAnimator* animator;
    static bool particleMorphActive;   // particle transport owns the animator
    static int particleCount;

    // Mesh selection: index 0 = blob, indices 1..N = entries in meshPaths
    static std::vector<std::string> meshNames;  // display names (filenames)
    static std::vector<std::string> meshPaths;  // full relative paths
    static int meshIndexA;
    static int meshIndexB;
    static int sampleCount;       // points sampled from mesh surface

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
    static void BakeTrueMorph(const std::string& pathA, const std::string& pathB);
    static void UpdateTrueMorph();
    static void BakeSplatMorph(const std::string& pathA, const std::string& pathB);
    static void UpdateSplatPreview();
    static void PollSplatBake();
    static void DrawMainGUI();
};
