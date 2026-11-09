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
#include "PlyLoader.h"
#include "GltfLoader.h"
#include "MorphDirector.h"
#include "MorphTimeline.h"

#include <string>
#include <vector>

enum class RenderMode {
    TriangleMorph = 0,  // one surface morphing A -> B (shared topology or SDF)
    Particles     = 1,  // point cloud carried by Gaussian / greedy discrete OT
    Splats        = 2,  // surface-aligned 3D Gaussian splats + mixture OT
};

// How TriangleMorph builds the in-between surfaces.
enum class MorphMethod {
    Auto           = 0,  // shared topology when both meshes allow it, else Piecewise
    SharedTopology = 1,  // fit the denser mesh onto the other, lerp vertices
    SDF            = 2,  // OT-aligned distance-field blend, re-meshed per frame
    Piecewise      = 3,  // A's mesh, then B's, following the SDF surface; switch where the topology changes
};


// How Particles mode carries the point cloud from A to B.
enum class ParticleTransport {
    Auto     = 0,  // Greedy between two shapes, Gaussian when either side is the blob
    Greedy   = 1,  // discrete matching of surface samples (exact shapes at both ends)
    Gaussian = 2,  // one closed-form Gaussian OT map, T(x) = muB + A (x - muA)
    Mixture  = 3,  // Gaussian-mixture OT: Sinkhorn plan between K parts, blended maps
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
    static bool        fitCotan;         // cotan vs uniform Laplacian in the template fit
    static int         morphMixtureK;    // parts the OT alignment matches (1 = one affine map)
    static bool        trueMorphActive;
    static bool        morphBaking;      // a background Topology bake is running
    static std::string morphStatus;      // which method ran, and why

    // GPU playback: per-frame morph work in compute shaders (OpenGL 4.3).
    // gpuAvailable is detected at startup; useGpu can be turned off to
    // compare against the CPU paths.
    static bool gpuAvailable;
    static bool useGpu;

    // Splat mode: one splat set travels from A's surface to B's surface.
    // splatStart[i] (on A) is matched one-to-one with splatEnd[i] (on B), so
    // both ends are fully covered without a crossfade.
    static SplatRenderer*     splatRenderer;
    static std::vector<Splat> splatStart, splatEnd, splatCurrent;
    static std::vector<glm::vec3> splatSmoothDisp;   // neighbour-averaged displacement per splat
    static float splatCohesion;      // 0 = straight lines, 1 = neighbours move as chunks
    static bool  splatAreaPreserve;  // size splats mid-flight by the OT map's area stretch
    static std::vector<float> splatStretch;   // per splat: that stretch at the halfway point
    static float splatWave;          // 0 = all splats leave together; up to 0.6 = they leave in a heat wave
    static std::vector<float> splatDelay;     // per splat: when the wave reaches it, in [0, 1]
    static bool  splatMorphActive;
    static bool  splatBaking;        // a background bake is running
    static float lastBakeMs;         // duration of the last finished bake
    static int   splatCount;
    static float splatSizeMul;
    static int   mixtureK;           // Gaussian-mixture components (1 = original single map)
    static float sinkhornEps;        // relative entropic regularization
    static float posteriorTemp;      // softness of the per-splat component blend
    static float colorMatch;         // 0 = match splats by position only; higher = similar colours pair up
    static PlyLoadOptions plyOptions;   // how .ply captures are cropped and turned upright

    // One bar for the costly settings (splat count, mixture K, samples, SDF
    // resolution, particles): 0 = fastest, 1 = finest. Auto fits it to the
    // chosen shapes and to how fast this machine has been.
    static float quality;
    static bool  autoQuality;

    // Gaussian OT pipeline
    static Gaussian* gaussianA;
    static Gaussian* gaussianB;
    static OTMap* otMap;
    static TransportAnimator* animator;
    static bool particleMorphActive;   // particle transport owns the animator
    static int particleCount;
    static ParticleTransport particleTransport;
    static int particleMixtureK;     // mixture components for ParticleTransport::Mixture

    // Mesh selection: index 0 = blob, indices 1..N = entries in meshPaths
    static std::vector<std::string> meshNames;  // display names (filenames)
    static std::vector<std::string> meshPaths;  // full relative paths
    static int meshIndexA;
    static int meshIndexB;
    static int sampleCount;       // points sampled from mesh surface

    // Shader Program
    static GLuint shaderProgram;
    static GLuint morphProgram;   // morph.vert + shader.frag: GPU blend of two shapes

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
    static void BakeTimeline();          // Topology: bake every segment of the timeline
    static void PollTrueMorphBake();
    static void UpdateTrueMorph(float dt = 0.0f);
    static void BakeSplatMorph(const std::string& pathA, const std::string& pathB);
    static void UpdateSplatPreview();
    static void PollSplatBake();
    static void DrawMainGUI();

    // Morph Director: the shape timeline (Topology: key 0 -> key 1 -> ...;
    // Splats: segment 0 only) with each segment's style and pins, and the
    // per-point state (origin, paint, springs) of what is playing.
    static MorphTimeline timeline;
    static MorphDirector director;
    static void DrawDirectorGUI();
    // A click of the current Director tool at window pixel (x, y), as the
    // mouse would (scripted runs use it to test picking).
    static void ToolClick(double x, double y);
    static void SelectTool(MorphDirector::Tool tool);
    static void TogglePlay();   // play / pause (bakes first if nothing is baked)
    static bool loopPlayback;

    // Set to a path: the next frame is saved there as a .ppm (scripted runs).
    static std::string screenshotPath;
    static bool hideGui;               // scripted runs: draw the scene only
    static void SaveScreenshot(const std::string& path, int w, int h);
};
