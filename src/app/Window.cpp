#include "Window.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "OBJLoader.h"
#include "MeshSampler.h"
#include "SplatGenerator.h"
#include "GaussianMixture.h"
#include "MeshTopology.h"
#include "SurfaceQuery.h"
#include "MeshFitter.h"
#include "SDFMorph.h"
#include "GpuSDFMorph.h"
#include "GpuTimer.h"
#include "PerfModel.h"
#include "SurfaceColor.h"
#include "PlyMeshLoader.h"
#include "TexturedMeshRenderer.h"

#include <glm/gtc/quaternion.hpp>
#include <chrono>
#include <atomic>
#include <execution>
#include <future>
#include <mutex>
#include <array>
#include <numeric>
#include <cmath>

#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <random>

//Window Properties
int Window::width;
int Window::height;
const char* Window::windowTitle = "Gaussian Cake";

//Camera Properties
Camera* Cam;

//Object particles
ParticleSystem* Window::particleSystem = nullptr;
MeshRenderer*   Window::meshRenderer   = nullptr;

//Render mode
RenderMode Window::renderMode = RenderMode::Particles;

//True morph state
MorphMethod Window::morphMethod     = MorphMethod::Auto;
int         Window::sdfResolution   = 64;
float       Window::fitStiffness    = 1.0f;
bool        Window::fitCotan        = true;
bool        Window::trueMorphActive = false;
bool        Window::morphBaking     = false;
std::string Window::morphStatus;

//GPU playback
bool Window::gpuAvailable = false;
bool Window::useGpu       = false;

// Real endpoint meshes (shown exactly at t = 0 and t = 1) and the in-between data.
static IndexedMesh               g_morphA, g_morphB;
// Per-vertex colours (all empty when neither mesh has colours)
static std::vector<glm::vec3>    g_colorsA, g_colorsB;                   // of g_morphA / g_morphB
static std::vector<glm::vec3>    g_sharedColorStart, g_sharedColorEnd;   // shared topology ends
static std::vector<glm::vec3>    g_sharedColorCurrent;                   // CPU path only
// The coloured originals, drawn with their real textures while the timeline
// rests at A or B (in-between surfaces only have vertex colours)
static MeshData                  g_dataA, g_dataB;
static TexturedMeshRenderer*     g_restMesh = nullptr;
static bool                      g_restTextured = false;   // g_restMesh is what's shown at rest
static bool                      g_morphShared = false;   // shared topology vs SDF
static std::vector<glm::vec3>    g_sharedStart, g_sharedEnd;
static std::vector<glm::vec3>    g_sharedCurrent;         // CPU path only
static std::vector<unsigned int> g_sharedIndices;
static SDFMorph                  g_sdfMorph;
static GpuSDFMorph*              g_gpuSdf   = nullptr;    // GPU copy of g_sdfMorph
static bool                      g_sdfOnGpu = false;      // the in-between SDF surface lives on the GPU
enum class MorphShown { None, A, Blend, B };
static MorphShown                g_morphShown = MorphShown::None;
static float                     g_lastSdfT   = -1.0f;

// Performance readout: smoothed CPU times (ms) and GPU timers
static double    g_cpuUpdateMs = -1.0;   // idleCallback
static double    g_cpuDrawMs   = -1.0;   // submitting this frame's scene draw
static GpuTimer* g_morphTimer  = nullptr;   // SDF extraction on the GPU
static GpuTimer* g_drawTimer   = nullptr;   // the scene draw (except GPU splats, timed by the renderer)
static std::string g_verifyText;

static void SmoothMs(double& avg, double sample) {
    avg = (avg < 0.0) ? sample : 0.9 * avg + 0.1 * sample;
}

static double MsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Splat attributes for this frame on the GPU path (computed in idleCallback)
static float g_splatE = 0.0f, g_splatG = 0.0f, g_splatGrow = 1.0f, g_splatStretchOn = 1.0f;

// Full-resolution captures on the GPU: [0] = A, [1] = B. Shown while the
// timeline rests at that end (the morph itself moves a subsample), and [0]
// also holds the static preview of a capture source.
static SplatRenderer* g_fullSplats[2] = { nullptr, nullptr };
static bool g_previewOnGpu = false;   // the splat preview lives in g_fullSplats[0]

static bool SplatsOnGpu() {
    return Window::useGpu && Window::gpuAvailable && Window::splatRenderer->HasKeyframes();
}

// The GPU splat set to draw this frame, or nullptr for the CPU path.
static SplatRenderer* GpuSplatSet() {
    if (!Window::useGpu || !Window::gpuAvailable) return nullptr;
    if (!Window::splatMorphActive) return g_previewOnGpu ? g_fullSplats[0] : nullptr;
    if (!Window::animator->IsTransporting()) {
        int end = (Window::animator->state == AnimationState::GAUSSIAN_B) ? 1 : 0;
        if (g_fullSplats[end]->HasKeyframes()) return g_fullSplats[end];
    }
    return SplatsOnGpu() ? Window::splatRenderer : nullptr;
}

//Splat mode state
SplatRenderer*     Window::splatRenderer = nullptr;
std::vector<Splat> Window::splatStart, Window::splatEnd, Window::splatCurrent;
bool  Window::splatMorphActive = false;
bool  Window::splatBaking      = false;
float Window::lastBakeMs       = -1.0f;
int   Window::splatCount       = 20000;
float Window::splatSizeMul     = 1.0f;
int   Window::mixtureK         = 32;
float Window::sinkhornEps      = 0.05f;
float Window::posteriorTemp    = 1.0f;
float Window::colorMatch       = 0.0f;
PlyLoadOptions Window::plyOptions;
float Window::splatCohesion    = 0.6f;
bool  Window::splatAreaPreserve = true;
std::vector<float> Window::splatStretch;
std::vector<glm::vec3> Window::splatSmoothDisp;
static int splatPreviewGen     = 0;   // bump to force the static preview to rebuild

//Gaussian OT pipeline
Gaussian*        Window::gaussianA     = nullptr;
Gaussian*        Window::gaussianB     = nullptr;
OTMap*           Window::otMap         = nullptr;
TransportAnimator* Window::animator    = nullptr;
bool             Window::particleMorphActive = false;
int              Window::particleCount = 3000;
std::vector<std::string> Window::meshNames;
std::vector<std::string> Window::meshPaths;
int              Window::meshIndexA    = 0;
int              Window::meshIndexB    = 0;
int              Window::sampleCount   = 5000;
float            Window::quality       = 0.5f;
bool             Window::autoQuality   = true;

//Interaction Variables
bool LeftDown, RightDown;
int MouseX, MouseY;

//The shader program id
GLuint Window::shaderProgram;
GLuint Window::morphProgram = 0;

// Default blob: isotropic unit 3D Gaussian at the origin
static void SetBlob(Gaussian& g) {
    g.mean       = Eigen::Vector3d::Zero();
    g.covariance = Eigen::Matrix3d::Identity();
}

// Load an OBJ and normalize it to [-1, 1]^3. Returns false on failure.
// A .ply without faces is a 3D Gaussian Splatting capture (Splats mode only);
// with faces it's a mesh like .obj / .glb. Asked every frame and from the bake
// thread, so the header check is cached behind a mutex.
static bool IsCapture(const std::string& path) {
    if (std::filesystem::path(path).extension() != ".ply") return false;
    static std::unordered_map<std::string, bool> known;
    static std::mutex knownMutex;
    std::lock_guard<std::mutex> lock(knownMutex);
    auto it = known.find(path);
    if (it == known.end()) it = known.emplace(path, !PlyMeshLoader::IsMesh(path)).first;
    return it->second;
}

// Any mesh file (.obj, .glb, .gltf, .ply with faces), with its appearance.
static bool LoadNormalized(const std::string& path, MeshData& mesh) {
    auto ext = std::filesystem::path(path).extension();
    bool ok = (ext == ".glb" || ext == ".gltf") ? GltfLoader::Load(path, mesh)
            : (ext == ".ply")                   ? PlyMeshLoader::Load(path, mesh)
                                                : OBJLoader::Load(path, mesh);
    if (!ok) return false;
    OBJLoader::Normalize(mesh.tris);
    return true;
}

// Geometry only, for the modes that don't use colour yet.
static bool LoadNormalized(const std::string& path, std::vector<Triangle>& tris) {
    MeshData mesh;
    if (!LoadNormalized(path, mesh)) return false;
    tris = std::move(mesh.tris);
    return true;
}

// ---- Quality bar ----
// Every costly setting, as a function of one quality q in [0, 1].
struct QualitySettings {
    int splatCount;
    int mixtureK;
    int sampleCount;
    int sdfResolution;
    int particleCount;
};

static const int    kSplatFloor    = 5000;     // splats at q = 0
static const int    kSplatMax      = 500000;   // the most the GPU path is built for
static const double kBakeBudgetMs  = 6000.0;   // Auto keeps the bake under this...
static const double kFrameBudgetMs = 12.0;     // ...and a GPU splat frame under this

static PerfModel g_perf;                // learned bake and frame costs
static bool      g_refitQuality = true; // Auto should pick again (new shapes, mode or costs)

// The settings at quality q (0 = fastest, 1 = finest). Blended in log space:
// cost grows with each value, so every step of the bar multiplies it by the
// same factor (5k..500k gives 50k at q = 0.5, not 252k).
static QualitySettings SettingsFor(float q, int splatCeiling) {
    auto logLerp = [q](int lo, int hi) {
        return (int)std::lround(lo * std::pow((double)hi / lo, (double)q));
    };
    return { logLerp(kSplatFloor, splatCeiling),
             logLerp(16, 64),
             logLerp(4000, 20000),
             logLerp(32, 160),
             logLerp(500, 10000) };
}

// A capture as loaded with these options (cleaning and cropping depend on them)
static std::string CaptureKey(const std::string& path, const PlyLoadOptions& ply) {
    return path + "|" + std::to_string(ply.cropRadius) + "|" + std::to_string((int)ply.up);
}

// Splats each capture kept after cleaning and cropping, once the preview or a
// bake has loaded it (main thread only).
static std::unordered_map<std::string, size_t> g_cleanedCounts;

static void RecordCleanedCount(const std::string& key, size_t count) {
    auto it = g_cleanedCounts.find(key);
    if (it != g_cleanedCounts.end() && it->second == count) return;
    g_cleanedCounts[key] = count;
    g_refitQuality = true;   // the ceiling changed
}

// The splat count both shapes can supply: a capture can't give more than it
// has. The header count is only an upper bound (cleaning drops splats), so
// the cleaned count wins once it's known.
static int SplatCeiling() {
    static std::unordered_map<std::string, size_t> headerCounts;   // header reads, once per file
    int ceiling = kSplatMax;
    for (int idx : { Window::meshIndexA, Window::meshIndexB }) {
        if (idx <= 0 || idx > (int)Window::meshPaths.size()) continue;
        const std::string& path = Window::meshPaths[idx - 1];
        if (!IsCapture(path)) continue;
        size_t have = 0;
        auto cleaned = g_cleanedCounts.find(CaptureKey(path, Window::plyOptions));
        if (cleaned != g_cleanedCounts.end()) {
            have = cleaned->second;
        } else {
            auto it = headerCounts.find(path);
            if (it == headerCounts.end()) it = headerCounts.emplace(path, PlyLoader::CountSplats(path)).first;
            have = it->second;
        }
        if (have > 0) ceiling = std::min(ceiling, (int)have);
    }
    return std::max(ceiling, kSplatFloor);
}

static QualitySettings CurrentSettings() {
    return { Window::splatCount, Window::mixtureK, Window::sampleCount,
             Window::sdfResolution, Window::particleCount };
}

static void ApplySettings(const QualitySettings& s) {
    Window::splatCount    = s.splatCount;
    Window::mixtureK      = s.mixtureK;
    Window::sampleCount   = s.sampleCount;
    Window::sdfResolution = s.sdfResolution;
    Window::particleCount = s.particleCount;
}

// Do this mode's settings still match the bar, or were they edited by hand?
static bool MatchesQuality(const QualitySettings& s) {
    QualitySettings c = CurrentSettings();
    switch (Window::renderMode) {
        case RenderMode::Splats:
            return c.splatCount == s.splatCount && c.mixtureK == s.mixtureK && c.sampleCount == s.sampleCount;
        case RenderMode::TriangleMorph:
            return c.sdfResolution == s.sdfResolution;
        case RenderMode::Particles:
            return c.particleCount == s.particleCount;
    }
    return true;
}

// Highest quality whose splat count fits both the shapes and the time budget.
// Searches q itself, so it works for whatever mapping SettingsFor uses.
static float FitQuality(int ceiling) {
    int target = g_perf.MaxCount(kSplatFloor, ceiling, kBakeBudgetMs, kFrameBudgetMs);
    if (SettingsFor(1.0f, ceiling).splatCount <= target) return 1.0f;
    float lo = 0.0f, hi = 1.0f;
    for (int i = 0; i < 20; ++i) {
        float mid = 0.5f * (lo + hi);
        if (SettingsFor(mid, ceiling).splatCount <= target) lo = mid;
        else                                                 hi = mid;
    }
    return lo;
}

// Auto re-picks only when something it depends on changed, never per frame:
// a new splat count rebuilds the preview.
static void UpdateQuality() {
    static int lastA = -1, lastB = -1;
    static RenderMode lastMode = RenderMode::Particles;
    static bool lastAuto = false;
    if (lastA != Window::meshIndexA || lastB != Window::meshIndexB ||
        lastMode != Window::renderMode || lastAuto != Window::autoQuality) {
        lastA = Window::meshIndexA; lastB = Window::meshIndexB;
        lastMode = Window::renderMode; lastAuto = Window::autoQuality;
        g_refitQuality = true;
    }
    if (!Window::autoQuality || !g_refitQuality) return;
    g_refitQuality = false;

    // only splats have a learned cost; the other modes take the middle
    int ceiling = SplatCeiling();
    Window::quality = (Window::renderMode == RenderMode::Splats) ? FitQuality(ceiling) : 0.5f;
    ApplySettings(SettingsFor(Window::quality, ceiling));
}

// ---- Static previews for Particles / Topology modes ----
// Until a morph is baked, Topology shows mesh A as a solid surface and
// Particles shows particles sampled on A (or the default blob). Rebuilt only
// when the source, particle count, mode or g_previewGen changes.
static MeshRenderer* g_previewMesh = nullptr;
static TexturedMeshRenderer* g_previewTextured = nullptr;   // the preview of a coloured mesh
static bool g_previewIsTextured = false;
static int g_previewGen = 0;   // bump to force the preview to rebuild

static void UpdateModePreview() {
    static int lastIndex = -1, lastCount = -1, lastMode = -1, lastGen = -1;
    int mode = (int)Window::renderMode;
    if (lastIndex == Window::meshIndexA && lastCount == Window::particleCount &&
        lastMode == mode && lastGen == g_previewGen)
        return;
    lastIndex = Window::meshIndexA; lastCount = Window::particleCount;
    lastMode = mode; lastGen = g_previewGen;

    // keep the loaded source so changing the particle count doesn't reload it
    static MeshData mesh;
    const std::vector<Triangle>& tris = mesh.tris;
    static int loadedIndex = -1;
    int idx = Window::meshIndexA;
    bool hasMesh = idx > 0 && idx <= (int)Window::meshPaths.size() &&
                   !IsCapture(Window::meshPaths[idx - 1]);   // captures: Splats mode only
    if (hasMesh && loadedIndex != idx) {
        if (LoadNormalized(Window::meshPaths[idx - 1], mesh)) {
            loadedIndex = idx;
        } else {
            hasMesh = false;
            loadedIndex = -1;
        }
    }

    if (Window::renderMode == RenderMode::TriangleMorph && hasMesh) {
        IndexedMesh m;
        MeshTopology::Build(tris, m);
        if (!g_previewMesh) {
            g_previewMesh = new MeshRenderer();
            g_previewMesh->diffuseColor = Window::meshRenderer->diffuseColor;
        }
        g_previewMesh->alpha = 1.0f;
        g_previewMesh->Initialize(m.verts, m.indices);
        g_previewIsTextured = mesh.HasAppearance();
        if (g_previewIsTextured) {
            if (!g_previewTextured) g_previewTextured = new TexturedMeshRenderer();
            g_previewTextured->Initialize(mesh);
        }
        return;
    }

    if (hasMesh) {
        // same samples the mesh-to-mesh particle bake starts from
        Window::particleSystem->SpawnAt(MeshSampler::Sample(tris, Window::particleCount));
    } else {
        SetBlob(*Window::gaussianA);
        Window::particleSystem->SpawnFromGaussian(*Window::gaussianA, Window::particleCount);
    }
}


// Scan the meshes/ directory and populate Window::meshNames / meshPaths.
// Index 0 is always the "blob" sentinel (no path).
static void ScanMeshes() {
    Window::meshNames.clear();
    Window::meshPaths.clear();

    namespace fs = std::filesystem;
    std::string meshDir = "meshes";

    if (fs::is_directory(meshDir)) {
        std::vector<std::string> found;
        for (const auto& entry : fs::directory_iterator(meshDir)) {
            auto ext = entry.path().extension();
            if (entry.is_regular_file() &&
                (ext == ".obj" || ext == ".ply" || ext == ".glb" || ext == ".gltf")) {
                found.push_back(entry.path().string());
            }
        }
        std::sort(found.begin(), found.end());
        for (const auto& p : found) {
            Window::meshPaths.push_back(p);
            Window::meshNames.push_back(fs::path(p).filename().string());
        }
    }
}

// Constructors and desctructors
bool Window::initializeProgram() {
    shaderProgram = LoadShaders("shaders/shader.vert", "shaders/shader.frag");
    morphProgram  = LoadShaders("shaders/morph.vert", "shaders/shader.frag");

    if (!shaderProgram || !morphProgram) {
        std::cerr << "Failed to initialize shader program" << std::endl;
        return false;
    }

    return true;
}

// Find the index of a mesh filename in meshNames (returns 0 if not found → "(blob)").
// meshIndex 0 is the blob sentinel, so real meshes are at index (position_in_meshNames + 1).
static int FindMeshIndex(const std::string& filename) {
    for (size_t i = 0; i < Window::meshNames.size(); ++i) {
        if (Window::meshNames[i] == filename) return (int)i + 1;
    }
    return 0;
}

bool Window::initializeObjects() {
    particleSystem = new ParticleSystem();
    meshRenderer   = new MeshRenderer();
    splatRenderer  = new SplatRenderer();
    if (!splatRenderer->Initialize()) return false;
    for (SplatRenderer*& r : g_fullSplats) {
        r = new SplatRenderer();
        if (!r->Initialize()) return false;
    }

    // GPU playback needs compute shaders (OpenGL 4.3); without them every
    // mode stays on its CPU path.
    g_gpuSdf      = new GpuSDFMorph();
    g_morphTimer  = new GpuTimer();
    g_drawTimer   = new GpuTimer();
    gpuAvailable  = GpuComputeAvailable() && g_gpuSdf->Init() && splatRenderer->InitializeGpu();
    useGpu        = gpuAvailable;
    std::cout << "[GPU] compute playback "
              << (gpuAvailable ? "available" : "unavailable (needs OpenGL 4.3), using the CPU paths") << "\n";

    gaussianA = new Gaussian();
    gaussianB = new Gaussian();
    animator  = new TransportAnimator();

    ScanMeshes();

    // Defaults: source = Guassian.obj, target = teapot.obj
    meshIndexA = FindMeshIndex("Guassian.obj");
    meshIndexB = FindMeshIndex("teapot.obj");

    Cam->SetDistance(8.0f);
    Cam->SetIncline(-20.0f);

    // Spawn initial blob so the scene isn't empty on startup
    SetBlob(*gaussianA);
    particleSystem->SpawnFromGaussian(*gaussianA, particleCount);

    return true;
}

void Window::cleanUp() {
    delete particleSystem;
    delete meshRenderer;
    delete splatRenderer;
    for (SplatRenderer* r : g_fullSplats) delete r;
    delete g_previewMesh;
    delete g_previewTextured;
    delete g_restMesh;
    delete g_gpuSdf;
    delete g_morphTimer;
    delete g_drawTimer;
    delete gaussianA;
    delete gaussianB;
    delete otMap;
    delete animator;
    glDeleteProgram(shaderProgram);
    glDeleteProgram(morphProgram);
}

//window settings
GLFWwindow* Window::createWindow(int width, int height) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return NULL;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(width, height, windowTitle, NULL, NULL);

    if (!window) {
        std::cerr << "Failed to open GLFW window." << std::endl;
        glfwTerminate();
        return NULL;
    }

    glfwMakeContextCurrent(window);
    glewInit();
    glfwSwapInterval(0);

    Cam = new Camera();
    Cam->SetAspect(float(width) / float(height));

    LeftDown = RightDown = false;
    MouseX = MouseY = 0;

    Window::resizeCallback(window, width, height);

    return window;
}

void Window::resizeCallback(GLFWwindow* window, int width, int height) {
    // minimized: a zero size would give an infinite aspect ratio
    if (width <= 0 || height <= 0) return;
    Window::width = width;
    Window::height = height;
    glViewport(0, 0, width, height);
    Cam->SetAspect(float(width) / float(height));
}

//update and draw functions
void Window::idleCallback() {
    Cam->Update();

    static double lastTime = 0.0;
    double currentTime = glfwGetTime();
    float deltaTime = (lastTime == 0.0) ? 0.016f : (float)(currentTime - lastTime);
    lastTime = currentTime;

    deltaTime = glm::min(deltaTime, 0.05f);

    auto cpuStart = std::chrono::steady_clock::now();
    PollSplatBake();
    PollTrueMorphBake();

    if (splatMorphActive) {
        // Animator only drives the timeline here; splat attributes are
        // interpolated below from the baked start/end keyframes.
        static std::vector<glm::vec3> unused;
        animator->Update(deltaTime, unused);

        float e = animator->GetBlend();

        // Splats spread apart while travelling; grow them a little mid-flight
        // (up to +30% at the halfway point, exact size at both ends).
        const float kMidGrow = 0.3f;
        float tc = glm::clamp(e, 0.0f, 1.0f);
        float grow = 1.0f + kMidGrow * 4.0f * tc * (1.0f - tc);

        // Cohesive path: follow the neighbour-smoothed displacement, and add
        // each splat's own correction late (g grows slowly, then reaches 1).
        // cohesion 0 -> g = e -> plain straight lines.
        float g = std::pow(std::max(e, 0.0f), 1.0f + 4.0f * splatCohesion);
        bool smooth = splatSmoothDisp.size() == splatCurrent.size();

        // GPU path: splat_prep.comp runs the loop below for every splat
        g_splatE = e;
        g_splatG = g;
        g_splatGrow = grow;
        size_t cpuCount = SplatsOnGpu() ? 0 : splatCurrent.size();

        // Area-preserving size: each splat's tangent size also follows the OT
        // map's own area stretch, s_i^(4t(1-t)): 1 at both ends, s_i halfway.
        bool stretch = splatAreaPreserve && splatStretch.size() == splatCurrent.size();
        g_splatStretchOn = splatAreaPreserve ? 1.0f : 0.0f;
        float stretchExp = 4.0f * tc * (1.0f - tc);

        for (size_t i = 0; i < cpuCount; ++i) {
            const Splat& a = splatStart[i];
            const Splat& b = splatEnd[i];
            Splat& s = splatCurrent[i];
            if (smooth) {
                glm::vec3 d = b.mean - a.mean;
                s.mean = a.mean + e * splatSmoothDisp[i] + g * (d - splatSmoothDisp[i]);
            } else {
                s.mean = glm::mix(a.mean, b.mean, e);
            }
            s.rot     = glm::slerp(a.rot, b.rot, e);   // shortest path
            s.scale   = grow * glm::exp(glm::mix(glm::log(a.scale), glm::log(b.scale), e));
            if (stretch) {
                float k = std::pow(splatStretch[i], stretchExp);
                s.scale.x *= k;
                s.scale.y *= k;   // z is the thin axis
            }
            s.color   = glm::mix(a.color, b.color, e);
            s.opacity = glm::mix(a.opacity, b.opacity, e);
        }
    } else if (trueMorphActive) {
        static std::vector<glm::vec3> unused;
        animator->Update(deltaTime, unused);
        UpdateTrueMorph();
    } else if (particleMorphActive) {
        animator->Update(deltaTime, particleSystem->transportPositions);
    }

    // after the animator, so particles show this frame's transport positions
    particleSystem->Update(deltaTime);
    SmoothMs(g_cpuUpdateMs, MsSince(cpuStart));
}

void Window::displayCallback(GLFWwindow* window) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto cpuStart = std::chrono::steady_clock::now();
    UpdateQuality();
    if (renderMode == RenderMode::Splats && !splatMorphActive) UpdateSplatPreview();

    // GPU splats time their own passes (timer queries can't nest)
    SplatRenderer* gpuSet = (renderMode == RenderMode::Splats) ? GpuSplatSet() : nullptr;
    if (!gpuSet) g_drawTimer->Begin();

    if (renderMode == RenderMode::Splats) {
        if (gpuSet == splatRenderer)
            splatRenderer->DrawMorph(g_splatE, g_splatG, g_splatGrow, g_splatStretchOn, Cam->GetViewMtx(), Cam->GetProjectMtx());
        else if (gpuSet)   // a full capture: start == end, so the time doesn't matter.
                           // The preview is uploaded unscaled, so Splat Size is applied here.
            gpuSet->DrawMorph(0.0f, 0.0f, splatMorphActive ? 1.0f : splatSizeMul, 0.0f,
                              Cam->GetViewMtx(), Cam->GetProjectMtx());
        else
            splatRenderer->Draw(splatCurrent, Cam->GetViewMtx(), Cam->GetProjectMtx());
    } else if (renderMode == RenderMode::TriangleMorph && trueMorphActive) {
        if (g_sdfOnGpu && g_morphShown == MorphShown::Blend)
            g_gpuSdf->Draw(shaderProgram, Cam->GetViewProjectMtx(), meshRenderer->diffuseColor);
        else if (g_restTextured && g_morphShown != MorphShown::Blend)
            g_restMesh->Draw(shaderProgram, Cam->GetViewProjectMtx());
        else
            meshRenderer->Draw(meshRenderer->IsMorphTarget() ? morphProgram : shaderProgram,
                               Cam->GetViewProjectMtx());
    } else if (renderMode == RenderMode::TriangleMorph && meshIndexA > 0) {
        // Topology preview: the source mesh as a solid surface
        UpdateModePreview();
        if (g_previewIsTextured && g_previewTextured)
            g_previewTextured->Draw(shaderProgram, Cam->GetViewProjectMtx());
        else if (g_previewMesh && g_previewMesh->isInitialized)
            g_previewMesh->Draw(shaderProgram, Cam->GetViewProjectMtx());
    } else {
        if (!particleMorphActive) UpdateModePreview();
        particleSystem->Draw(Cam->GetViewProjectMtx());
    }

    if (!gpuSet) g_drawTimer->End();
    SmoothMs(g_cpuDrawMs, MsSince(cpuStart));

    // Teach the quality bar how fast this GPU handles splats (once a second;
    // the timers are already smoothed)
    static double lastObserved = 0.0;
    if (gpuSet && glfwGetTime() - lastObserved > 1.0 &&
        gpuSet->PrepSortMs() >= 0.0 && gpuSet->DrawMs() >= 0.0) {
        lastObserved = glfwGetTime();
        g_perf.ObserveFrame(gpuSet->Count(), gpuSet->PrepSortMs() + gpuSet->DrawMs());
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    Window::DrawMainGUI();

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwPollEvents();
    glfwSwapBuffers(window);
}

// OT pipeline

// Load an OBJ and fit a Gaussian to its surface. Returns false on failure.
static bool LoadMeshGaussian(const std::string& path, int N, Gaussian* g) {
    std::vector<Triangle> tris;
    if (!LoadNormalized(path, tris)) return false;
    g->FitFromPoints(MeshSampler::Sample(tris, N));
    return true;
}

// Greedy approximate discrete OT: match each point in ptsA to the nearest
// unmatched point in ptsB. O(N^2) — fine for N <= 2000 on a single Bake click.
static std::vector<glm::vec3> GreedyDiscreteOT(
    const std::vector<glm::vec3>& ptsA,
    const std::vector<glm::vec3>& ptsB)
{
    int N = (int)std::min(ptsA.size(), ptsB.size());
    std::vector<bool>    used(ptsB.size(), false);
    std::vector<glm::vec3> result(N);

    for (int i = 0; i < N; i++) {
        float bestDist = 1e30f;
        int   bestJ    = 0;
        for (int j = 0; j < (int)ptsB.size(); j++) {
            if (used[j]) continue;
            glm::vec3 d = ptsA[i] - ptsB[j];
            float dist = glm::dot(d, d); // squared distance — avoids sqrt
            if (dist < bestDist) { bestDist = dist; bestJ = j; }
        }
        result[i]   = ptsB[bestJ];
        used[bestJ] = true;
    }
    return result;
}

// Rotation with local z = n (flipped to face the same side as ref's z) and
// local x as close as possible to ref's x, so the splat spins as little as
// possible on its way between the two frames.
static glm::quat FrameFacing(glm::vec3 n, const glm::quat& ref) {
    glm::vec3 refZ = ref * glm::vec3(0, 0, 1);
    glm::vec3 refX = ref * glm::vec3(1, 0, 0);
    if (glm::dot(n, refZ) < 0.0f) n = -n;

    glm::vec3 t1 = refX - n * glm::dot(n, refX);
    if (glm::length(t1) < 1e-6f) {
        t1 = (std::abs(n.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        t1 -= n * glm::dot(n, t1);
    }
    t1 = glm::normalize(t1);
    glm::vec3 t2 = glm::cross(n, t1);
    return glm::normalize(glm::quat_cast(glm::mat3(t1, t2, n)));
}

// ---- Splat bake: runs on a background thread, reports progress ----
//
// One splat set travels the whole way: A's splats start exactly on A and each
// one lands on a *distinct* splat of B, so both ends are fully covered and no
// crossfade is needed. Both meshes get the same splat count; each mesh's splat
// size comes from its own surface area.

static const char* kBakeStageNames[] = {
    "Loading meshes", "Fitting mixtures", "Solving OT plan",
    "Mapping splats", "Matching splats", "Smoothing paths", "Done"
};
static const char* kMorphStageNames[] = {
    "Loading meshes", "Fitting Gaussians", "Building the morph", "Done"
};
// Progress labels of the bake that is running (splat or topology)
static const char* const* g_stageNames = kBakeStageNames;
static int g_stageCount = 6;
static const int kSplatNeighbors = 8;   // k for the cohesion smoothing graph
static std::atomic<int> g_bakeStage{0};
static std::atomic<int> g_bakeDone{0};
static std::atomic<int> g_bakeTotal{1};

static void SetBakeStage(int stage, int total) {
    g_bakeDone  = 0;
    g_bakeTotal = std::max(1, total);
    g_bakeStage = stage;
}

// Everything the bake needs, copied when Bake is clicked so GUI changes
// during the bake can't affect it.
struct SplatBakeSettings {
    std::string pathA, pathB;
    int   splatCount, sampleCount, mixtureK;
    float sizeMul, sinkhornEps, posteriorTemp;
    float colorMatch;
    PlyLoadOptions ply;
};

struct SplatBakeResult {
    bool ok = false;
    std::vector<Splat> start, end;   // start[i] travels to end[i]
    std::vector<glm::vec3> smoothDisp;   // neighbour-averaged end - start
    std::vector<float> stretch;      // per splat: the OT map's extra size factor halfway (MidwayStretch)
    double detMin = 0.0, detMed = 0.0, detMax = 0.0;   // det J of the map over the splats
    std::vector<int> knn;            // kSplatNeighbors nearest A-neighbours per splat (-1 = none)
    std::vector<Splat> restA, restB; // full captures shown at rest (empty when nothing was dropped)
    size_t cleanedA = 0, cleanedB = 0; // splats each capture kept after cleaning (0 = a mesh)
    double ms = 0.0;
    double loadMs = 0.0;             // part of ms spent reading the files
    double meanMatchDist = 0.0;      // rms |T(a_i) - b_match(i)| after refinement
    double nearestBound  = 0.0;      // rms distance to the nearest b (no one-to-one)
    std::string pathA, pathB;        // what was baked, for RecordCleanedCount
    PlyLoadOptions ply;
};

// Per-shape work that doesn't depend on the other shape, kept between bakes.
// A shape is a mesh (.obj -> tris) or a capture (.ply -> splats).
// The bake and the capture preview both run on workers and share it, so each
// holds g_meshCacheMutex while it uses the cache.
struct SplatMeshCache {
    MeshData              mesh;          // meshes
    std::vector<Splat>    splats;        // captures
    bool  capture  = false;
    float cropUsed = -1.0f;              // plyOptions the splats were loaded with
    int   upUsed   = -1;
    int mixK = -1, mixSamples = -1;      // settings the mixture was fit with
    GaussianMixture mix;
};
static std::unordered_map<std::string, SplatMeshCache> g_meshCache;
static std::mutex g_meshCacheMutex;

static SplatMeshCache* GetMeshCache(const std::string& path, const PlyLoadOptions& ply) {
    SplatMeshCache& c = g_meshCache[path];   // an empty entry the first time
    if (IsCapture(path)) {
        if (c.capture && c.cropUsed == ply.cropRadius && c.upUsed == (int)ply.up) return &c;
        c = SplatMeshCache();                // new crop or up axis: reload everything
        c.capture = true;
        if (!PlyLoader::Load(path, c.splats, ply)) {
            g_meshCache.erase(path);
            return nullptr;
        }
        c.cropUsed = ply.cropRadius;
        c.upUsed   = (int)ply.up;
        return &c;
    }
    if (c.mesh.tris.empty() && !LoadNormalized(path, c.mesh)) {
        g_meshCache.erase(path);
        return nullptr;
    }
    return &c;
}

// How much of the shape a capture splat stands for: opacity x footprint.
static float SplatWeight(const Splat& s) {
    return s.opacity * s.scale.x * s.scale.y;
}

// Points spread over a capture the way MeshSampler spreads them over a mesh:
// pick splats in proportion to their weight, then a random point inside each
// picked splat's Gaussian.
static std::vector<glm::vec3> SampleSplats(const std::vector<Splat>& splats, int count,
                                           unsigned int seed) {
    std::vector<double> w(splats.size());
    for (size_t i = 0; i < splats.size(); ++i) w[i] = SplatWeight(splats[i]);
    std::discrete_distribution<size_t> pick(w.begin(), w.end());
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::mt19937 rng(seed);

    std::vector<glm::vec3> pts(count);
    for (glm::vec3& p : pts) {
        const Splat& s = splats[pick(rng)];
        glm::vec3 local(gauss(rng), gauss(rng), gauss(rng));
        p = s.mean + s.rot * (s.scale * local);
    }
    return pts;
}

// n of a capture's m splats, without replacement, in proportion to their
// weight (weighted reservoir sampling: the n smallest -log(u) / w win). The
// kept splats widen by sqrt(m / n), so n of them still cover what m did, but
// at most 2x: captures are many splats thick, so they keep covering anyway,
// and wider splats only blur.
static std::vector<Splat> SubsampleSplats(const std::vector<Splat>& splats, int n,
                                          float sizeMul, unsigned int seed) {
    const size_t m = splats.size();
    std::vector<int> order(m);
    std::iota(order.begin(), order.end(), 0);
    if ((size_t)n < m) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uni(1e-7f, 1.0f);
        std::vector<float> key(m);
        for (size_t i = 0; i < m; ++i)
            key[i] = -std::log(uni(rng)) / std::max(SplatWeight(splats[i]), 1e-12f);
        std::nth_element(order.begin(), order.begin() + n, order.end(),
                         [&](int a, int b) { return key[a] < key[b]; });
        order.resize(n);
    }

    float widen = std::min(std::sqrt((float)m / (float)order.size()), 2.0f);
    std::vector<Splat> out(order.size());
    for (size_t k = 0; k < order.size(); ++k) {
        Splat s = splats[order[k]];
        s.scale *= sizeMul * glm::vec3(widen, widen, 1.0f);   // z is the thin axis
        out[k] = s;
    }
    return out;
}

// The n splats a shape brings to the bake.
static std::vector<Splat> BakeSplats(const SplatMeshCache& c, int n, float sizeMul, unsigned int seed) {
    return c.capture ? SubsampleSplats(c.splats, n, sizeMul, seed)
                     : SplatGenerator::FromMesh(c.mesh, n, sizeMul, seed);
}

// Typical splat radius among the n baked splats; sets the neighbour-search cell size.
static float SplatRadius(const SplatMeshCache& c, const std::vector<Splat>& splats, float sizeMul) {
    if (!c.capture) return sizeMul * SplatGenerator::BaseSigma(c.mesh.tris, (int)splats.size());
    std::vector<float> r(splats.size());
    for (size_t i = 0; i < splats.size(); ++i) r[i] = std::sqrt(splats[i].scale.x * splats[i].scale.y);
    std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end());
    return r[r.size() / 2];
}

static void EnsureMixture(SplatMeshCache& c, int K, int samples) {
    if (c.mixK == K && c.mixSamples == samples) return;
    c.mix.Fit(c.capture ? SampleSplats(c.splats, samples, 7u) : MeshSampler::Sample(c.mesh.tris, samples), K);
    c.mixK = K;
    c.mixSamples = samples;
}

// Run f(a) and f(b) on two threads, or once if a and b are the same mesh
// (they share one cache entry, so building it twice in parallel would race).
template <typename F>
static void ForBoth(SplatMeshCache& a, SplatMeshCache& b, F f) {
    if (&a == &b) { f(a); return; }
    auto job = std::async(std::launch::async, [&] { f(b); });
    f(a);
    job.get();
}

static float Dist2(const glm::vec3& a, const glm::vec3& b) {
    glm::vec3 d = a - b;
    return glm::dot(d, d);
}

// Total squared distance of a matching y[i] -> b[perm[i]].
static double MatchCost(const std::vector<glm::vec3>& y, const std::vector<glm::vec3>& b,
                        const std::vector<int>& perm) {
    std::vector<size_t> idx(y.size());
    std::iota(idx.begin(), idx.end(), size_t(0));
    return std::transform_reduce(std::execution::par, idx.begin(), idx.end(), 0.0, std::plus<>(),
                                 [&](size_t i) { return (double)Dist2(y[i], b[perm[i]]); });
}

// A point as the matching sees it: position, then colour times the colour
// weight. With weight 0 the colour axes are flat and never split on, so the
// matching is purely by position.
using MatchPoint = std::array<float, 6>;

static std::vector<MatchPoint> MatchPoints(const std::vector<glm::vec3>& pos,
                                           const std::vector<glm::vec3>& color) {
    std::vector<MatchPoint> pts(pos.size());
    for (size_t i = 0; i < pos.size(); ++i)
        pts[i] = { pos[i].x, pos[i].y, pos[i].z, color[i].x, color[i].y, color[i].z };
    return pts;
}

// One-to-one matching by recursive median splits: split both point sets at
// their median along the longest axis of the region, so each half of y gets
// the same number of b points, and recurse. Small leaves are matched by
// sorting along their longest axis (exact in 1D). Every b is used once.
static void SplitMatch(const std::vector<MatchPoint>& y, const std::vector<MatchPoint>& b,
                       int* iy, int* ib, int n, std::vector<int>& perm, int depth) {
    MatchPoint lo, hi;
    lo.fill(1e30f);
    hi.fill(-1e30f);
    for (int k = 0; k < n; ++k)
        for (int d = 0; d < 6; ++d) {
            lo[d] = std::min({ lo[d], y[iy[k]][d], b[ib[k]][d] });
            hi[d] = std::max({ hi[d], y[iy[k]][d], b[ib[k]][d] });
        }
    int axis = 0;
    for (int d = 1; d < 6; ++d)
        if (hi[d] - lo[d] > hi[axis] - lo[axis]) axis = d;

    if (n <= 16) {
        std::sort(iy, iy + n, [&](int a, int c) { return y[a][axis] < y[c][axis]; });
        std::sort(ib, ib + n, [&](int a, int c) { return b[a][axis] < b[c][axis]; });
        for (int k = 0; k < n; ++k) perm[iy[k]] = ib[k];
        g_bakeDone += n;
        return;
    }

    int mid = n / 2;
    std::nth_element(iy, iy + mid, iy + n, [&](int a, int c) { return y[a][axis] < y[c][axis]; });
    std::nth_element(ib, ib + mid, ib + n, [&](int a, int c) { return b[a][axis] < b[c][axis]; });

    if (depth < 4) {   // top levels run in parallel
        auto job = std::async(std::launch::async, [&] {
            SplitMatch(y, b, iy, ib, mid, perm, depth + 1);
        });
        SplitMatch(y, b, iy + mid, ib + mid, n - mid, perm, depth + 1);
        job.get();
    } else {
        SplitMatch(y, b, iy, ib, mid, perm, depth + 1);
        SplitMatch(y, b, iy + mid, ib + mid, n - mid, perm, depth + 1);
    }
}

static std::vector<int> MedianSplitMatch(const std::vector<MatchPoint>& y,
                                         const std::vector<MatchPoint>& b) {
    const int n = (int)y.size();
    std::vector<int> iy(n), ib(n), perm(n);
    std::iota(iy.begin(), iy.end(), 0);
    std::iota(ib.begin(), ib.end(), 0);
    SplitMatch(y, b, iy.data(), ib.data(), n, perm, 0);
    return perm;
}

// Local improvement: for each i, try swapping partners with the owners of
// nearby b points; keep a swap when it lowers the total cost, squared
// distance plus squared (weighted) colour difference. Candidates come from
// the positions; yc / bc are the weighted colours (zero = position only).
// Returns the number of passes run.
static int RefineMatch(const std::vector<glm::vec3>& y, const std::vector<glm::vec3>& b,
                       const std::vector<glm::vec3>& yc, const std::vector<glm::vec3>& bc,
                       std::vector<int>& perm, float cellSize, int maxPasses) {
    const int n = (int)y.size();
    PointGrid grid;
    grid.Build(b, cellSize);

    std::vector<int> owner(n);
    for (int i = 0; i < n; ++i) owner[perm[i]] = i;

    int pass = 0;
    for (; pass < maxPasses; ++pass) {
        int swaps = 0;
        for (int i = 0; i < n; ++i) {
            grid.ForEachNear(y[i], [&](int j2) {   // candidate b point
                int i2 = owner[j2];                // its current partner
                int j  = perm[i];
                if (i2 == i) return;
                auto cost = [&](int a, int bi) { return Dist2(y[a], b[bi]) + Dist2(yc[a], bc[bi]); };
                float before = cost(i, j)  + cost(i2, j2);
                float after  = cost(i, j2) + cost(i2, j);
                if (after < before - 1e-12f) {
                    perm[i] = j2;  perm[i2] = j;
                    owner[j2] = i; owner[j] = i2;
                    ++swaps;
                }
            });
        }
        g_bakeDone++;
        if (swaps == 0) { ++pass; break; }
    }
    return pass;
}

// How much the OT path stretches a splat's surface patch halfway, beyond what
// the log-lerp of its end sizes already does. Under a linear map M the patch
// spanned by unit tangents t1, t2 gets area |M t1 x M t2|: the area 2-form
// pulled back by M, i.e. the wedge of the pushed tangents. Along the path
// M_t = (1 - t) I + t J that area is a(t), with a(0) = 1. The log-lerp of the
// end sizes already reaches sqrt(a(1)) halfway, so the extra linear size is
//   s = sqrt(a(1/2) / sqrt(a(1)))
// Clamped to [0.5, 2] so one badly stretched splat can't blow up.
static float MidwayStretch(const glm::quat& rot, const Eigen::Matrix3d& J) {
    glm::vec3 t1 = rot * glm::vec3(1, 0, 0);
    glm::vec3 t2 = rot * glm::vec3(0, 1, 0);
    Eigen::Vector3d u(t1.x, t1.y, t1.z);
    Eigen::Vector3d v(t2.x, t2.y, t2.z);
    Eigen::Matrix3d Mhalf = 0.5 * (Eigen::Matrix3d::Identity() + J);
    double aHalf = (Mhalf * u).cross(Mhalf * v).norm();
    double aEnd  = (J * u).cross(J * v).norm();
    if (aEnd < 1e-12) return 1.0f;
    double s = std::sqrt(aHalf / std::sqrt(aEnd));
    return (float)std::clamp(s, 0.5, 2.0);
}

static SplatBakeResult RunSplatBake(SplatBakeSettings st) {
    SplatBakeResult res;
    res.pathA = st.pathA;
    res.pathB = st.pathB;
    res.ply   = st.ply;
    auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(g_meshCacheMutex);   // A and B live in the shared cache

    SetBakeStage(0, 2);
    SplatMeshCache* A = GetMeshCache(st.pathA, st.ply);
    g_bakeDone++;
    SplatMeshCache* B = GetMeshCache(st.pathB, st.ply);
    g_bakeDone++;
    res.loadMs = MsSince(t0);
    if (!A || !B) return res;
    if (A->capture) res.cleanedA = A->splats.size();
    if (B->capture) res.cleanedB = B->splats.size();

    // Same count on both shapes (a capture can't give more splats than it has);
    // each shape sizes its own splats, so both ends are fully covered.
    int count = st.splatCount;
    if (A->capture) count = std::min(count, (int)A->splats.size());
    if (B->capture) count = std::min(count, (int)B->splats.size());
    std::vector<Splat> splatsA = BakeSplats(*A, count, st.sizeMul, 42u);
    std::vector<Splat> splatsB = BakeSplats(*B, count, st.sizeMul, 43u);
    const int n = (int)std::min(splatsA.size(), splatsB.size());
    splatsA.resize(n);
    splatsB.resize(n);
    const float radiusA = SplatRadius(*A, splatsA, st.sizeMul);
    const float radiusB = SplatRadius(*B, splatsB, st.sizeMul);

    // A capture that lost splats to the subsample is shown in full at rest
    // (up to kSplatMax, the most the GPU path is built for)
    auto fullCapture = [&](const SplatMeshCache& c, unsigned int seed) {
        std::vector<Splat> full;
        if (!c.capture || c.splats.size() <= (size_t)n) return full;
        if (c.splats.size() > (size_t)kSplatMax) return SubsampleSplats(c.splats, kSplatMax, st.sizeMul, seed);
        full = c.splats;
        for (Splat& s : full) s.scale *= st.sizeMul;
        return full;
    };
    res.restA = fullCapture(*A, 44u);
    res.restB = fullCapture(*B, 45u);

    // Mixture fit on surface samples; K = 1 reproduces the original single map
    SetBakeStage(1, 1);
    ForBoth(*A, *B, [&](SplatMeshCache& c) { EnsureMixture(c, st.mixtureK, st.sampleCount); });

    SetBakeStage(2, 1);
    MixtureOTMap map = OptimalTransport::ComputeMixture(A->mix, B->mix, st.sinkhornEps, st.posteriorTemp);

    // Coarse: the mixture map sends each A splat roughly where its part of A
    // should go on B (keeps "which part goes where").
    SetBakeStage(3, n);
    std::vector<glm::vec3> y(n), bPos(n);
    std::vector<double> detJ(n);
    res.stretch.resize(n);
    {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            Eigen::Matrix3d J;
            OptimalTransport::ApplyMixture(map, splatsA[i].mean, y[i], J);
            bPos[i] = splatsB[i].mean;
            detJ[i] = J.determinant();   // volume form ratio: the Monge-Ampere factor
            res.stretch[i] = MidwayStretch(splatsA[i].rot, J);
        });
        g_bakeDone = n;
    }
    if (n > 0) {
        std::vector<double> d = detJ;
        std::sort(d.begin(), d.end());
        res.detMin = d.front();
        res.detMed = d[d.size() / 2];
        res.detMax = d.back();
    }
    // K = 1: one Gaussian map, so det J must equal sqrt(det Sigma_B / det Sigma_A)
    if (st.mixtureK == 1 && !map.maps.empty()) {
        double expect = std::sqrt(B->mix.components[0].covariance.determinant() /
                                  A->mix.components[0].covariance.determinant());
        std::cout << "[Splats] det J " << res.detMed << " vs closed form sqrt(det SB / det SA) "
                  << expect << "\n";
    }

    // Fine: one-to-one matching of the mapped points to B's splats
    const int maxPasses = 20;
    SetBakeStage(4, n + maxPasses);
    // Colour-aware: a colour difference of 1 (one full channel) costs as much
    // as colorMatch * kColorUnits of distance (the shapes span about 2 units)
    const float kColorUnits = 0.5f;
    const float colorWeight = st.colorMatch * kColorUnits;
    std::vector<glm::vec3> yColor(n), bColor(n);
    for (int i = 0; i < n; ++i) {
        yColor[i] = colorWeight * splatsA[i].color;
        bColor[i] = colorWeight * splatsB[i].color;
    }
    std::vector<int> perm = MedianSplitMatch(MatchPoints(y, yColor), MatchPoints(bPos, bColor));
    double splitCost = MatchCost(y, bPos, perm);
    g_bakeDone = n;
    float cell = 2.0f * radiusB;
    int passes = RefineMatch(y, bPos, yColor, bColor, perm, cell, maxPasses);
    double refinedCost = MatchCost(y, bPos, perm);
    res.meanMatchDist = std::sqrt(refinedCost / n);

    // Lower bound for reference: every mapped point to its nearest b point,
    // ignoring the one-to-one constraint.
    {
        PointGrid g;
        g.Build(bPos, cell);
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        double nn = std::transform_reduce(std::execution::par, idx.begin(), idx.end(), 0.0,
            std::plus<>(), [&](int i) { return (double)Dist2(y[i], bPos[g.Nearest(y[i])]); });
        res.nearestBound = std::sqrt(nn / n);
    }

    // Keyframes: start = A's real splats, end = the matched B splat, with its
    // disk frame turned to stay close to the start frame (less spinning).
    res.start = std::move(splatsA);
    res.end.resize(n);
    for (int i = 0; i < n; ++i) {
        Splat e = splatsB[perm[i]];
        e.rot = FrameFacing(e.rot * glm::vec3(0, 0, 1), res.start[i].rot);
        res.end[i] = e;
    }

    // Cohesion: average each splat's displacement over its nearest neighbours
    // on A (repeated Laplacian smoothing). At runtime splats travel along this
    // smooth field first, so neighbours move together as chunks, and only add
    // their own correction near the end, so the endpoints stay exact.
    const int kNN = kSplatNeighbors, kSmoothIters = 10;
    SetBakeStage(5, 1 + kSmoothIters);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    res.knn.assign((size_t)n * kNN, -1);
    {
        std::vector<glm::vec3> aPos(n);
        for (int i = 0; i < n; ++i) aPos[i] = res.start[i].mean;
        PointGrid g;
        g.Build(aPos, 2.0f * radiusA);

        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            std::vector<std::pair<float, int>> cand;
            g.ForEachNear(aPos[i], [&](int j) {
                if (j != i) cand.push_back({ Dist2(aPos[i], aPos[j]), j });
            });
            int m = std::min(kNN, (int)cand.size());
            std::partial_sort(cand.begin(), cand.begin() + m, cand.end());
            for (int k = 0; k < m; ++k) res.knn[(size_t)i * kNN + k] = cand[k].second;
        });
    }
    g_bakeDone++;

    std::vector<glm::vec3> disp(n), cur, next(n);
    for (int i = 0; i < n; ++i) disp[i] = res.end[i].mean - res.start[i].mean;
    cur = disp;
    for (int it = 0; it < kSmoothIters; ++it) {
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            glm::vec3 sum = cur[i];
            int cnt = 1;
            for (int k = 0; k < kNN; ++k) {
                int j = res.knn[(size_t)i * kNN + k];
                if (j >= 0) { sum += cur[j]; ++cnt; }
            }
            next[i] = sum / (float)cnt;
        });
        std::swap(cur, next);
        g_bakeDone++;
    }
    res.smoothDisp = std::move(cur);

    // How differently neighbours move: mean |d_i - d_j| over neighbour pairs
    auto neighborSpread = [&](const std::vector<glm::vec3>& d) {
        return std::transform_reduce(std::execution::par, idx.begin(), idx.end(), 0.0, std::plus<>(),
            [&](int i) {
                double s = 0.0;
                for (int k = 0; k < kNN; ++k) {
                    int j = res.knn[(size_t)i * kNN + k];
                    if (j >= 0) s += glm::length(d[i] - d[j]);
                }
                return s;
            }) / ((double)n * kNN);
    };
    double rawSpread = neighborSpread(disp), smoothSpread = neighborSpread(res.smoothDisp);

    SetBakeStage(6, 1);
    res.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "[Splats] matching: split cost " << splitCost << " -> refined " << refinedCost
              << " (" << passes << " passes), rms dist " << res.meanMatchDist
              << " vs nearest-point bound " << res.nearestBound << "\n";
    std::cout << "[Splats] neighbour displacement spread: raw " << rawSpread
              << " -> smoothed " << smoothSpread << "\n";
    std::cout << "[Splats] map volume ratio det J: min " << res.detMin << ", median " << res.detMed
              << ", max " << res.detMax << "\n";
    res.ok = true;
    return res;
}

static std::future<SplatBakeResult> g_bakeFuture;
static int g_bakeEpoch  = 0;   // bumped by Clear / mode switches; a running bake is then dropped
static int g_bakeLaunchEpoch = 0;   // epoch when the running bake started

void Window::BakeSplatMorph(const std::string& pathA, const std::string& pathB) {
    if (splatBaking || morphBaking) return;

    SplatBakeSettings st;
    st.pathA         = pathA;
    st.pathB         = pathB;
    st.splatCount    = splatCount;
    st.sampleCount   = sampleCount;
    st.mixtureK      = mixtureK;
    st.sizeMul       = splatSizeMul;
    st.sinkhornEps   = sinkhornEps;
    st.posteriorTemp = posteriorTemp;
    st.colorMatch    = colorMatch;
    st.ply           = plyOptions;

    g_stageNames = kBakeStageNames;
    g_stageCount = 6;
    SetBakeStage(0, 1);
    splatBaking  = true;
    g_bakeLaunchEpoch = g_bakeEpoch;
    g_bakeFuture = std::async(std::launch::async, RunSplatBake, st);
}

// Called every frame: when the background bake has finished, hand its
// keyframes to the renderer and start the animation (main thread only).
void Window::PollSplatBake() {
    if (!splatBaking || !g_bakeFuture.valid()) return;
    if (g_bakeFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;

    splatBaking = false;
    SplatBakeResult res;
    try {
        res = g_bakeFuture.get();
    } catch (const std::exception& ex) {
        std::cerr << "[Splats] bake failed: " << ex.what() << "\n";
        return;
    }
    if (!res.ok) {
        std::cerr << "[Splats] bake failed: could not load meshes\n";
        return;
    }
    if (res.cleanedA > 0) RecordCleanedCount(CaptureKey(res.pathA, res.ply), res.cleanedA);
    if (res.cleanedB > 0) RecordCleanedCount(CaptureKey(res.pathB, res.ply), res.cleanedB);
    // user pressed Clear or left Splats mode while the bake was running
    if (g_bakeLaunchEpoch != g_bakeEpoch || renderMode != RenderMode::Splats) return;

    splatStart   = std::move(res.start);
    splatEnd        = std::move(res.end);
    splatSmoothDisp = std::move(res.smoothDisp);
    splatStretch    = std::move(res.stretch);
    splatCurrent    = splatStart;

    // GPU path: the keyframes go up once; each frame then only sends the time
    if (gpuAvailable) {
        splatRenderer->UploadKeyframes(splatStart, splatEnd, splatSmoothDisp, splatStretch);
        g_fullSplats[0]->UploadKeyframes(res.restA, res.restA, {});   // empty = none
        g_fullSplats[1]->UploadKeyframes(res.restB, res.restB, {});
    }

    // The animator only provides the timeline (easing, duration, state)
    animator->InitializeWithEndPoints({}, {});
    animator->StartTransport();
    splatMorphActive    = true;
    trueMorphActive     = false;
    particleMorphActive = false;
    lastBakeMs          = (float)res.ms;
    g_perf.ObserveBake((int)splatStart.size(), res.ms - res.loadMs);
    g_refitQuality = true;

    std::cout << "[Splats] baked " << splatStart.size() << " matched splats, K=" << mixtureK
              << ", mean match distance " << res.meanMatchDist << ", " << res.ms << " ms\n";
}

// What the preview worker hands back for a capture.
struct CapturePreview {
    std::vector<Splat> splats;   // what to show, unscaled (Splat Size is applied later)
    std::string countKey;        // CaptureKey of the capture
    size_t cleaned = 0;          // splats it kept after cleaning; 0 = the load failed
};

static const int kCpuPreviewMax = 50000;   // the CPU path sorts every frame, so show a subsample

// Runs on a worker. Loads through the bake's cache, so the bake won't parse
// the file again, and keeps at most maxCount splats.
static CapturePreview LoadCapturePreview(std::string path, PlyLoadOptions ply, int maxCount) {
    std::lock_guard<std::mutex> lock(g_meshCacheMutex);
    CapturePreview out;
    out.countKey = CaptureKey(path, ply);
    SplatMeshCache* c = GetMeshCache(path, ply);
    if (!c) return out;
    out.cleaned = c->splats.size();
    if (c->splats.size() > (size_t)maxCount) out.splats = SubsampleSplats(c->splats, maxCount, 1.0f, 42u);
    else                                     out.splats = c->splats;
    return out;
}

// Source A is a capture. A big .ply takes seconds to parse, so it loads on a
// worker and the window keeps running; nothing is shown until it lands. Once
// it's on screen, the Splat Size slider only rescales it: the GPU path applies
// it at draw time, the CPU path rescales its kept subsample.
// resume: we're coming back from another preview, so splatCurrent is stale.
static void UpdateCapturePreview(const std::string& path, bool gpu, bool resume) {
    static std::future<CapturePreview> pending;
    static std::string pendingKey, shownKey, failedKey;   // capture being loaded / on screen / unreadable
    static std::vector<Splat> cpuBase;                     // CPU path: the unscaled subsample
    static float shownSize = -1.0f;

    std::string key = CaptureKey(path, Window::plyOptions) + (gpu ? "|gpu|" : "|cpu|") +
                      std::to_string(splatPreviewGen);
    if (resume) shownSize = -1.0f;

    // a finished load: show it if it's still the one wanted
    if (pending.valid() && pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        CapturePreview p = pending.get();
        if (p.cleaned > 0) RecordCleanedCount(p.countKey, p.cleaned);
        if (pendingKey == key) {
            if (p.cleaned == 0) {
                failedKey = key;   // don't retry every frame
            } else {
                shownKey  = key;
                shownSize = -1.0f;
                if (gpu) g_fullSplats[0]->UploadKeyframes(p.splats, p.splats, {});
                else     cpuBase = std::move(p.splats);
            }
        }
        pendingKey.clear();
    }

    if (shownKey != key) {
        g_previewOnGpu = false;
        Window::splatCurrent.clear();
        if (!pending.valid() && key != failedKey) {
            pendingKey = key;
            int maxCount = gpu ? kSplatMax : kCpuPreviewMax;
            pending = std::async(std::launch::async, LoadCapturePreview, path, Window::plyOptions, maxCount);
        }
        return;
    }

    if (gpu) {
        g_previewOnGpu = true;
    } else if (shownSize != Window::splatSizeMul) {
        shownSize = Window::splatSizeMul;
        Window::splatCurrent = cpuBase;
        for (Splat& s : Window::splatCurrent) s.scale *= shownSize;
    }
}

// Show source A (mesh, capture or the default blob) as static splats until a
// morph is baked. Regenerates only when the inputs change.
void Window::UpdateSplatPreview() {
    static int   lastIndex = -1, lastCount = -1, lastGen = -1;
    static float lastSize  = -1.0f;
    static bool  lastWasCapture = false;

    bool hasShape = meshIndexA > 0 && meshIndexA <= (int)meshPaths.size();
    if (hasShape && IsCapture(meshPaths[meshIndexA - 1])) {
        UpdateCapturePreview(meshPaths[meshIndexA - 1], useGpu && gpuAvailable, !lastWasCapture);
        lastWasCapture = true;
        lastIndex = -1;   // the mesh / blob preview must rebuild when we come back
        return;
    }
    lastWasCapture = false;
    g_previewOnGpu = false;

    if (lastIndex == meshIndexA && lastCount == splatCount && lastSize == splatSizeMul &&
        lastGen == splatPreviewGen)
        return;
    lastIndex = meshIndexA; lastCount = splatCount; lastSize = splatSizeMul;
    lastGen = splatPreviewGen;

    if (hasShape) {
        // keep the loaded mesh so dragging the count/size sliders doesn't reload it
        static MeshData mesh;
        static int loadedIndex = -1;
        if (loadedIndex != meshIndexA) {
            loadedIndex = -1;
            if (!LoadNormalized(meshPaths[meshIndexA - 1], mesh)) return;
            loadedIndex = meshIndexA;
        }
        splatCurrent = SplatGenerator::FromMesh(mesh, splatCount, splatSizeMul, 42u);
    } else {
        // Blob: round splats sampled from the default isotropic Gaussian
        Gaussian blob;
        SetBlob(blob);
        auto pts = blob.Sample(splatCount);
        splatCurrent.assign(pts.size(), Splat());
        for (size_t i = 0; i < pts.size(); ++i) {
            splatCurrent[i].mean    = pts[i];
            splatCurrent[i].scale   = glm::vec3(0.04f * splatSizeMul);
            splatCurrent[i].color   = SplatGenerator::baseColor;
            splatCurrent[i].opacity = 0.5f;
        }
    }
}

void Window::ComputeOTAndSpawnParticles() {
    // meshIndex 0 = blob; 1..N = meshPaths[index-1]
    bool hasMeshA = (meshIndexA > 0 && meshIndexA <= (int)meshPaths.size());
    bool hasMeshB = (meshIndexB > 0 && meshIndexB <= (int)meshPaths.size());
    const std::string& pathA = hasMeshA ? meshPaths[meshIndexA - 1] : "";
    const std::string& pathB = hasMeshB ? meshPaths[meshIndexB - 1] : "";

    if (renderMode == RenderMode::Splats) {
        if (hasMeshA && hasMeshB) BakeSplatMorph(pathA, pathB);
        return;
    }
    splatMorphActive = false;
    trueMorphActive  = false;

    if (renderMode == RenderMode::TriangleMorph && hasMeshA && hasMeshB) {
        BakeTrueMorph(pathA, pathB);
        return;
    }

    // === Particles ===
    if (hasMeshA && hasMeshB) {
        // === Discrete mesh-to-mesh OT ===
        // Particles start exactly on mesh A surface, end exactly on mesh B surface.
        // The Gaussian OT map is not used — the greedy matching IS the transport.
        std::vector<Triangle> trisA, trisB;
        if (!LoadNormalized(pathA, trisA)) return;
        if (!LoadNormalized(pathB, trisB)) return;

        auto ptsA = MeshSampler::Sample(trisA, particleCount);
        auto ptsB = MeshSampler::Sample(trisB, particleCount);

        auto endPts = GreedyDiscreteOT(ptsA, ptsB);

        // Place particles directly at mesh A surface positions
        particleSystem->SpawnAt(ptsA);

        animator->InitializeWithEndPoints(ptsA, endPts);
        particleSystem->useOTPositions = true;
        animator->StartTransport();
        particleMorphActive = true;

    } else {
        // === Gaussian OT (mesh → Gaussian or blob → blob) ===
        if (hasMeshA) {
            if (!LoadMeshGaussian(pathA, sampleCount, gaussianA)) return;
        } else {
            SetBlob(*gaussianA);
        }

        if (hasMeshB) {
            if (!LoadMeshGaussian(pathB, sampleCount, gaussianB)) return;
        } else {
            SetBlob(*gaussianB);
        }

        delete otMap;
        otMap = new OTMap(OptimalTransport::Compute(*gaussianA, *gaussianB));

        particleSystem->SpawnFromGaussian(*gaussianA, particleCount);
        animator->Initialize(particleSystem->transportPositions, *otMap);
        particleSystem->useOTPositions = true;
        animator->StartTransport();
        particleMorphActive = true;
    }
}

// ---- True morph: one surface going from A to B ----
//
// Shared topology: the denser mesh is the template. The Gaussian OT map gives
// a rough affine alignment, then MeshFitter deforms the template onto the
// other surface, so both shapes have the same vertices and the morph is a
// per-vertex lerp. This needs both meshes to have the same topology (one
// part, same genus, same number of holes).
//
// SDF: for every other pair, blend OT-aligned signed distance fields and
// re-mesh each frame. Holes and parts can appear and disappear.
//
// Either way the real meshes are shown exactly at t = 0 and t = 1.

static const int kMaxTemplateFaces = 40000;

// Load, normalize, weld, close small holes, orient outward.
// m: the cleaned, welded mesh the morph uses; data: the file as loaded, with its colours.
static bool LoadMorphMesh(const std::string& path, IndexedMesh& m, TopologyInfo& info, MeshData& data) {
    if (!LoadNormalized(path, data)) return false;
    MeshTopology::Build(data.tris, m);
    int filled = MeshTopology::FillSmallHoles(m, 40);
    // the signed volume in OrientOutward only means something once the winding agrees
    bool orientable = MeshTopology::OrientConsistently(m);
    MeshTopology::OrientOutward(m);
    info = MeshTopology::Analyze(m);
    info.orientable = orientable;
    std::cout << "[Morph] " << path << ": " << info.Describe();
    if (filled) std::cout << " (" << filled << " small holes filled)";
    std::cout << "\n";
    return true;
}

// Empty if the two meshes can share one topology, else the reason they can't.
static std::string TopologyMismatch(const TopologyInfo& a, const std::string& nameA,
                                    const TopologyInfo& b, const std::string& nameB) {
    const TopologyInfo* infos[2] = { &a, &b };
    const std::string*  names[2] = { &nameA, &nameB };
    for (int k = 0; k < 2; ++k) {
        if (infos[k]->parts != 1)
            return *names[k] + " has " + std::to_string(infos[k]->parts) + " parts";
        if (infos[k]->nonManifoldEdges > 0)
            return *names[k] + " has non-manifold edges";
        if (!infos[k]->orientable)
            return *names[k] + " is non-orientable";
    }
    if (a.genus != b.genus)
        return "different genus (" + std::to_string(a.genus) + " vs " + std::to_string(b.genus) + ")";
    if (a.euler != b.euler)
        return "different Euler characteristic (chi " + std::to_string(a.euler) + " vs " +
               std::to_string(b.euler) + ")";
    if (a.boundaryLoops != b.boundaryLoops)
        return "different number of holes (" + std::to_string(a.boundaryLoops) + " vs " +
               std::to_string(b.boundaryLoops) + ")";
    return "";
}

// Everything a topology bake needs, copied when Bake is clicked so GUI
// changes during the bake can't affect it.
struct TrueMorphSettings {
    std::string pathA, pathB;
    MorphMethod method;
    int   sdfResolution, sampleCount;
    float fitStiffness;
    bool  fitCotan;
    glm::vec3 fallbackColor;   // display colour of a mesh without colours, beside one with them
};

struct TrueMorphResult {
    bool ok = false;
    IndexedMesh a, b;                    // cleaned endpoint meshes
    MeshData dataA, dataB;               // the files as loaded (with materials), if coloured
    std::vector<glm::vec3> colorsA, colorsB;               // per vertex of a / b (empty: no colours)
    std::vector<glm::vec3> sharedColorStart, sharedColorEnd;
    bool shared = false;
    std::vector<unsigned int> sharedIndices;
    std::vector<glm::vec3>    sharedStart, sharedEnd;
    SDFMorph sdf;
    int sdfVertCap = 0, sdfIndexCap = 0; // GPU buffer sizes that fit every frame (with room)
    Gaussian gA, gB;
    OTMap map;
    std::string status;
};

// The slow part of a topology bake. Runs on a background thread, so it only
// touches its own data (no GL, no Window state).
static TrueMorphResult RunTrueMorphBake(TrueMorphSettings st) {
    TrueMorphResult res;
    auto t0 = std::chrono::steady_clock::now();

    SetBakeStage(0, 2);
    TopologyInfo infoA, infoB;
    MeshData dataA, dataB;
    if (!LoadMorphMesh(st.pathA, res.a, infoA, dataA)) return res;
    g_bakeDone++;
    if (!LoadMorphMesh(st.pathB, res.b, infoB, dataB)) return res;
    g_bakeDone++;

    // Colours: every surface the morph shows takes them from the closest
    // point of the original meshes. Only if at least one of them has any.
    const bool colored = dataA.HasAppearance() || dataB.HasAppearance();
    SurfaceColor colorA, colorB;
    if (colored) {
        colorA.Build(dataA, st.fallbackColor);
        colorB.Build(dataB, st.fallbackColor);
        res.colorsA = colorA.At(res.a.verts);
        res.colorsB = colorB.At(res.b.verts);
    }
    if (dataA.HasAppearance()) res.dataA = std::move(dataA);
    if (dataB.HasAppearance()) res.dataB = std::move(dataB);
    std::string nameA = std::filesystem::path(st.pathA).filename().string();
    std::string nameB = std::filesystem::path(st.pathB).filename().string();

    std::vector<Triangle> trisA = MeshTopology::ToTriangles(res.a);
    std::vector<Triangle> trisB = MeshTopology::ToTriangles(res.b);

    std::string mismatch = TopologyMismatch(infoA, nameA, infoB, nameB);
    res.shared = (st.method == MorphMethod::SharedTopology) ||
                 (st.method == MorphMethod::Auto && mismatch.empty());

    SetBakeStage(1, 1);
    res.gA.FitFromPoints(MeshSampler::Sample(trisA, st.sampleCount));
    res.gB.FitFromPoints(MeshSampler::Sample(trisB, st.sampleCount));
    res.map = OptimalTransport::Compute(res.gA, res.gB);

    SetBakeStage(2, 1);
    char buf[512];
    if (res.shared) {
        // The template is the mesh with more faces, unless it is too big to fit
        // quickly; then the smaller one is subdivided up to a similar size.
        bool templateIsA = res.a.FaceCount() >= res.b.FaceCount();
        if ((templateIsA ? res.a : res.b).FaceCount() > kMaxTemplateFaces)
            templateIsA = !templateIsA;
        const IndexedMesh& other = templateIsA ? res.b : res.a;

        IndexedMesh tmpl = templateIsA ? res.a : res.b;
        bool subdivided = false;
        while (tmpl.FaceCount() * 4 <= kMaxTemplateFaces && tmpl.FaceCount() < other.FaceCount()) {
            std::vector<Triangle> t = MeshTopology::ToTriangles(tmpl);
            MeshTopology::Subdivide(t, 1);
            MeshTopology::Build(t, tmpl);
            subdivided = true;
        }

        // Rough affine alignment by the Gaussian OT map, template -> target
        OTMap toTarget = templateIsA ? res.map : OptimalTransport::Compute(res.gB, res.gA);
        std::vector<glm::vec3> aligned(tmpl.verts.size());
        for (size_t i = 0; i < aligned.size(); ++i)
            aligned[i] = OptimalTransport::Apply(toTarget, tmpl.verts[i]);

        FitParams params;
        params.stiffnessScale = st.fitStiffness;
        params.cotanLaplacian = st.fitCotan;
        FitStats stats;
        std::vector<glm::vec3> fitted = MeshFitter::Fit(tmpl, aligned,
                                                        templateIsA ? trisB : trisA, params, &stats);

        res.sharedIndices = tmpl.indices;
        res.sharedStart   = templateIsA ? tmpl.verts : fitted;
        res.sharedEnd     = templateIsA ? fitted : tmpl.verts;
        if (colored) {
            res.sharedColorStart = colorA.At(res.sharedStart);
            res.sharedColorEnd   = colorB.At(res.sharedEnd);
        }

        snprintf(buf, sizeof(buf),
                 "Shared topology: %s (%d tris%s) fitted onto %s in %.0f ms. "
                 "Mean error %.4f, %d folded tris.%s%s",
                 (templateIsA ? nameA : nameB).c_str(), tmpl.FaceCount(),
                 subdivided ? ", subdivided" : "", (templateIsA ? nameB : nameA).c_str(),
                 stats.ms, stats.meanDist, stats.flipped,
                 mismatch.empty() ? "" : " Warning: ", mismatch.c_str());
    } else {
        res.sdf.Build(trisA, trisB, res.map, st.sdfResolution,
                      colored ? &colorA : nullptr, colored ? &colorB : nullptr);

        // Size the GPU buffers from a few CPU extractions, with room to spare
        size_t maxVerts = 0, maxIndices = 0;
        for (float t : { 0.0f, 0.5f, 1.0f }) {
            std::vector<glm::vec3> v;
            std::vector<unsigned int> idx;
            res.sdf.Extract(t, v, idx);
            maxVerts   = std::max(maxVerts, v.size());
            maxIndices = std::max(maxIndices, idx.size());
        }
        res.sdfVertCap  = (int)(2 * maxVerts + 1024);
        res.sdfIndexCap = (int)(2 * maxIndices + 6 * 1024);

        snprintf(buf, sizeof(buf), "SDF morph (%d^3 grid) built in %.0f ms%s%s",
                 st.sdfResolution, res.sdf.buildMs,
                 mismatch.empty() ? "" : ": ", mismatch.c_str());
    }
    res.status = buf;
    SetBakeStage(3, 1);
    std::cout << "[Morph] " << res.status << " (bake " << (int)MsSince(t0) << " ms)\n";
    res.ok = true;
    return res;
}

static std::future<TrueMorphResult> g_morphFuture;

// Starts the topology bake on a background thread; PollTrueMorphBake picks
// up the result, so the UI keeps running meanwhile.
void Window::BakeTrueMorph(const std::string& pathA, const std::string& pathB) {
    if (splatBaking || morphBaking) return;

    TrueMorphSettings st;
    st.pathA         = pathA;
    st.pathB         = pathB;
    st.method        = morphMethod;
    st.sdfResolution = sdfResolution;
    st.sampleCount   = sampleCount;
    st.fitStiffness  = fitStiffness;
    st.fitCotan      = fitCotan;
    st.fallbackColor = glm::sqrt(meshRenderer->diffuseColor);   // shader.frag's grey, as a display value

    g_stageNames = kMorphStageNames;
    g_stageCount = 3;
    SetBakeStage(0, 1);
    morphBaking       = true;
    trueMorphActive   = false;
    g_bakeLaunchEpoch = g_bakeEpoch;
    g_morphFuture     = std::async(std::launch::async, RunTrueMorphBake, st);
}

// Called every frame: when the topology bake has finished, take its result,
// upload what the GPU path needs and start the animation (main thread only,
// since it makes GL calls).
void Window::PollTrueMorphBake() {
    if (!morphBaking || !g_morphFuture.valid()) return;
    if (g_morphFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;

    morphBaking = false;
    TrueMorphResult res;
    try {
        res = g_morphFuture.get();
    } catch (const std::exception& ex) {
        morphStatus = std::string("Bake failed: ") + ex.what();
        std::cerr << "[Morph] " << morphStatus << "\n";
        return;
    }
    if (!res.ok) {
        morphStatus = "Bake failed: could not load meshes";
        return;
    }
    // user pressed Clear or left Topology mode while the bake was running
    if (g_bakeLaunchEpoch != g_bakeEpoch || renderMode != RenderMode::TriangleMorph) return;

    g_morphA       = std::move(res.a);
    g_morphB       = std::move(res.b);
    g_morphShared  = res.shared;
    g_sharedIndices = std::move(res.sharedIndices);
    g_sharedStart   = std::move(res.sharedStart);
    g_sharedEnd     = std::move(res.sharedEnd);
    g_sharedCurrent = g_sharedStart;
    g_colorsA            = std::move(res.colorsA);
    g_colorsB            = std::move(res.colorsB);
    g_sharedColorStart   = std::move(res.sharedColorStart);
    g_sharedColorEnd     = std::move(res.sharedColorEnd);
    g_sharedColorCurrent = g_sharedColorStart;
    g_dataA              = std::move(res.dataA);
    g_dataB              = std::move(res.dataB);
    g_restTextured       = false;
    if (!res.shared) {
        g_sdfMorph = std::move(res.sdf);
        if (gpuAvailable) g_gpuSdf->Upload(g_sdfMorph, res.sdfVertCap, res.sdfIndexCap);
    }
    *gaussianA = res.gA;
    *gaussianB = res.gB;
    delete otMap;
    otMap = new OTMap(res.map);
    morphStatus = res.status;

    // The animator only provides the timeline (easing, duration, state)
    animator->InitializeWithEndPoints({}, {});
    animator->StartTransport();
    g_morphShown        = MorphShown::None;
    g_lastSdfT          = -1.0f;
    g_sdfOnGpu          = false;
    trueMorphActive     = true;
    particleMorphActive = false;
    UpdateTrueMorph();
}

// Gets this frame's surface ready to draw: the real A or B mesh at the ends
// of the timeline, the in-between surface everywhere else.
//   Shared topology, GPU: both shapes are uploaded once, the vertex shader
//     blends them (morph.vert); per frame only morphT changes.
//   SDF, GPU: GpuSDFMorph re-meshes in compute shaders, drawn indirectly.
//   CPU paths (useGpu off or no GL 4.3): the original lerp / surface nets.
void Window::UpdateTrueMorph() {
    float e = animator->GetBlend();
    bool gpu = useGpu && gpuAvailable;

    // counts from a few frames ago; re-mesh if the GPU buffers had to grow
    if (!g_morphShared && g_gpuSdf->IsReady() && g_gpuSdf->PollStats())
        g_lastSdfT = -1.0f;

    MorphShown want = MorphShown::Blend;
    if (e <= 0.0f && !animator->IsTransporting())
        want = MorphShown::A;
    else if (e >= 1.0f && !animator->IsTransporting())
        want = MorphShown::B;

    if (want != MorphShown::Blend) {
        if (g_morphShown != want) {
            const IndexedMesh& m = (want == MorphShown::A) ? g_morphA : g_morphB;
            meshRenderer->Initialize(m.verts, m.indices, (want == MorphShown::A) ? g_colorsA : g_colorsB);
            // a coloured original is shown as itself, textures and all
            const MeshData& data = (want == MorphShown::A) ? g_dataA : g_dataB;
            g_restTextured = data.HasAppearance();
            if (g_restTextured) {
                if (!g_restMesh) g_restMesh = new TexturedMeshRenderer();
                g_restMesh->Initialize(data);
            }
            g_morphShown = want;
        }
        return;
    }
    g_restTextured = false;

    if (g_morphShared) {
        if (gpu) {
            if (g_morphShown != MorphShown::Blend || !meshRenderer->IsMorphTarget()) {
                meshRenderer->InitializeMorph(g_sharedStart, g_sharedEnd, g_sharedIndices,
                                              g_sharedColorStart, g_sharedColorEnd);
                g_morphShown = MorphShown::Blend;
            }
            meshRenderer->morphT = e;
            return;
        }
        for (size_t i = 0; i < g_sharedCurrent.size(); ++i)
            g_sharedCurrent[i] = g_sharedStart[i] + e * (g_sharedEnd[i] - g_sharedStart[i]);
        for (size_t i = 0; i < g_sharedColorCurrent.size(); ++i)
            g_sharedColorCurrent[i] = glm::mix(g_sharedColorStart[i], g_sharedColorEnd[i], glm::clamp(e, 0.0f, 1.0f));
        if (g_morphShown != MorphShown::Blend || meshRenderer->IsMorphTarget()) {
            meshRenderer->Initialize(g_sharedCurrent, g_sharedIndices, g_sharedColorCurrent);
            g_morphShown = MorphShown::Blend;
        } else {
            meshRenderer->UpdatePositions(g_sharedCurrent, g_sharedColorCurrent);
        }
        return;
    }

    // SDF: re-mesh only when t moved (bounce easing can overshoot; clamp it)
    float t = glm::clamp(e, 0.0f, 1.0f);
    bool onGpu = gpu && g_gpuSdf->IsReady();
    if (g_morphShown == MorphShown::Blend && t == g_lastSdfT && onGpu == g_sdfOnGpu) return;
    if (onGpu) {
        g_morphTimer->Begin();
        g_gpuSdf->Extract(t);
        g_morphTimer->End();
    } else {
        std::vector<glm::vec3> verts, colors;
        std::vector<unsigned int> indices;
        g_sdfMorph.Extract(t, verts, indices, &colors);
        if (!indices.empty()) meshRenderer->Initialize(verts, indices, colors);
    }
    g_sdfOnGpu   = onGpu;
    g_morphShown = MorphShown::Blend;
    g_lastSdfT   = t;
}

// Checks the GPU path against the CPU one at the current state (reads GPU
// results back, so it stalls; only for the Verify button).
static void VerifyGpuAgainstCpu() {
    char buf[256];
    if (Window::renderMode == RenderMode::TriangleMorph && Window::trueMorphActive) {
        if (g_morphShared) {
            g_verifyText = "Shared topology: both paths compute start + t (end - start) per vertex "
                           "(GPU in morph.vert), nothing to read back.";
            return;
        }
        if (!g_gpuSdf->IsReady()) { g_verifyText = "GPU SDF not uploaded."; return; }
        float t = (g_lastSdfT >= 0.0f) ? g_lastSdfT : 0.5f;
        g_gpuSdf->Extract(t);
        int gv, gt;
        bool overflow;
        g_gpuSdf->ReadCountsNow(gv, gt, overflow);
        std::vector<glm::vec3> verts;
        std::vector<unsigned int> indices;
        g_sdfMorph.Extract(t, verts, indices);
        int cv = (int)verts.size(), ct = (int)indices.size() / 3;
        snprintf(buf, sizeof(buf), "SDF t=%.3f: GPU %d verts / %d tris, CPU %d / %d (%+.2f%% tris)%s",
                 t, gv, gt, cv, ct, ct ? 100.0 * (gt - ct) / ct : 0.0,
                 overflow ? ", GPU overflowed" : "");
        g_lastSdfT = -1.0f;   // re-extract normally next frame
    } else if (Window::renderMode == RenderMode::Splats && Window::splatMorphActive && SplatsOnGpu()) {
        std::vector<uint32_t> keys;
        bool ok = Window::splatRenderer->CheckGpuSortOrder(&keys);
        size_t culled = std::count(keys.begin(), keys.end(), 0xFFFFFFFFu);
        snprintf(buf, sizeof(buf), "GPU sort %s: %zu splats, back to front, each once (%zu culled)",
                 ok ? "OK" : "FAILED", keys.size(), culled);
    } else {
        snprintf(buf, sizeof(buf), "Bake a Topology (SDF) or Splats morph with GPU playback on first.");
    }
    g_verifyText = buf;
    std::cout << "[Verify] " << g_verifyText << "\n";
}

// Stop every morph and drop any bake still running, so the display goes back
// to the static preview of the current source. Used by Clear, mode switches
// and source changes.
static void StopAllMorphs() {
    Window::animator->Reset();
    Window::particleSystem->useOTPositions = false;
    Window::trueMorphActive     = false;
    Window::particleMorphActive = false;
    g_bakeEpoch++;   // a bake still running is discarded
    if (Window::splatMorphActive) {
        Window::splatMorphActive = false;
        splatPreviewGen++;
    }
    g_previewGen++;
}

// Source / target picker: entry 0 is the default Gaussian blob, the rest are
// the .obj files found in meshes/.
static bool DrawShapeCombo(const char* label, int& meshIndex) {
    int total = 1 + (int)Window::meshNames.size();
    auto getter = [](void* data, int idx, const char** out) -> bool {
        auto* names = static_cast<std::vector<std::string>*>(data);
        if (idx == 0) { *out = "Gaussian blob"; return true; }
        if (idx - 1 < (int)names->size()) { *out = (*names)[idx - 1].c_str(); return true; }
        return false;
    };
    return ImGui::Combo(label, &meshIndex, getter, &Window::meshNames, total);
}

static void HintText(const char* text) {
    ImGui::TextDisabled("%s", text);
}

void Window::DrawMainGUI() {
    ImGui::Begin("Gaussian Cake");
    ImGui::PushItemWidth(ImGui::GetFontSize() * -9.0f);   // room for labels on the right

    // ---- 1. Type ----
    ImGui::SeparatorText("Type");
    int modeInt = (int)renderMode;
    bool modeChanged = false;
    modeChanged |= ImGui::RadioButton("Particles##mode", &modeInt, (int)RenderMode::Particles);
    ImGui::SameLine();
    modeChanged |= ImGui::RadioButton("Topology##mode", &modeInt, (int)RenderMode::TriangleMorph);
    ImGui::SameLine();
    modeChanged |= ImGui::RadioButton("Splats##mode", &modeInt, (int)RenderMode::Splats);
    if (modeChanged && modeInt != (int)renderMode) {
        renderMode = (RenderMode)modeInt;
        // Switching modes stops any active morph and shows the new mode's
        // preview of the source; user re-clicks Bake.
        StopAllMorphs();
    }
    HintText(renderMode == RenderMode::Particles     ? "Point cloud carried from A to B by optimal transport" :
             renderMode == RenderMode::TriangleMorph ? "One solid surface reshaped from A into B" :
                                                       "Gaussian splats, each matched from A to a spot on B");

    // ---- 2. Shapes ----
    ImGui::SeparatorText("Shapes");
    if (DrawShapeCombo("Source (A)", meshIndexA))
        StopAllMorphs();   // show the newly selected source right away
    DrawShapeCombo("Target (B)", meshIndexB);

    bool hasA = meshIndexA > 0, hasB = meshIndexB > 0;
    bool needsMeshes = renderMode != RenderMode::Particles;
    bool captureOutsideSplats = renderMode != RenderMode::Splats &&
        ((hasA && IsCapture(meshPaths[meshIndexA - 1])) || (hasB && IsCapture(meshPaths[meshIndexB - 1])));
    bool baking  = splatBaking || morphBaking;
    bool canBake = !baking && !captureOutsideSplats && (!needsMeshes || (hasA && hasB));

    // ---- 3. Transport ----
    ImGui::SeparatorText("Transport");
    animator->DrawParamsGUI();

    // Bake | Revert | Clear on one row
    ImGui::Spacing();
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float smallW  = ImGui::CalcTextSize("Revert").x + ImGui::GetStyle().FramePadding.x * 4.0f;
    float bakeW   = ImGui::GetContentRegionAvail().x - 2.0f * (smallW + spacing);

    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(1.0f, 0.4f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.6f, 0.8f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.8f, 0.2f, 0.5f, 1.0f));
    ImGui::BeginDisabled(!canBake);
    if (ImGui::Button(baking ? "Baking..." : "Bake It!", ImVec2(bakeW, 0)))
        ComputeOTAndSpawnParticles();
    ImGui::EndDisabled();
    ImGui::PopStyleColor(3);

    ImGui::SameLine();
    ImGui::BeginDisabled(animator->state != AnimationState::GAUSSIAN_B);
    if (ImGui::Button("Revert", ImVec2(smallW, 0)))
        animator->ReverseTransport();
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Play the morph back from B to A");

    ImGui::SameLine();
    if (ImGui::Button("Clear", ImVec2(smallW, 0)))
        StopAllMorphs();
    ImGui::SetItemTooltip("Stop and show the source again");

    if (needsMeshes && !(hasA && hasB))
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), "Pick a mesh for both A and B to bake.");
    if (captureOutsideSplats)
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.4f, 1.0f), ".ply captures only work in Splats mode.");

    // Status: bake progress while baking, then the timeline position
    if (baking) {
        int stage = g_bakeStage.load();
        float frac = (float)g_bakeDone.load() / (float)g_bakeTotal.load();
        char label[64];
        snprintf(label, sizeof(label), "%s (%d/%d)", g_stageNames[stage],
                 std::min(stage + 1, g_stageCount), g_stageCount);
        ImGui::ProgressBar(glm::clamp(frac, 0.0f, 1.0f), ImVec2(-1, 0), label);
    } else {
        animator->DrawStatusGUI();
    }
    if (renderMode == RenderMode::Splats && lastBakeMs >= 0.0f && !splatBaking)
        HintText(("Last bake: " + std::to_string((int)lastBakeMs) + " ms").c_str());
    if (renderMode == RenderMode::TriangleMorph && !morphStatus.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s", morphStatus.c_str());
        ImGui::PopStyleColor();
    }

    // ---- 4. Settings for the selected type ----
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Quality: one bar for every costly setting. The bar moves freely while
        // dragging; the settings follow on release (a new count rebuilds the preview).
        int ceiling = SplatCeiling();
        QualitySettings atBar = SettingsFor(quality, ceiling);
        bool custom = !MatchesQuality(atBar);
        const char* tier = quality < 0.25f ? "Low" : quality < 0.5f ? "Medium" :
                           quality < 0.75f ? "High" : "Ultra";
        char barText[64];
        if (custom)
            snprintf(barText, sizeof(barText), "Custom");
        else if (renderMode == RenderMode::Splats)
            snprintf(barText, sizeof(barText), "%s - %dk splats", tier, (atBar.splatCount + 500) / 1000);
        else if (renderMode == RenderMode::TriangleMorph)
            snprintf(barText, sizeof(barText), "%s - SDF %d", tier, atBar.sdfResolution);
        else
            snprintf(barText, sizeof(barText), "%s - %d particles", tier, atBar.particleCount);

        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetFrameHeight() -
                                ImGui::GetStyle().ItemInnerSpacing.x - ImGui::CalcTextSize("Auto").x);
        ImGui::SliderFloat("##quality", &quality, 0.0f, 1.0f, barText, ImGuiSliderFlags_NoInput);
        if (ImGui::IsItemActivated()) autoQuality = false;
        if (ImGui::IsItemDeactivatedAfterEdit()) ApplySettings(SettingsFor(quality, ceiling));
        ImGui::SetItemTooltip("Left = fast, right = fine.\nMoves the splat count, mixture K, samples,\n"
                              "SDF resolution and particle count together.");
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::Checkbox("Auto", &autoQuality);
        ImGui::SetItemTooltip("Pick the quality from the chosen shapes and how fast\n"
                              "this machine has baked and drawn so far");
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::TextUnformatted("Quality");
        if (renderMode == RenderMode::Splats) {
            int n = custom ? splatCount : atBar.splatCount;
            char est[96];
            snprintf(est, sizeof(est), "~%.1f s bake, ~%.1f ms per frame (GPU)",
                     g_perf.PredictBakeMs(n) / 1000.0, g_perf.PredictFrameMs(n));
            HintText(est);
        }

        // Settings the bar doesn't touch
        if (renderMode == RenderMode::Splats) {
            ImGui::SeparatorText("Captures (.ply)");
            // Changing the crop reloads the capture, so apply it on release, not while dragging
            static float crop = plyOptions.cropRadius;
            ImGui::SliderFloat("Crop Radius", &crop, 1.0f, 4.0f);
            if (ImGui::IsItemDeactivatedAfterEdit()) plyOptions.cropRadius = crop;
            ImGui::SetItemTooltip("Keep splats within this many times the capture's typical radius");
            const char* upAxes[] = { "As is", "Flip Y (COLMAP)", "Rotate X +90", "Rotate X -90" };
            int up = (int)plyOptions.up;
            if (ImGui::Combo("Up Axis", &up, upAxes, 4)) plyOptions.up = (PlyLoadOptions::Up)up;
        }

        // Advanced: every setting by hand. Editing a costly one turns Auto off,
        // so a model change doesn't overwrite it.
        if (ImGui::TreeNode("Advanced")) {
            if (renderMode == RenderMode::Particles) {
                if (ImGui::SliderInt("Particles", &particleCount, 100, 10000)) autoQuality = false;
                ImGui::SliderFloat("Particle Radius", &particleSystem->particleRadius, 0.01f, 1.0f);
            } else if (renderMode == RenderMode::TriangleMorph) {
                const char* methods[] = { "Auto", "Shared Topology", "SDF" };
                int m = (int)morphMethod;
                if (ImGui::Combo("Method", &m, methods, 3)) morphMethod = (MorphMethod)m;
                HintText(morphMethod == MorphMethod::Auto ? "Shared topology if both meshes match, else SDF" :
                         morphMethod == MorphMethod::SharedTopology ? "Fit the denser mesh onto the other" :
                                                                      "Distance-field blend: any genus / parts");
                ImGui::SliderFloat("Fit Stiffness", &fitStiffness, 0.1f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::SetItemTooltip("Shared topology: how stiffly the template keeps its shape while fitting");
                ImGui::Checkbox("Cotan Laplacian", &fitCotan);
                ImGui::SetItemTooltip("Shared topology: weight the shape-keeping term by triangle angles\n"
                                      "(cotan Laplacian) instead of treating every neighbour the same");
                if (ImGui::SliderInt("SDF Resolution", &sdfResolution, 32, 160)) autoQuality = false;
                ImGui::SetItemTooltip("SDF: grid points per axis (higher = finer, slower)");
            } else {
                if (ImGui::SliderInt("Splats", &splatCount, 1000, 500000, "%d", ImGuiSliderFlags_Logarithmic))
                    autoQuality = false;
                ImGui::SliderFloat("Splat Size", &splatSizeMul, 0.3f, 3.0f);
                ImGui::SliderFloat("Cohesion", &splatCohesion, 0.0f, 1.0f);
                ImGui::SetItemTooltip("0 = every splat flies straight\n1 = neighbours move together as chunks");
                ImGui::Checkbox("Area-preserving splats", &splatAreaPreserve);
                ImGui::SetItemTooltip("Mid-flight, grow or shrink each splat by how much the OT map\n"
                                      "stretches its patch of surface (exact size at both ends)");
                if (ImGui::SliderInt("Mixture K", &mixtureK, 1, 128)) autoQuality = false;
                ImGui::SetItemTooltip("Gaussians fitted per shape for the OT map\n1 = original single-Gaussian map");
                ImGui::SliderFloat("Sinkhorn eps", &sinkhornEps, 0.005f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::SetItemTooltip("Softness of the cluster-to-cluster transport plan");
                ImGui::SliderFloat("Blend Temp", &posteriorTemp, 0.25f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::SetItemTooltip("How smoothly each splat blends between nearby clusters");
                ImGui::SliderFloat("Color Match", &colorMatch, 0.0f, 2.0f, "%.2f");
                ImGui::SetItemTooltip("0 = splats pair up by position only\nHigher = similar colours pair up\n(red parts travel to red parts)");
            }
            if (ImGui::SliderInt("Mesh Samples", &sampleCount, 500, 20000)) autoQuality = false;
            ImGui::SetItemTooltip("Surface points used to fit the Gaussians for the OT map");
            ImGui::TreePop();
        }
    }

    // ---- 5. Performance ----
    if (ImGui::CollapsingHeader("Performance")) {
        ImGui::BeginDisabled(!gpuAvailable);
        if (ImGui::Checkbox("GPU playback", &useGpu)) {
            g_morphShown = MorphShown::None;   // rebuild the current frame on the other path
            g_lastSdfT   = -1.0f;
        }
        ImGui::EndDisabled();
        ImGui::SetItemTooltip(gpuAvailable
            ? "On: per-frame morph work runs on the GPU (vertex/compute shaders).\nOff: the CPU path, for comparison."
            : "Needs OpenGL 4.3 (compute shaders)");

        auto ms = [](double v) {
            static char buf[4][32];
            static int k = 0;
            k = (k + 1) % 4;
            if (v < 0.0) snprintf(buf[k], sizeof(buf[k]), "-");
            else         snprintf(buf[k], sizeof(buf[k]), "%.3f ms", v);
            return buf[k];
        };
        ImGui::Text("Frame        %.2f ms (%.0f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::TextDisabled("Auto quality budget: bake <= %.0f s, GPU frame <= %.0f ms",
                            kBakeBudgetMs / 1000.0, kFrameBudgetMs);
        ImGui::Text("CPU update   %s", ms(g_cpuUpdateMs));
        ImGui::Text("CPU draw     %s", ms(g_cpuDrawMs));
        bool gpuSplats = renderMode == RenderMode::Splats && splatMorphActive && SplatsOnGpu();
        if (gpuSplats) {
            ImGui::Text("GPU prep+sort %s", ms(splatRenderer->PrepSortMs()));
            ImGui::Text("GPU draw     %s", ms(splatRenderer->DrawMs()));
        } else {
            if (renderMode == RenderMode::TriangleMorph && trueMorphActive && !g_morphShared)
                ImGui::Text("GPU re-mesh  %s", g_sdfOnGpu ? ms(g_morphTimer->Ms()) : "- (CPU)");
            ImGui::Text("GPU draw     %s", ms(g_drawTimer->Ms()));
        }
        if (renderMode == RenderMode::TriangleMorph && trueMorphActive) {
            bool gpuMesh = g_sdfOnGpu && g_morphShown == MorphShown::Blend;
            ImGui::Text("Surface      %d verts, %d tris",
                        gpuMesh ? g_gpuSdf->LastVertexCount() : meshRenderer->VertexCount(),
                        gpuMesh ? g_gpuSdf->LastTriangleCount() : meshRenderer->TriangleCount());
        } else if (renderMode == RenderMode::Splats) {
            ImGui::Text("Splats       %d", gpuSplats ? (int)splatStart.size() : (int)splatCurrent.size());
        }

        if (ImGui::Button("Verify GPU vs CPU")) VerifyGpuAgainstCpu();
        ImGui::SetItemTooltip("Topology (SDF): compare GPU and CPU mesh sizes at this t\n"
                              "Splats: check the GPU depth sort");
        if (!g_verifyText.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            ImGui::TextWrapped("%s", g_verifyText.c_str());
            ImGui::PopStyleColor();
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("%.0f FPS", ImGui::GetIO().Framerate);

    ImGui::PopItemWidth();
    ImGui::End();
}

// helper to reset the camera
void Window::resetCamera() {
    Cam->Reset();
    Cam->SetAspect(float(Window::width) / float(Window::height));
}

//key callbacks - for Interaction
void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT ) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_R:
                resetCamera();
                break;
            default:
                break;
        }
    }
}

void Window::mouse_callback(GLFWwindow* window, int button, int action, int mods) {
    // Only start a drag outside the GUI, but always end it, even when the
    // button is released over a GUI window.
    bool down = (action == GLFW_PRESS) && !ImGui::GetIO().WantCaptureMouse;
    if (button == GLFW_MOUSE_BUTTON_LEFT)  LeftDown  = down;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) RightDown = down;
}

void Window::cursor_callback(GLFWwindow* window, double currX, double currY) {
    if(ImGui::GetIO().WantCaptureMouse){
        MouseX = (int)currX;
        MouseY = (int)currY;
        return;
    }

    int maxDelta = 100;
    int dx = glm::clamp((int)currX - MouseX, -maxDelta, maxDelta);
    int dy = glm::clamp(-((int)currY - MouseY), -maxDelta, maxDelta);

    MouseX = (int)currX;
    MouseY = (int)currY;

    if (LeftDown) {
        const float rate = 1.0f;
        Cam->SetAzimuth(Cam->GetAzimuth() + dx * rate);
        Cam->SetIncline(glm::clamp(Cam->GetIncline() - dy * rate, -90.0f, 90.0f));
    }
    if (RightDown) {
        const float rate = 0.005f;
        float dist = glm::clamp(Cam->GetDistance() * (1.0f - dx * rate), 0.01f, 1000.0f);
        Cam->SetDistance(dist);
    }
}
