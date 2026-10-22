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

#include <glm/gtc/quaternion.hpp>
#include <chrono>
#include <atomic>
#include <execution>
#include <future>
#include <numeric>
#include <cmath>

#include <filesystem>
#include <algorithm>
#include <unordered_map>

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
bool        Window::trueMorphActive = false;
bool        Window::morphBaking     = false;
std::string Window::morphStatus;

//GPU playback
bool Window::gpuAvailable = false;
bool Window::useGpu       = false;

// Real endpoint meshes (shown exactly at t = 0 and t = 1) and the in-between data.
static IndexedMesh               g_morphA, g_morphB;
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
static float g_splatE = 0.0f, g_splatG = 0.0f, g_splatGrow = 1.0f;

static bool SplatsOnGpu() {
    return Window::useGpu && Window::gpuAvailable && Window::splatRenderer->HasKeyframes();
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
float Window::splatCohesion    = 0.6f;
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
static bool LoadNormalized(const std::string& path, std::vector<Triangle>& tris) {
    if (!OBJLoader::Load(path, tris)) return false;
    OBJLoader::Normalize(tris);
    return true;
}

// ---- Static previews for Particles / Topology modes ----
// Until a morph is baked, Topology shows mesh A as a solid surface and
// Particles shows particles sampled on A (or the default blob). Rebuilt only
// when the source, particle count, mode or g_previewGen changes.
static MeshRenderer* g_previewMesh = nullptr;
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
    static std::vector<Triangle> tris;
    static int loadedIndex = -1;
    int idx = Window::meshIndexA;
    bool hasMesh = idx > 0 && idx <= (int)Window::meshPaths.size();
    if (hasMesh && loadedIndex != idx) {
        if (LoadNormalized(Window::meshPaths[idx - 1], tris)) {
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
            if (entry.is_regular_file() &&
                entry.path().extension() == ".obj") {
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
    delete g_previewMesh;
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

    // GPU splats time their own passes (timer queries can't nest)
    bool gpuSplats = renderMode == RenderMode::Splats && splatMorphActive && SplatsOnGpu();
    auto cpuStart = std::chrono::steady_clock::now();
    if (!gpuSplats) g_drawTimer->Begin();

    if (renderMode == RenderMode::Splats) {
        if (!splatMorphActive) UpdateSplatPreview();
        if (gpuSplats)
            splatRenderer->DrawMorph(g_splatE, g_splatG, g_splatGrow, Cam->GetViewMtx(), Cam->GetProjectMtx());
        else
            splatRenderer->Draw(splatCurrent, Cam->GetViewMtx(), Cam->GetProjectMtx());
    } else if (renderMode == RenderMode::TriangleMorph && trueMorphActive) {
        if (g_sdfOnGpu && g_morphShown == MorphShown::Blend)
            g_gpuSdf->Draw(shaderProgram, Cam->GetViewProjectMtx(), meshRenderer->diffuseColor);
        else
            meshRenderer->Draw(meshRenderer->IsMorphTarget() ? morphProgram : shaderProgram,
                               Cam->GetViewProjectMtx());
    } else if (renderMode == RenderMode::TriangleMorph && meshIndexA > 0) {
        // Topology preview: the source mesh as a solid surface
        UpdateModePreview();
        if (g_previewMesh && g_previewMesh->isInitialized)
            g_previewMesh->Draw(shaderProgram, Cam->GetViewProjectMtx());
    } else {
        if (!particleMorphActive) UpdateModePreview();
        particleSystem->Draw(Cam->GetViewProjectMtx());
    }

    if (!gpuSplats) g_drawTimer->End();
    SmoothMs(g_cpuDrawMs, MsSince(cpuStart));

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
};

struct SplatBakeResult {
    bool ok = false;
    std::vector<Splat> start, end;   // start[i] travels to end[i]
    std::vector<glm::vec3> smoothDisp;   // neighbour-averaged end - start
    std::vector<int> knn;            // kSplatNeighbors nearest A-neighbours per splat (-1 = none)
    double ms = 0.0;
    double meanMatchDist = 0.0;      // rms |T(a_i) - b_match(i)| after refinement
    double nearestBound  = 0.0;      // rms distance to the nearest b (no one-to-one)
};

// Per-mesh work that doesn't depend on the other mesh, kept between bakes.
// Only the bake thread touches it, and only one bake runs at a time.
struct SplatMeshCache {
    std::vector<Triangle> tris;
    int mixK = -1, mixSamples = -1;      // settings the mixture was fit with
    GaussianMixture mix;
};
static std::unordered_map<std::string, SplatMeshCache> g_meshCache;

static SplatMeshCache* GetMeshCache(const std::string& path) {
    auto it = g_meshCache.find(path);
    if (it != g_meshCache.end()) return &it->second;
    SplatMeshCache c;
    if (!LoadNormalized(path, c.tris)) return nullptr;
    return &g_meshCache.emplace(path, std::move(c)).first->second;
}

static void EnsureMixture(SplatMeshCache& c, int K, int samples) {
    if (c.mixK == K && c.mixSamples == samples) return;
    c.mix.Fit(MeshSampler::Sample(c.tris, samples), K);
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

// One-to-one matching by recursive median splits: split both point sets at
// their median along the longest axis of the region, so each half of y gets
// the same number of b points, and recurse. Small leaves are matched by
// sorting along their longest axis (exact in 1D). Every b is used once.
static void SplitMatch(const std::vector<glm::vec3>& y, const std::vector<glm::vec3>& b,
                       int* iy, int* ib, int n, std::vector<int>& perm, int depth) {
    glm::vec3 lo(1e30f), hi(-1e30f);
    for (int k = 0; k < n; ++k) {
        lo = glm::min(lo, glm::min(y[iy[k]], b[ib[k]]));
        hi = glm::max(hi, glm::max(y[iy[k]], b[ib[k]]));
    }
    glm::vec3 ext = hi - lo;
    int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);

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

static std::vector<int> MedianSplitMatch(const std::vector<glm::vec3>& y,
                                         const std::vector<glm::vec3>& b) {
    const int n = (int)y.size();
    std::vector<int> iy(n), ib(n), perm(n);
    std::iota(iy.begin(), iy.end(), 0);
    std::iota(ib.begin(), ib.end(), 0);
    SplitMatch(y, b, iy.data(), ib.data(), n, perm, 0);
    return perm;
}

// Local improvement: for each i, try swapping partners with the owners of
// nearby b points; keep a swap when it lowers the total squared distance.
// Returns the number of passes run.
static int RefineMatch(const std::vector<glm::vec3>& y, const std::vector<glm::vec3>& b,
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
                float before = Dist2(y[i], b[j])  + Dist2(y[i2], b[j2]);
                float after  = Dist2(y[i], b[j2]) + Dist2(y[i2], b[j]);
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

static SplatBakeResult RunSplatBake(SplatBakeSettings st) {
    SplatBakeResult res;
    auto t0 = std::chrono::steady_clock::now();

    SetBakeStage(0, 2);
    SplatMeshCache* A = GetMeshCache(st.pathA);
    g_bakeDone++;
    SplatMeshCache* B = GetMeshCache(st.pathB);
    g_bakeDone++;
    if (!A || !B) return res;

    // Same count on both surfaces; each surface sizes its own splats from its
    // area, so both ends are fully covered.
    std::vector<Splat> splatsA = SplatGenerator::FromTriangles(A->tris, st.splatCount, st.sizeMul, 42u);
    std::vector<Splat> splatsB = SplatGenerator::FromTriangles(B->tris, st.splatCount, st.sizeMul, 43u);
    const int n = (int)std::min(splatsA.size(), splatsB.size());
    splatsA.resize(n);
    splatsB.resize(n);

    // Mixture fit on surface samples; K = 1 reproduces the original single map
    SetBakeStage(1, 1);
    ForBoth(*A, *B, [&](SplatMeshCache& c) { EnsureMixture(c, st.mixtureK, st.sampleCount); });

    SetBakeStage(2, 1);
    MixtureOTMap map = OptimalTransport::ComputeMixture(A->mix, B->mix, st.sinkhornEps, st.posteriorTemp);

    // Coarse: the mixture map sends each A splat roughly where its part of A
    // should go on B (keeps "which part goes where").
    SetBakeStage(3, n);
    std::vector<glm::vec3> y(n), bPos(n);
    {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            Eigen::Matrix3d J;
            OptimalTransport::ApplyMixture(map, splatsA[i].mean, y[i], J);
            bPos[i] = splatsB[i].mean;
        });
        g_bakeDone = n;
    }

    // Fine: one-to-one matching of the mapped points to B's splats
    const int maxPasses = 20;
    SetBakeStage(4, n + maxPasses);
    std::vector<int> perm = MedianSplitMatch(y, bPos);
    double splitCost = MatchCost(y, bPos, perm);
    g_bakeDone = n;
    float cell = 2.0f * st.sizeMul * SplatGenerator::BaseSigma(B->tris, n);
    int passes = RefineMatch(y, bPos, perm, cell, maxPasses);
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
        g.Build(aPos, 2.0f * st.sizeMul * SplatGenerator::BaseSigma(A->tris, n));

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
    // user pressed Clear or left Splats mode while the bake was running
    if (g_bakeLaunchEpoch != g_bakeEpoch || renderMode != RenderMode::Splats) return;

    splatStart   = std::move(res.start);
    splatEnd        = std::move(res.end);
    splatSmoothDisp = std::move(res.smoothDisp);
    splatCurrent    = splatStart;

    // GPU path: the keyframes go up once; each frame then only sends the time
    if (gpuAvailable) splatRenderer->UploadKeyframes(splatStart, splatEnd, splatSmoothDisp);

    // The animator only provides the timeline (easing, duration, state)
    animator->InitializeWithEndPoints({}, {});
    animator->StartTransport();
    splatMorphActive    = true;
    trueMorphActive     = false;
    particleMorphActive = false;
    lastBakeMs          = (float)res.ms;

    std::cout << "[Splats] baked " << splatStart.size() << " matched splats, K=" << mixtureK
              << ", mean match distance " << res.meanMatchDist << ", " << res.ms << " ms\n";
}

// Show mesh A (or the default blob) as static splats until a morph is baked.
// Regenerates only when the inputs change.
void Window::UpdateSplatPreview() {
    static int   lastIndex = -1, lastCount = -1, lastGen = -1;
    static float lastSize  = -1.0f;
    if (lastIndex == meshIndexA && lastCount == splatCount &&
        lastSize == splatSizeMul && lastGen == splatPreviewGen)
        return;
    lastIndex = meshIndexA; lastCount = splatCount;
    lastSize = splatSizeMul; lastGen = splatPreviewGen;

    if (meshIndexA > 0 && meshIndexA <= (int)meshPaths.size()) {
        // keep the loaded mesh so dragging the count/size sliders doesn't reload it
        static std::vector<Triangle> tris;
        static int loadedIndex = -1;
        if (loadedIndex != meshIndexA) {
            loadedIndex = -1;
            if (!LoadNormalized(meshPaths[meshIndexA - 1], tris)) return;
            loadedIndex = meshIndexA;
        }
        splatCurrent = SplatGenerator::FromTriangles(tris, splatCount, splatSizeMul, 42u);
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
static bool LoadMorphMesh(const std::string& path, IndexedMesh& m, TopologyInfo& info) {
    std::vector<Triangle> tris;
    if (!LoadNormalized(path, tris)) return false;
    MeshTopology::Build(tris, m);
    int filled = MeshTopology::FillSmallHoles(m, 40);
    MeshTopology::OrientOutward(m);
    info = MeshTopology::Analyze(m);
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
    }
    if (a.euler != b.euler)
        return "different genus (chi " + std::to_string(a.euler) + " vs " + std::to_string(b.euler) + ")";
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
};

struct TrueMorphResult {
    bool ok = false;
    IndexedMesh a, b;                    // cleaned endpoint meshes
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
    if (!LoadMorphMesh(st.pathA, res.a, infoA)) return res;
    g_bakeDone++;
    if (!LoadMorphMesh(st.pathB, res.b, infoB)) return res;
    g_bakeDone++;
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
        FitStats stats;
        std::vector<glm::vec3> fitted = MeshFitter::Fit(tmpl, aligned,
                                                        templateIsA ? trisB : trisA, params, &stats);

        res.sharedIndices = tmpl.indices;
        res.sharedStart   = templateIsA ? tmpl.verts : fitted;
        res.sharedEnd     = templateIsA ? fitted : tmpl.verts;

        snprintf(buf, sizeof(buf),
                 "Shared topology: %s (%d tris%s) fitted onto %s in %.0f ms. "
                 "Mean error %.4f, %d folded tris.%s%s",
                 (templateIsA ? nameA : nameB).c_str(), tmpl.FaceCount(),
                 subdivided ? ", subdivided" : "", (templateIsA ? nameB : nameA).c_str(),
                 stats.ms, stats.meanDist, stats.flipped,
                 mismatch.empty() ? "" : " Warning: ", mismatch.c_str());
    } else {
        res.sdf.Build(trisA, trisB, res.map, st.sdfResolution);

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
            meshRenderer->Initialize(m.verts, m.indices);
            g_morphShown = want;
        }
        return;
    }

    if (g_morphShared) {
        if (gpu) {
            if (g_morphShown != MorphShown::Blend || !meshRenderer->IsMorphTarget()) {
                meshRenderer->InitializeMorph(g_sharedStart, g_sharedEnd, g_sharedIndices);
                g_morphShown = MorphShown::Blend;
            }
            meshRenderer->morphT = e;
            return;
        }
        for (size_t i = 0; i < g_sharedCurrent.size(); ++i)
            g_sharedCurrent[i] = g_sharedStart[i] + e * (g_sharedEnd[i] - g_sharedStart[i]);
        if (g_morphShown != MorphShown::Blend || meshRenderer->IsMorphTarget()) {
            meshRenderer->Initialize(g_sharedCurrent, g_sharedIndices);
            g_morphShown = MorphShown::Blend;
        } else {
            meshRenderer->UpdatePositions(g_sharedCurrent);
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
        std::vector<glm::vec3> verts;
        std::vector<unsigned int> indices;
        g_sdfMorph.Extract(t, verts, indices);
        if (!indices.empty()) meshRenderer->Initialize(verts, indices);
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
    bool baking  = splatBaking || morphBaking;
    bool canBake = !baking && (!needsMeshes || (hasA && hasB));

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
        if (renderMode == RenderMode::Particles) {
            ImGui::SliderInt("Particles", &particleCount, 100, 10000);
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
            ImGui::SliderInt("SDF Resolution", &sdfResolution, 32, 160);
            ImGui::SetItemTooltip("SDF: grid points per axis (higher = finer, slower)");
        } else {
            ImGui::SliderInt("Splats", &splatCount, 1000, 500000, "%d", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Splat Size", &splatSizeMul, 0.3f, 3.0f);
            ImGui::SliderFloat("Cohesion", &splatCohesion, 0.0f, 1.0f);
            ImGui::SetItemTooltip("0 = every splat flies straight\n1 = neighbours move together as chunks");
            ImGui::SliderInt("Mixture K", &mixtureK, 1, 128);
            ImGui::SetItemTooltip("Gaussians fitted per shape for the OT map\n1 = original single-Gaussian map");
            ImGui::SliderFloat("Sinkhorn eps", &sinkhornEps, 0.005f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SetItemTooltip("Softness of the cluster-to-cluster transport plan");
            ImGui::SliderFloat("Blend Temp", &posteriorTemp, 0.25f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SetItemTooltip("How smoothly each splat blends between nearby clusters");
        }
        ImGui::SliderInt("Mesh Samples", &sampleCount, 500, 20000);
        ImGui::SetItemTooltip("Surface points used to fit the Gaussians for the OT map");
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
