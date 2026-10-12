#include "Window.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "OBJLoader.h"
#include "MeshSampler.h"
#include "SplatGenerator.h"
#include "GaussianMixture.h"

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
ParticleSystem* Window::particleSystem     = nullptr;
MeshRenderer*   Window::meshRenderer       = nullptr;
MeshRenderer*   Window::meshRendererTarget = nullptr;

//Render mode
RenderMode Window::renderMode = RenderMode::TriangleMorph;

//Triangle-morph state
std::vector<glm::vec3>    Window::meshStartPositions;
std::vector<glm::vec3>    Window::meshEndPositions;
std::vector<unsigned int> Window::meshIndices;
bool                      Window::meshMorphActive = false;

//Dual-mesh (Option B) target-side state
std::vector<glm::vec3>    Window::meshTargetStart;
std::vector<glm::vec3>    Window::meshTargetEnd;
std::vector<glm::vec3>    Window::meshTargetCurrent;
std::vector<unsigned int> Window::meshTargetIndices;

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
bool             Window::otMapComputed = false;
bool             Window::particleMorphActive = false;
int              Window::particleCount = 5000;
std::vector<std::string> Window::meshNames;
std::vector<std::string> Window::meshPaths;
int              Window::meshIndexA    = 0;
int              Window::meshIndexB    = 0;
int              Window::sampleCount        = 5000;
int              Window::subdivisionLevel   = 0;
float            Window::crossfadeStart     = 0.35f;

//Interaction Variables
bool LeftDown, RightDown;
int MouseX, MouseY;

//The shader program id
GLuint Window::shaderProgram;

// Default blob: isotropic 3D Gaussian (no color, no named presets)
static const Eigen::Matrix3d blobCovariance = Eigen::Vector3d(1.0, 1.0, 1.0).asDiagonal();

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

    if (!shaderProgram) {
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
    particleSystem     = new ParticleSystem();
    meshRenderer       = new MeshRenderer();
    meshRendererTarget = new MeshRenderer();
    splatRenderer      = new SplatRenderer();
    if (!splatRenderer->Initialize()) return false;

    glm::vec3 meshColor(0.85f, 0.55f, 0.75f);  // shared pink for both meshes
    meshRenderer->diffuseColor       = meshColor;
    meshRendererTarget->diffuseColor = meshColor;

    gaussianA = new Gaussian();
    gaussianB = new Gaussian();
    otMap = nullptr;
    animator = new TransportAnimator();
    otMapComputed = false;
    particleCount = 3000;

    ScanMeshes();

    // Defaults: source = Guassian.obj, target = teapot.obj
    meshIndexA = FindMeshIndex("Guassian.obj");
    meshIndexB = FindMeshIndex("teapot.obj");

    Cam->SetDistance(8.0f);
    Cam->SetIncline(-20.0f);

    // Spawn initial blob so the scene isn't empty on startup
    gaussianA->mean       = Eigen::Vector3d::Zero();
    gaussianA->covariance = blobCovariance;
    particleSystem->SpawnFromGaussian(*gaussianA, particleCount);

    return true;
}

void Window::cleanUp() {
    delete particleSystem;
    delete meshRenderer;
    delete meshRendererTarget;
    delete splatRenderer;
    delete gaussianA;
    delete gaussianB;
    delete otMap;
    delete animator;
    glDeleteProgram(shaderProgram);
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

    PollSplatBake();

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

        for (size_t i = 0; i < splatCurrent.size(); ++i) {
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
    } else if (meshMorphActive) {
        // Source mesh: animator advances the shared timeline and writes
        // lerp'd source-vertex positions into meshStartPositions.
        animator->Update(deltaTime, meshStartPositions);
        meshRenderer->UpdatePositions(meshStartPositions);

        // Target mesh: manually lerp using the same easing curve so it stays
        // perfectly in sync with the source.
        if (!meshTargetCurrent.empty()) {
            float e = animator->GetBlend();
            for (size_t i = 0; i < meshTargetCurrent.size(); ++i)
                meshTargetCurrent[i] = glm::mix(meshTargetStart[i], meshTargetEnd[i], e);
            meshRendererTarget->UpdatePositions(meshTargetCurrent);
        }
    } else if (particleMorphActive) {
        animator->Update(deltaTime, particleSystem->transportPositions);
    }

    // after the animator, so particles show this frame's transport positions
    particleSystem->Update(deltaTime);
}

void Window::displayCallback(GLFWwindow* window) {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    // glClearColor(83/255.0f, 203/255.0f, 243/255.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLfloat light0_pos[] = {10.0f, 10.0f, 10.0f, 1.0f};
    GLfloat light0_color[] = {1.0f, 0.8f, 0.8f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light0_pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_color);

    GLfloat light1_pos[] = {-10.0f, 5.0f, -10.0f, 1.0f};
    GLfloat light1_color[] = {0.8f, 0.8f, 1.0f, 1.0f};
    glLightfv(GL_LIGHT1, GL_POSITION, light1_pos);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_color);

    if (renderMode == RenderMode::Splats) {
        if (!splatMorphActive) UpdateSplatPreview();
        splatRenderer->Draw(splatCurrent, Cam->GetViewMtx(), Cam->GetProjectMtx());
    } else if (renderMode == RenderMode::TriangleMorph && meshMorphActive) {
        float e = animator->GetBlend();

        // Delayed crossfade: source stays fully opaque until crossfadeStart,
        // then both meshes smoothstep-fade across the remaining window.
        float tail = 1.0f - crossfadeStart;
        float t = (tail > 1e-4f)
                ? glm::clamp((e - crossfadeStart) / tail, 0.0f, 1.0f)
                : (e >= crossfadeStart ? 1.0f : 0.0f);
        t = t * t * (3.0f - 2.0f * t);  // smoothstep
        meshRenderer->alpha       = 1.0f - t;
        meshRendererTarget->alpha = t;

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (meshRenderer->alpha >= meshRendererTarget->alpha) {
            glDepthMask(GL_TRUE);
            meshRenderer->Draw(shaderProgram, Cam->GetViewProjectMtx());
            glDepthMask(GL_FALSE);
            meshRendererTarget->Draw(shaderProgram, Cam->GetViewProjectMtx());
        } else {
            glDepthMask(GL_TRUE);
            meshRendererTarget->Draw(shaderProgram, Cam->GetViewProjectMtx());
            glDepthMask(GL_FALSE);
            meshRenderer->Draw(shaderProgram, Cam->GetViewProjectMtx());
        }

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
    } else {
        particleSystem->Draw(Cam->GetViewProjectMtx());
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
static bool LoadMeshGaussian(const char* path, int N, Gaussian* g) {
    std::vector<Triangle> tris;
    if (!OBJLoader::Load(path, tris)) return false;
    OBJLoader::Normalize(tris);
    auto pts = MeshSampler::Sample(tris, N);
    g->FitFromPoints(pts);
    return true;
}

// Midpoint subdivision: split every triangle into 4 smaller ones.
// Increasing levels: 1→4x, 2→16x, 3→64x triangles. Applied to source mesh
// only to add vertices for better endpoint coverage after surface snapping.
static void SubdivideTriangles(std::vector<Triangle>& tris, int levels) {
    for (int l = 0; l < levels; ++l) {
        std::vector<Triangle> result;
        result.reserve(tris.size() * 4);
        for (const auto& t : tris) {
            glm::vec3 m01 = (t.v0 + t.v1) * 0.5f;
            glm::vec3 m12 = (t.v1 + t.v2) * 0.5f;
            glm::vec3 m20 = (t.v2 + t.v0) * 0.5f;
            result.push_back({t.v0, m01, m20});
            result.push_back({m01, t.v1, m12});
            result.push_back({m20, m12, t.v2});
            result.push_back({m01, m12, m20});
        }
        tris = std::move(result);
    }
}

// Deduplicate vertices from an OBJ triangle soup into a unique-vertex list +
// triangle index list. Uses exact float equality via bit-hashed vec3 keys.
// This is required because glDrawElements(GL_TRIANGLES) expects an EBO.
static void BuildMeshTopology(const std::vector<Triangle>& tris,
                              std::vector<glm::vec3>& outVerts,
                              std::vector<unsigned int>& outIndices) {
    outVerts.clear();
    outIndices.clear();
    outIndices.reserve(tris.size() * 3);

    struct Vec3Hash {
        size_t operator()(const glm::vec3& v) const {
            // Hash the raw float bits so structurally identical positions collide.
            auto h = std::hash<float>();
            size_t a = h(v.x), b = h(v.y), c = h(v.z);
            return a ^ (b * 0x9E3779B97F4A7C15ULL) ^ (c * 0xBF58476D1CE4E5B9ULL);
        }
    };
    struct Vec3Eq {
        bool operator()(const glm::vec3& a, const glm::vec3& b) const {
            return a.x == b.x && a.y == b.y && a.z == b.z;
        }
    };

    std::unordered_map<glm::vec3, unsigned int, Vec3Hash, Vec3Eq> lookup;
    lookup.reserve(tris.size() * 3);

    auto addVertex = [&](const glm::vec3& v) -> unsigned int {
        auto it = lookup.find(v);
        if (it != lookup.end()) return it->second;
        unsigned int idx = (unsigned int)outVerts.size();
        outVerts.push_back(v);
        lookup.emplace(v, idx);
        return idx;
    };

    for (const auto& t : tris) {
        outIndices.push_back(addVertex(t.v0));
        outIndices.push_back(addVertex(t.v1));
        outIndices.push_back(addVertex(t.v2));
    }
}

// Phase 3: closest point on triangle (a, b, c) to a query point p.
// Ericson, "Real-Time Collision Detection" §5.1.5 — Voronoi-region test.
// Returns a point on the triangle surface (interior or boundary edge/vertex).
static glm::vec3 ClosestPointOnTriangle(const glm::vec3& p,
                                        const glm::vec3& a,
                                        const glm::vec3& b,
                                        const glm::vec3& c) {
    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 ap = p - a;

    float d1 = glm::dot(ab, ap);
    float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;                   // vertex A region

    glm::vec3 bp = p - b;
    float d3 = glm::dot(ab, bp);
    float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;                     // vertex B region

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {             // edge AB
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    glm::vec3 cp = p - c;
    float d5 = glm::dot(ab, cp);
    float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;                     // vertex C region

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {             // edge AC
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) { // edge BC
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    // Interior projection via barycentric coordinates.
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + v * ab + w * ac;
}

// Phase 3 snap: for each endpoint, find the closest point on the target
// surface. Uses the SurfaceGrid below instead of testing every triangle.
static void SnapToNearestOnTriangles(std::vector<glm::vec3>& endVerts,
                                     const std::vector<Triangle>& tris);

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

// Uniform grid over dense surface samples, for fast approximate closest-point
// queries on large meshes (the brute-force snap is O(points * triangles)).
// Each query finds the nearest sample, then projects exactly onto that
// sample's triangle.
struct SurfaceGrid {
    std::vector<glm::vec3> pts;
    std::vector<int>       tri;        // triangle each sample came from
    std::vector<int>       cellStart;  // CSR layout: cellStart[c]..cellStart[c+1]
    std::vector<int>       cellItems;
    glm::vec3 lo = glm::vec3(0.0f);
    float cell = 1.0f;
    int nx = 1, ny = 1, nz = 1;

    void Build(const std::vector<Triangle>& tris, int M) {
        std::vector<glm::vec3> samples = MeshSampler::Sample(tris, M, 7u, tri);
        // ~2 sample spacings per cell
        BuildFromPoints(samples, 2.0f * std::sqrt(MeshSampler::TotalArea(tris) / (float)M));
    }

    // Grid over arbitrary points; cellSize is capped so the grid stays small.
    void BuildFromPoints(const std::vector<glm::vec3>& points, float cellSize) {
        pts = points;
        lo = glm::vec3(1e30f);
        glm::vec3 hi(-1e30f);
        for (const auto& p : pts) { lo = glm::min(lo, p); hi = glm::max(hi, p); }

        cell = std::max(1e-4f, cellSize);
        glm::vec3 ext = hi - lo;
        cell = std::max(cell, std::max(ext.x, std::max(ext.y, ext.z)) / 128.0f);
        nx = (int)(ext.x / cell) + 1;
        ny = (int)(ext.y / cell) + 1;
        nz = (int)(ext.z / cell) + 1;

        std::vector<int> count(nx * ny * nz + 1, 0);
        std::vector<int> cellOf(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            cellOf[i] = CellIndex(pts[i]);
            count[cellOf[i] + 1]++;
        }
        for (size_t c = 1; c < count.size(); ++c) count[c] += count[c - 1];
        cellStart = count;
        cellItems.resize(pts.size());
        std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
        for (size_t i = 0; i < pts.size(); ++i) cellItems[fill[cellOf[i]]++] = (int)i;
    }

    glm::ivec3 Cell(const glm::vec3& p) const {
        glm::ivec3 c = glm::ivec3(glm::floor((p - lo) / cell));
        return glm::clamp(c, glm::ivec3(0), glm::ivec3(nx - 1, ny - 1, nz - 1));
    }
    int CellIndex(const glm::vec3& p) const {
        glm::ivec3 c = Cell(p);
        return (c.z * ny + c.y) * nx + c.x;
    }

    // Index of the nearest sample. Searches shells of cells outward and stops
    // once the next shell can't contain anything closer.
    int Nearest(const glm::vec3& q) const {
        glm::ivec3 c0 = Cell(q);
        // distance from q to the grid box (q may lie outside it)
        glm::vec3 hiBox = lo + cell * glm::vec3((float)nx, (float)ny, (float)nz);
        float outside = glm::length(glm::max(glm::vec3(0.0f), glm::max(lo - q, q - hiBox)));

        int best = 0;
        float bestD = 1e30f;
        int maxR = std::max(nx, std::max(ny, nz));
        for (int r = 0; r <= maxR; ++r) {
            for (int z = c0.z - r; z <= c0.z + r; ++z) {
                if (z < 0 || z >= nz) continue;
                for (int y = c0.y - r; y <= c0.y + r; ++y) {
                    if (y < 0 || y >= ny) continue;
                    for (int x = c0.x - r; x <= c0.x + r; ++x) {
                        if (x < 0 || x >= nx) continue;
                        // only the shell of this ring
                        int ring = std::max(std::abs(x - c0.x), std::max(std::abs(y - c0.y), std::abs(z - c0.z)));
                        if (ring != r) continue;
                        int c = (z * ny + y) * nx + x;
                        for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                            glm::vec3 d = pts[cellItems[k]] - q;
                            float dd = glm::dot(d, d);
                            if (dd < bestD) { bestD = dd; best = cellItems[k]; }
                        }
                    }
                }
            }
            // anything in shell r+1 is at least sqrt(outside^2 + (r*cell)^2) away
            float reach2 = outside * outside + (r * cell) * (r * cell);
            if (bestD < 1e29f && bestD <= reach2) break;
        }
        return best;
    }
};

// Closest point on the surface near p: find the nearest dense sample, then
// project exactly onto every triangle sampled in the 3x3x3 cells around it.
static glm::vec3 SnapPoint(const SurfaceGrid& g, const std::vector<Triangle>& tris,
                           const glm::vec3& p, int* outTri = nullptr) {
    int s0 = g.Nearest(p);
    int bestTri = g.tri[s0];
    const Triangle& t0 = tris[bestTri];
    glm::vec3 best = ClosestPointOnTriangle(p, t0.v0, t0.v1, t0.v2);
    float bestD = glm::dot(best - p, best - p);

    glm::ivec3 c0 = g.Cell(g.pts[s0]);
    for (int z = std::max(0, c0.z - 1); z <= std::min(g.nz - 1, c0.z + 1); ++z)
        for (int y = std::max(0, c0.y - 1); y <= std::min(g.ny - 1, c0.y + 1); ++y)
            for (int x = std::max(0, c0.x - 1); x <= std::min(g.nx - 1, c0.x + 1); ++x) {
                int c = (z * g.ny + y) * g.nx + x;
                for (int k = g.cellStart[c]; k < g.cellStart[c + 1]; ++k) {
                    int ti = g.tri[g.cellItems[k]];
                    if (ti == bestTri) continue;
                    const Triangle& t = tris[ti];
                    glm::vec3 q = ClosestPointOnTriangle(p, t.v0, t.v1, t.v2);
                    float d = glm::dot(q - p, q - p);
                    if (d < bestD) { bestD = d; best = q; bestTri = ti; }
                }
            }
    if (outTri) *outTri = bestTri;
    return best;
}

static void SnapToNearestOnTriangles(std::vector<glm::vec3>& endVerts,
                                     const std::vector<Triangle>& tris) {
    if (tris.empty() || endVerts.empty()) return;
    SurfaceGrid grid;
    grid.Build(tris, std::clamp(8 * (int)endVerts.size(), 50000, 400000));
    std::for_each(std::execution::par, endVerts.begin(), endVerts.end(),
                  [&](glm::vec3& p) { p = SnapPoint(grid, tris, p); });
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
    if (!OBJLoader::Load(path.c_str(), c.tris)) return nullptr;
    OBJLoader::Normalize(c.tris);
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
    SurfaceGrid grid;
    grid.BuildFromPoints(b, cellSize);

    std::vector<int> owner(n);
    for (int i = 0; i < n; ++i) owner[perm[i]] = i;

    int pass = 0;
    for (; pass < maxPasses; ++pass) {
        int swaps = 0;
        for (int i = 0; i < n; ++i) {
            glm::ivec3 c0 = grid.Cell(y[i]);
            for (int z = std::max(0, c0.z - 1); z <= std::min(grid.nz - 1, c0.z + 1); ++z)
            for (int yy = std::max(0, c0.y - 1); yy <= std::min(grid.ny - 1, c0.y + 1); ++yy)
            for (int x = std::max(0, c0.x - 1); x <= std::min(grid.nx - 1, c0.x + 1); ++x) {
                int c = (z * grid.ny + yy) * grid.nx + x;
                for (int k = grid.cellStart[c]; k < grid.cellStart[c + 1]; ++k) {
                    int j2 = grid.cellItems[k];       // candidate b point
                    int i2 = owner[j2];               // its current partner
                    int j  = perm[i];
                    if (i2 == i) continue;
                    float before = Dist2(y[i], b[j])  + Dist2(y[i2], b[j2]);
                    float after  = Dist2(y[i], b[j2]) + Dist2(y[i2], b[j]);
                    if (after < before - 1e-12f) {
                        perm[i] = j2;  perm[i2] = j;
                        owner[j2] = i; owner[j] = i2;
                        ++swaps;
                    }
                }
            }
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
        SurfaceGrid g;
        g.BuildFromPoints(bPos, cell);
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
        SurfaceGrid g;
        g.BuildFromPoints(aPos, 2.0f * st.sizeMul * SplatGenerator::BaseSigma(A->tris, n));

        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            std::vector<std::pair<float, int>> cand;
            glm::ivec3 c0 = g.Cell(aPos[i]);
            for (int z = std::max(0, c0.z - 1); z <= std::min(g.nz - 1, c0.z + 1); ++z)
            for (int yy = std::max(0, c0.y - 1); yy <= std::min(g.ny - 1, c0.y + 1); ++yy)
            for (int x = std::max(0, c0.x - 1); x <= std::min(g.nx - 1, c0.x + 1); ++x) {
                int c = (z * g.ny + yy) * g.nx + x;
                for (int k = g.cellStart[c]; k < g.cellStart[c + 1]; ++k) {
                    int j = g.cellItems[k];
                    if (j != i) cand.push_back({ Dist2(aPos[i], aPos[j]), j });
                }
            }
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
static int g_splatBakeEpoch  = 0;   // bumped by Clear / leaving Splats mode
static int g_bakeLaunchEpoch = 0;   // epoch when the running bake started

void Window::BakeSplatMorph(const std::string& pathA, const std::string& pathB) {
    if (splatBaking) return;

    SplatBakeSettings st;
    st.pathA         = pathA;
    st.pathB         = pathB;
    st.splatCount    = splatCount;
    st.sampleCount   = sampleCount;
    st.mixtureK      = mixtureK;
    st.sizeMul       = splatSizeMul;
    st.sinkhornEps   = sinkhornEps;
    st.posteriorTemp = posteriorTemp;

    SetBakeStage(0, 1);
    splatBaking  = true;
    g_bakeLaunchEpoch = g_splatBakeEpoch;
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
    if (g_bakeLaunchEpoch != g_splatBakeEpoch || renderMode != RenderMode::Splats) return;

    splatStart   = std::move(res.start);
    splatEnd        = std::move(res.end);
    splatSmoothDisp = std::move(res.smoothDisp);
    splatCurrent    = splatStart;

    // The animator only provides the timeline (easing, duration, state)
    animator->InitializeWithEndPoints({}, {});
    animator->StartTransport();
    splatMorphActive    = true;
    meshMorphActive     = false;
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
            tris.clear();
            if (!OBJLoader::Load(meshPaths[meshIndexA - 1].c_str(), tris)) return;
            OBJLoader::Normalize(tris);
            loadedIndex = meshIndexA;
        }
        splatCurrent = SplatGenerator::FromTriangles(tris, splatCount, splatSizeMul, 42u);
    } else {
        // Blob: round splats sampled from the default isotropic Gaussian
        Gaussian blob;
        blob.covariance = blobCovariance;
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

    // === Triangle-morph mode (Phase 1 MVP) ===
    // Uses the source mesh's original triangulation. Each source vertex is
    // displaced by the Gaussian OT map T(x) = mu_B + A*(x - mu_A). Requires
    // both A and B to be actual meshes (not blobs).
    if (renderMode == RenderMode::TriangleMorph && hasMeshA && hasMeshB) {
        std::vector<Triangle> trisA, trisB;
        if (!OBJLoader::Load(pathA.c_str(), trisA)) return;
        if (!OBJLoader::Load(pathB.c_str(), trisB)) return;
        OBJLoader::Normalize(trisA);
        OBJLoader::Normalize(trisB);

        // Optional midpoint subdivision: adds more source vertices so snap
        // endpoints cover the target surface more densely.
        if (subdivisionLevel > 0)
            SubdivideTriangles(trisA, subdivisionLevel);

        // Extract the source mesh's actual vertices + triangle indices.
        std::vector<glm::vec3>    verts;
        std::vector<unsigned int> indices;
        BuildMeshTopology(trisA, verts, indices);

        // Fit Gaussians to both surfaces to build the OT map.
        // (Sample from A so the source Gaussian reflects A's surface distribution,
        //  which is what T is applied at.)
        auto sampledA = MeshSampler::Sample(trisA, sampleCount);
        auto sampledB = MeshSampler::Sample(trisB, sampleCount);
        gaussianA->FitFromPoints(sampledA);
        gaussianB->FitFromPoints(sampledB);

        delete otMap;
        otMap = new OTMap(OptimalTransport::Compute(*gaussianA, *gaussianB));
        otMapComputed = true;

        // Compute end positions by applying the OT map to each source vertex.
        std::vector<glm::vec3> endVerts(verts.size());
        for (size_t i = 0; i < verts.size(); ++i)
            endVerts[i] = OptimalTransport::Apply(*otMap, verts[i]);

        // Snap each source endpoint to the nearest point on any target triangle.
        SnapToNearestOnTriangles(endVerts, trisB);

        // === Option B: parallel target-topology mesh ===
        // Extract the target mesh's actual vertices + indices.
        std::vector<glm::vec3>    tVerts;
        std::vector<unsigned int> tIndices;
        BuildMeshTopology(trisB, tVerts, tIndices);

        // Compute A^-1 once. OT^-1(y) = mu_A + A^-1 * (y - mu_B).
        Eigen::Matrix3d Ainv = otMap->A.inverse();

        std::vector<glm::vec3> tStart(tVerts.size());
        for (size_t i = 0; i < tVerts.size(); ++i) {
            Eigen::Vector3d yE((double)tVerts[i].x, (double)tVerts[i].y, (double)tVerts[i].z);
            Eigen::Vector3d xE = otMap->muA + Ainv * (yE - otMap->muB);
            tStart[i] = glm::vec3((float)xE.x(), (float)xE.y(), (float)xE.z());
        }

        // Snap each target-start position onto the source surface.
        SnapToNearestOnTriangles(tStart, trisA);

        // Hand off to renderers + animator. Source animator drives the timeline;
        // target buffer is manually lerp'd each frame using animator.GetProgress().
        meshStartPositions = verts;
        meshEndPositions   = endVerts;
        meshIndices        = indices;

        meshTargetStart   = tStart;
        meshTargetEnd     = tVerts;
        meshTargetCurrent = tStart;       // t=0 initial state
        meshTargetIndices = tIndices;

        meshRenderer->Initialize(verts, indices);
        meshRendererTarget->Initialize(tStart, tIndices);

        animator->InitializeWithEndPoints(verts, endVerts);
        animator->StartTransport();
        meshMorphActive     = true;
        particleMorphActive = false;
        return;
    }

    // Non-triangle-morph or fallback: use the existing particle pipeline.
    meshMorphActive = false;

    if (hasMeshA && hasMeshB) {
        // === Discrete mesh-to-mesh OT ===
        // Particles start exactly on mesh A surface, end exactly on mesh B surface.
        // The Gaussian OT map is not used — the greedy matching IS the transport.
        std::vector<Triangle> trisA, trisB;
        if (!OBJLoader::Load(pathA.c_str(), trisA)) return;
        if (!OBJLoader::Load(pathB.c_str(), trisB)) return;
        OBJLoader::Normalize(trisA);
        OBJLoader::Normalize(trisB);

        auto ptsA = MeshSampler::Sample(trisA, particleCount);
        auto ptsB = MeshSampler::Sample(trisB, particleCount);

        auto endPts = GreedyDiscreteOT(ptsA, ptsB);

        // Place particles directly at mesh A surface positions
        particleSystem->particles.clear();
        particleSystem->transportPositions = ptsA;
        for (const auto& pos : ptsA) {
            Particle p(particleSystem->particleRadius);
            p.position = pos;
            particleSystem->particles.push_back(p);
        }

        animator->InitializeWithEndPoints(ptsA, endPts);
        particleSystem->useOTPositions = true;
        animator->StartTransport();
        particleMorphActive = true;

    } else {
        // === Gaussian OT (mesh → Gaussian or blob → blob) ===
        if (hasMeshA) {
            if (!LoadMeshGaussian(pathA.c_str(), sampleCount, gaussianA)) return;
        } else {
            gaussianA->mean       = Eigen::Vector3d::Zero();
            gaussianA->covariance = blobCovariance;
        }

        if (hasMeshB) {
            if (!LoadMeshGaussian(pathB.c_str(), sampleCount, gaussianB)) return;
        } else {
            gaussianB->mean       = Eigen::Vector3d::Zero();
            gaussianB->covariance = blobCovariance;
        }

        delete otMap;
        otMap         = new OTMap(OptimalTransport::Compute(*gaussianA, *gaussianB));
        otMapComputed = true;

        particleSystem->SpawnFromGaussian(*gaussianA, particleCount);
        animator->Initialize(particleSystem->transportPositions, *otMap);
        particleSystem->useOTPositions = true;
        animator->StartTransport();
        particleMorphActive = true;
    }
}

// Draw a mesh selector combo: first entry is the blob default, rest are scanned .obj files.
static void DrawGaussianSelector(const char* label, int& meshIndex) {
    ImGui::Text("%s:", label);
    ImGui::PushID(label);

    // Build list: "(blob)" + mesh filenames
    int total = 1 + (int)Window::meshNames.size();
    auto getter = [](void* data, int idx, const char** out) -> bool {
        auto* names = static_cast<std::vector<std::string>*>(data);
        if (idx == 0) { *out = "Show Gaussian"; return true; }
        if (idx - 1 < (int)names->size()) { *out = (*names)[idx - 1].c_str(); return true; }
        return false;
    };
    ImGui::Combo("##mesh", &meshIndex, getter, &Window::meshNames, total);

    if (meshIndex > 0)
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "  mesh selected");
    else
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "  Show Gaussian(default)");

    ImGui::PopID();
}

void Window::DrawMainGUI() {
    ImGui::Begin("Gaussian Cake");

    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.7f, 1.0f), "Gaussian Cake");
    ImGui::Separator();

    // Render mode: all three choices visible as radio buttons
    int modeInt = (int)renderMode;
    bool modeChanged = false;
    modeChanged |= ImGui::RadioButton("Triangle Morph##mode", &modeInt, (int)RenderMode::TriangleMorph);
    ImGui::SameLine();
    modeChanged |= ImGui::RadioButton("Particles##mode", &modeInt, (int)RenderMode::Particles);
    ImGui::SameLine();
    modeChanged |= ImGui::RadioButton("Splats##mode", &modeInt, (int)RenderMode::Splats);
    if (modeChanged && modeInt != (int)renderMode) {
        renderMode = (RenderMode)modeInt;
        // Switching modes disables any active morph; user re-clicks Bake.
        if (renderMode != RenderMode::TriangleMorph) meshMorphActive = false;
        if (renderMode != RenderMode::Splats) {
            g_splatBakeEpoch++;   // discard a bake still running
            if (splatMorphActive) {
                splatMorphActive = false;
                splatPreviewGen++;
            }
        }
    }
    if (renderMode == RenderMode::TriangleMorph || renderMode == RenderMode::Splats) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                           "  (requires both A and B to be meshes)");
    }

    ImGui::Separator();

    DrawGaussianSelector("Source (A)", meshIndexA);
    ImGui::Spacing();
    DrawGaussianSelector("Target (B)", meshIndexB);

    ImGui::Separator();
    ImGui::SliderInt("Particles",    &particleCount, 100, 10000);
    ImGui::SliderInt("Mesh Samples", &sampleCount,   500, 20000);
    if (renderMode == RenderMode::TriangleMorph) {
        ImGui::SliderInt("Subdivide Source", &subdivisionLevel, 0, 3);
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            subdivisionLevel == 0 ? "  no subdivision (original mesh)" :
            subdivisionLevel == 1 ? "  1 pass = 4x triangles" :
            subdivisionLevel == 2 ? "  2 passes = 16x triangles" :
                                    "  3 passes = 64x triangles (slow Bake)");
        ImGui::SliderFloat("Crossfade Start", &crossfadeStart, 0.0f, 1.0f);
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "  0 = swap immediately, 1 = no crossfade");
    }
    if (renderMode == RenderMode::Splats) {
        ImGui::SliderInt("Splats", &splatCount, 1000, 50000);
        ImGui::SliderFloat("Splat Size", &splatSizeMul, 0.3f, 3.0f);
        ImGui::SliderInt("Mixture K", &mixtureK, 1, 128);
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            mixtureK == 1 ? "  1 = original single-Gaussian map"
                          : "  Gaussians per shape (mixture OT)");
        ImGui::SliderFloat("Sinkhorn eps", &sinkhornEps, 0.005f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Blend Temp", &posteriorTemp, 0.25f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Cohesion", &splatCohesion, 0.0f, 1.0f);
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "  0 = straight lines, 1 = move as chunks");
        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
    }

    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(1.0f, 0.4f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.6f, 0.8f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.8f, 0.2f, 0.5f, 1.0f));
    ImGui::BeginDisabled(splatBaking);
    if (ImGui::Button(splatBaking ? "Baking..." : "Bake It!", ImVec2(-1, 0)))
        ComputeOTAndSpawnParticles();
    ImGui::EndDisabled();
    ImGui::PopStyleColor(3);

    if (splatBaking) {
        int stage = g_bakeStage.load();
        float frac = (float)g_bakeDone.load() / (float)g_bakeTotal.load();
        char label[64];
        snprintf(label, sizeof(label), "%s (%d/6)", kBakeStageNames[stage], std::min(stage + 1, 6));
        ImGui::ProgressBar(glm::clamp(frac, 0.0f, 1.0f), ImVec2(-1, 0), label);
    } else if (renderMode == RenderMode::Splats && lastBakeMs >= 0.0f) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  last bake: %.0f ms", lastBakeMs);
    }

    ImGui::Spacing();
    animator->DrawGUI();

    ImGui::Spacing();
    ImGui::Separator();

    if (ImGui::Button("Revert"))
        animator->ReverseTransport();
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        animator->Reset();
        particleSystem->useOTPositions = false;
        particleSystem->particles.clear();
        meshMorphActive     = false;
        particleMorphActive = false;
        g_splatBakeEpoch++;   // a bake still running is discarded
        if (splatMorphActive) {
            splatMorphActive = false;
            splatPreviewGen++;
        }
    }

    ImGui::Separator();
    ImGui::SliderFloat("Particle Radius", &particleSystem->particleRadius, 0.01f, 1.0f);

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
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if(!ImGui::GetIO().WantCaptureMouse){
            LeftDown = (action == GLFW_PRESS);
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if(!ImGui::GetIO().WantCaptureMouse){
            RightDown = (action == GLFW_PRESS);
        }
    }
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
