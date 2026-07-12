#include "Window.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "OBJLoader.h"
#include "MeshSampler.h"

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
SnapMode                  Window::snapMode        = SnapMode::NearestTriangle;  // Phase 3 default

//Dual-mesh (Option B) target-side state
std::vector<glm::vec3>    Window::meshTargetStart;
std::vector<glm::vec3>    Window::meshTargetEnd;
std::vector<glm::vec3>    Window::meshTargetCurrent;
std::vector<unsigned int> Window::meshTargetIndices;

//Gaussian OT pipeline
Gaussian*        Window::gaussianA     = nullptr;
Gaussian*        Window::gaussianB     = nullptr;
OTMap*           Window::otMap         = nullptr;
TransportAnimator* Window::animator    = nullptr;
bool             Window::otMapComputed = false;
int              Window::particleCount = 5000;
std::vector<std::string> Window::meshNames;
std::vector<std::string> Window::meshPaths;
int              Window::meshIndexA    = 0;
int              Window::meshIndexB    = 0;
int              Window::sampleCount        = 5000;
int              Window::subdivisionLevel   = 0;

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

    // Distinct diffuse colors so overlapping source/target are visually separable.
    meshRenderer->diffuseColor       = glm::vec3(0.85f, 0.55f, 0.75f);  // pink (source)
    meshRendererTarget->diffuseColor = glm::vec3(0.40f, 0.75f, 0.85f);  // teal (target)

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
    particleSystem->Update(deltaTime);

    if (meshMorphActive) {
        // Source mesh: animator advances the shared timeline and writes
        // lerp'd source-vertex positions into meshStartPositions.
        animator->Update(deltaTime, meshStartPositions);
        meshRenderer->UpdatePositions(meshStartPositions);

        // Target mesh: manually lerp using the same easing curve so it stays
        // perfectly in sync with the source.
        if (!meshTargetCurrent.empty()) {
            float e = animator->ApplyEasing(animator->GetProgress());
            for (size_t i = 0; i < meshTargetCurrent.size(); ++i)
                meshTargetCurrent[i] = glm::mix(meshTargetStart[i], meshTargetEnd[i], e);
            meshRendererTarget->UpdatePositions(meshTargetCurrent);
        }
    } else if (otMapComputed) {
        animator->Update(deltaTime, particleSystem->transportPositions);
        particleSystem->useOTPositions = (animator->state != AnimationState::GAUSSIAN_A);
    }
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

    if (renderMode == RenderMode::TriangleMorph && meshMorphActive) {
        // Crossfade: source fades out (1→0) as target fades in (0→1).
        float e = animator->ApplyEasing(animator->GetProgress());
        meshRenderer->alpha       = 1.0f - e;
        meshRendererTarget->alpha = e;

        // Enable alpha blending. Draw the more-opaque mesh first so it
        // gets depth priority; draw the less-opaque one with depth-write
        // disabled so it blends correctly on top without z-fighting.
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

// Phase 3 snap: for each endpoint, find the closest point on *any* target
// triangle. Continuous surface projection — sub-triangle precision, no
// collapse to shared samples. Brute force O(V*T) — one-time Bake cost.
// For large target meshes this can take a few seconds; a BVH would fix that.
static void SnapToNearestOnTriangles(std::vector<glm::vec3>& endVerts,
                                     const std::vector<Triangle>& tris) {
    if (tris.empty()) return;
    for (auto& p : endVerts) {
        float bestDist = 1e30f;
        glm::vec3 best = p;
        for (const auto& t : tris) {
            glm::vec3 q = ClosestPointOnTriangle(p, t.v0, t.v1, t.v2);
            glm::vec3 d = p - q;
            float dist = glm::dot(d, d);
            if (dist < bestDist) { bestDist = dist; best = q; }
        }
        p = best;
    }
}

// Phase 2 snap: pull each OT-mapped endpoint to the nearest sampled point on
// the target surface. Runs in-place. Brute force O(V*K) — one-time Bake cost.
// Unlike greedy discrete OT, the same target sample can attract multiple source
// vertices (may cause local triangle collapse near dense clusters).
static void SnapToNearestOnTarget(std::vector<glm::vec3>& endVerts,
                                  const std::vector<glm::vec3>& targetSamples) {
    if (targetSamples.empty()) return;
    for (auto& p : endVerts) {
        float bestDist = 1e30f;
        glm::vec3 bestPoint = p;
        for (const auto& s : targetSamples) {
            glm::vec3 d = p - s;
            float dist = glm::dot(d, d);   // squared distance — no sqrt needed
            if (dist < bestDist) { bestDist = dist; bestPoint = s; }
        }
        p = bestPoint;
    }
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

void Window::ComputeOTAndSpawnParticles() {
    // meshIndex 0 = blob; 1..N = meshPaths[index-1]
    bool hasMeshA = (meshIndexA > 0 && meshIndexA <= (int)meshPaths.size());
    bool hasMeshB = (meshIndexB > 0 && meshIndexB <= (int)meshPaths.size());
    const std::string& pathA = hasMeshA ? meshPaths[meshIndexA - 1] : "";
    const std::string& pathB = hasMeshB ? meshPaths[meshIndexB - 1] : "";

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

        // Refine SOURCE endpoints toward target surface.
        //   None            (Phase 1): no snap — endpoint = OT-warped source
        //   NearestSample   (Phase 2): snap to closest of the K sampled points
        //   NearestTriangle (Phase 3): snap to closest point on any target triangle
        switch (snapMode) {
            case SnapMode::NearestSample:
                SnapToNearestOnTarget(endVerts, sampledB);
                break;
            case SnapMode::NearestTriangle:
                SnapToNearestOnTriangles(endVerts, trisB);
                break;
            case SnapMode::None:
            default:
                break;
        }

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

        // Symmetric snap: pull target-start positions onto the source surface,
        // so at t=0 the target mesh's vertices sit on/near the source.
        switch (snapMode) {
            case SnapMode::NearestSample:
                SnapToNearestOnTarget(tStart, sampledA);
                break;
            case SnapMode::NearestTriangle:
                SnapToNearestOnTriangles(tStart, trisA);
                break;
            case SnapMode::None:
            default:
                break;
        }

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
        meshMorphActive = true;
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

    // Render mode toggle: triangle morph (default) vs legacy particles
    const char* modeItems[] = { "Triangle Morph", "Particles" };
    int modeInt = (int)renderMode;
    if (ImGui::Combo("Mode", &modeInt, modeItems, 2)) {
        renderMode = (RenderMode)modeInt;
        // Switching modes disables any active triangle morph; user re-clicks Bake.
        if (renderMode != RenderMode::TriangleMorph) meshMorphActive = false;
    }
    if (renderMode == RenderMode::TriangleMorph) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                           "  (triangle morph requires both A and B to be meshes)");

        const char* snapItems[] = {
            "None (Phase 1: OT warp only)",
            "Nearest Sample (Phase 2)",
            "Nearest Triangle (Phase 3)"
        };
        int snapInt = (int)snapMode;
        if (ImGui::Combo("Endpoint Snap", &snapInt, snapItems, 3))
            snapMode = (SnapMode)snapInt;

        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            snapMode == SnapMode::NearestTriangle ? "  continuous surface projection"
          : snapMode == SnapMode::NearestSample   ? "  snap to K sampled points"
          :                                          "  raw linear OT warp");
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
    }

    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(1.0f, 0.4f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.6f, 0.8f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.8f, 0.2f, 0.5f, 1.0f));
    if (ImGui::Button("Bake It!", ImVec2(-1, 0)))
        ComputeOTAndSpawnParticles();
    ImGui::PopStyleColor(3);

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
        meshMorphActive = false;
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
