#include "Window.h"
#include "core.h"
#include <cstdio>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef _WIN32
// On laptops with two GPUs, ask the NVIDIA / AMD driver to run this program on
// the discrete GPU instead of the integrated one (the same exports games use).
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

extern Camera* Cam;   // Window.cpp

void error_callback(int error, const char* description) {
    // Print error.
    std::cerr << description << std::endl;
}

void setup_callbacks(GLFWwindow* window) {
    // Set the error callback.
    glfwSetErrorCallback(error_callback);
    // Set the window resize callback.
    glfwSetWindowSizeCallback(window, Window::resizeCallback);

    // Set the key callback.
    glfwSetKeyCallback(window, Window::keyCallback);

    // Set the mouse and cursor callbacks
    glfwSetMouseButtonCallback(window, Window::mouse_callback);
    glfwSetCursorPosCallback(window, Window::cursor_callback);
}

void setup_opengl_settings() {
    // Enable depth buffering.
    glEnable(GL_DEPTH_TEST);
    // Related to shaders and z value comparisons for the depth buffer.
    glDepthFunc(GL_LEQUAL);
    // Set polygon drawing mode to fill front and back of each polygon.
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    // Set clear color to white.
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void print_versions() {
    // Get info of GPU and supported OpenGL version.
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "OpenGL version supported: " << glGetString(GL_VERSION)
              << std::endl;

    // If the shading language symbol is defined.
#ifdef GL_SHADING_LANGUAGE_VERSION
    std::cout << "Supported GLSL version is: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
#endif
}

// ---- Scripted runs (for testing without clicking) ----
//   menv --shot out/prefix --mode topology|particles|splats --a teapot.obj --b bunny.obj
//        [--method auto|shared|sdf|piecewise] [--transport greedy|gaussian|mixture]
//        [--k 8] [--times 0,0.5,1] [--size 1280x800] [--nogui 1]
//        [--sh 0..3] [--azimuth deg] [--distance d] [--up asis|flipy] [--gpu 0] [--temp x]
//        Morph Director: [--keys a.obj,b.obj,c.obj] [--style classic|bake|ripple|melt|pop|dissolve]
//        [--origin x,y,z] [--pin ax,ay,az:bx,by,bz] (repeatable) [--jiggle 0..1] [--wave 0..0.8]
//        [--sag x] [--tool origin|paint|pin] (show the tool overlay)
//        [--click x,y] (repeatable: a click of the tool at that pixel, after the bake, at time 0)
// Bakes A -> B, then saves <prefix>_t<time>.ppm for every time and exits.
struct ScriptArgs {
    std::string shot, mode, a, b, method, transport;
    std::vector<std::string> keys;          // Morph Director timeline (overrides a / b)
    std::string style, tool;
    std::vector<glm::vec3> origin;          // 0 or 1 entries
    std::vector<MorphPin> pins;
    float jiggle = -1.0f, wave = -1.0f, sag = -1.0f;
    std::vector<glm::vec2> clicks;          // tool clicks (window pixels) before the shots
    std::vector<float> times = { 0.0f, 0.5f, 1.0f };
    int width = 800, height = 600, k = -1;
    int gpu = -1;               // 0 = force the CPU paths
    float azimuth = -1000.0f;   // camera azimuth in degrees (unset: the default view)
    float distance = -1.0f;     // camera distance (unset: the default view)
};

static bool ParseArgs(int argc, char** argv, ScriptArgs& args) {
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key = argv[i], val = argv[i + 1];
        if      (key == "--shot")      args.shot = val;
        else if (key == "--mode")      args.mode = val;
        else if (key == "--a")         args.a = val;
        else if (key == "--b")         args.b = val;
        else if (key == "--method")    args.method = val;
        else if (key == "--transport") args.transport = val;
        else if (key == "--k")         args.k = std::stoi(val);
        else if (key == "--nogui")     Window::hideGui = (val == "1");
        else if (key == "--sh")        SplatRenderer::shDegree = std::stoi(val);
        else if (key == "--gpu")       args.gpu = std::stoi(val);
        else if (key == "--temp")      Window::posteriorTemp = std::stof(val);
        else if (key == "--distance")  args.distance = std::stof(val);
        else if (key == "--azimuth")   args.azimuth = std::stof(val);
        else if (key == "--up")        Window::plyOptions.up = val == "asis" ? PlyLoadOptions::Up::AsIs
                                                                             : PlyLoadOptions::Up::FlipY;
        else if (key == "--size")      sscanf(val.c_str(), "%dx%d", &args.width, &args.height);
        else if (key == "--style")     args.style = val;
        else if (key == "--tool")      args.tool = val;
        else if (key == "--jiggle")    args.jiggle = std::stof(val);
        else if (key == "--wave")      args.wave = std::stof(val);
        else if (key == "--sag")       args.sag = std::stof(val);
        else if (key == "--click") {
            glm::vec2 c;
            if (sscanf(val.c_str(), "%f,%f", &c.x, &c.y) != 2) { std::cerr << "--click x,y\n"; return false; }
            args.clicks.push_back(c);
        } else if (key == "--origin") {
            glm::vec3 o;
            if (sscanf(val.c_str(), "%f,%f,%f", &o.x, &o.y, &o.z) != 3) { std::cerr << "--origin x,y,z\n"; return false; }
            args.origin = { o };
        } else if (key == "--pin") {
            MorphPin p;
            if (sscanf(val.c_str(), "%f,%f,%f:%f,%f,%f", &p.onA.x, &p.onA.y, &p.onA.z,
                       &p.onB.x, &p.onB.y, &p.onB.z) != 6) { std::cerr << "--pin ax,ay,az:bx,by,bz\n"; return false; }
            args.pins.push_back(p);
        } else if (key == "--keys") {
            size_t pos = 0;
            while (pos < val.size()) {
                size_t comma = val.find(',', pos);
                if (comma == std::string::npos) comma = val.size();
                args.keys.push_back(val.substr(pos, comma - pos));
                pos = comma + 1;
            }
        }
        else if (key == "--times") {
            args.times.clear();
            size_t pos = 0;
            while (pos < val.size()) {
                size_t comma = val.find(',', pos);
                if (comma == std::string::npos) comma = val.size();
                args.times.push_back(std::stof(val.substr(pos, comma - pos)));
                pos = comma + 1;
            }
        } else {
            std::cerr << "Unknown option " << key << "\n";
            return false;
        }
    }
    return true;
}

static int MeshIndexByName(const std::string& name) {
    if (name.empty() || name == "blob") return 0;
    for (size_t i = 0; i < Window::meshNames.size(); ++i)
        if (Window::meshNames[i] == name) return (int)i + 1;   // index 0 is the blob
    std::cerr << "[Script] no mesh named " << name << "\n";
    return -1;
}

static void Frame(GLFWwindow* window) {
    Window::idleCallback();
    Window::displayCallback(window);
}

static int RunScript(GLFWwindow* window, const ScriptArgs& args) {
    if (args.mode == "topology")       Window::renderMode = RenderMode::TriangleMorph;
    else if (args.mode == "particles") Window::renderMode = RenderMode::Particles;
    else if (args.mode == "splats")    Window::renderMode = RenderMode::Splats;
    if (args.method == "auto")      Window::morphMethod = MorphMethod::Auto;
    if (args.method == "shared")    Window::morphMethod = MorphMethod::SharedTopology;
    if (args.method == "sdf")       Window::morphMethod = MorphMethod::SDF;
    if (args.method == "piecewise") Window::morphMethod = MorphMethod::Piecewise;
    if (args.transport == "greedy")   Window::particleTransport = ParticleTransport::Greedy;
    if (args.transport == "gaussian") Window::particleTransport = ParticleTransport::Gaussian;
    if (args.transport == "mixture")  Window::particleTransport = ParticleTransport::Mixture;
    if (args.k > 0) Window::mixtureK = Window::particleMixtureK = Window::morphMixtureK = args.k;

    int a = MeshIndexByName(args.a), b = MeshIndexByName(args.b);
    if (a < 0 || b < 0) return EXIT_FAILURE;
    Window::meshIndexA = a;
    Window::meshIndexB = b;

    // Morph Director: the shape timeline and the first step's style
    MorphTimeline& tl = Window::timeline;
    tl.keys = { a, b };
    if (!args.keys.empty()) {
        tl.keys.clear();
        for (const std::string& k : args.keys) {
            int idx = MeshIndexByName(k);
            if (idx < 0) return EXIT_FAILURE;
            tl.keys.push_back(idx);
        }
        Window::meshIndexA = tl.keys[0];
        Window::meshIndexB = tl.keys[1];
    }
    tl.SyncSegments();
    for (size_t k = 0; k < tl.segs.size(); ++k) {
        if (!args.style.empty()) {
            int st = FindStyle(args.style);
            if (st < 0) { std::cerr << "[Script] no style named " << args.style << "\n"; return EXIT_FAILURE; }
            tl.ApplyPreset((int)k, st);
        }
        MorphStyle& style = tl.segs[k].style;
        if (args.jiggle >= 0.0f) style.jiggle = args.jiggle;
        if (args.wave >= 0.0f)   style.wave = args.wave;
        if (args.sag >= 0.0f)    style.sag = args.sag;
    }
    tl.segs[0].pins = args.pins;
    if (!args.origin.empty()) {
        Window::director.origin = args.origin[0];
        Window::director.hasOrigin = true;
        for (auto& seg : tl.segs) seg.style.origin = OriginMode::Clicked;
    }
    if (args.tool == "origin") Window::director.tool = MorphDirector::Tool::Origin;
    if (args.tool == "paint")  Window::director.tool = MorphDirector::Tool::PaintEarly;
    if (args.tool == "pin")    Window::director.tool = MorphDirector::Tool::Pin;

    if (args.azimuth > -999.0f) Cam->SetAzimuth(args.azimuth);
    if (args.gpu == 0) Window::useGpu = false;
    if (args.distance > 0.0f) Cam->SetDistance(args.distance);
    for (int i = 0; i < 5; ++i) Frame(window);
    Window::ComputeOTAndSpawnParticles();

    double start = glfwGetTime();
    for (;;) {
        Frame(window);
        bool busy = Window::morphBaking || Window::splatBaking;
        bool ready = Window::trueMorphActive || Window::splatMorphActive || Window::particleMorphActive;
        if (!busy && ready) break;
        if (!busy && glfwGetTime() - start > 2.0) {
            std::cerr << "[Script] bake produced nothing\n";
            return EXIT_FAILURE;
        }
        if (glfwGetTime() - start > 900.0) {
            std::cerr << "[Script] bake timed out\n";
            return EXIT_FAILURE;
        }
    }
    std::cout << "[Script] bake finished in " << glfwGetTime() - start << " s\n";

    if (!args.clicks.empty()) {
        Window::animator->Seek(0.0f);
        for (int i = 0; i < 3; ++i) Frame(window);
        for (const glm::vec2& c : args.clicks) {
            Window::ToolClick(c.x, c.y);
            Frame(window);
        }
        std::cout << "[Script] origin " << (Window::director.hasOrigin ? "set" : "not set") << " at "
                  << Window::director.origin.x << ", " << Window::director.origin.y << ", "
                  << Window::director.origin.z << "; " << Window::timeline.segs[0].pins.size() << " pins\n";
    }

    for (float t : args.times) {
        Window::animator->Seek(t);
        for (int i = 0; i < 4; ++i) Frame(window);
        char name[64];
        snprintf(name, sizeof(name), "_t%.2f.ppm", t);
        Window::screenshotPath = args.shot + name;
        Frame(window);
    }
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    ScriptArgs script;
    if (!ParseArgs(argc, argv, script)) exit(EXIT_FAILURE);

    // Create the GLFW window.
    GLFWwindow* window = Window::createWindow(script.width, script.height);
    if (!window) exit(EXIT_FAILURE);

    // Print OpenGL and GLSL versions.
    print_versions();
    // Setup callbacks.
    setup_callbacks(window);
    // Setup OpenGL settings.
    setup_opengl_settings();

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup scaling
    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor());
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    style.FontScaleDpi = main_scale;        // Set initial font scale. (using io.ConfigDpiScaleFonts=true makes this unnecessary. We leave both here for documentation purpose)

    /*float fontScale = 3.0f;
    io.Fonts->AddFontDefault();
    io.FontGlobalScale = fontScale;*/

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);

    // GL 3.0 + GLSL 130
    // A core profile (macOS) needs GLSL 150 for ImGui's shaders
    GLint profile = 0;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile);
    const char* glsl_version = (profile & GL_CONTEXT_CORE_PROFILE_BIT) ? "#version 150" : "#version 130";
    ImGui_ImplOpenGL3_Init(glsl_version);


    // Initialize the shader program; exit if initialization fails.
    if (!Window::initializeProgram()) exit(EXIT_FAILURE);
    // Initialize objects/pointers for rendering; exit if initialization fails.
    if (!Window::initializeObjects()) exit(EXIT_FAILURE);

    if (!script.shot.empty()) {
        int code = RunScript(window, script);
        Window::cleanUp();
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(code);
    }

    // Loop while GLFW window should stay open.
    while (!glfwWindowShouldClose(window)) {
        // Main render display callback. Rendering of objects is done here.
        Window::displayCallback(window);

        // Idle callback. Updating objects, etc. can be done here.
        Window::idleCallback();
    }

    Window::cleanUp();
    // Destroy the window.
    glfwDestroyWindow(window);
    // Terminate GLFW.
    glfwTerminate();

    exit(EXIT_SUCCESS);
}