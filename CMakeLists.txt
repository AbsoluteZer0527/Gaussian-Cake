cmake_minimum_required(VERSION 3.16)

project(menv)

# Our code lives in src/<module>/, each header next to its .cpp:
#   app        window, GUI, camera, entry point, performance model
#   core       OpenGL setup, shaders, GPU timers, threading helpers
#   io         .obj / .glb / .gltf meshes (with materials and textures), .ply splat captures
#   mesh       topology, sampling, closest-point queries, template fitting, drawing
#   transport  Gaussians, Gaussian mixtures, optimal transport, the morph timeline
#   morph      signed-distance-field morph (CPU and GPU)
#   splats     Gaussian splats: generation and rendering
#   particles  the particle system
#   director   Morph Director: timing curves, styles, springs, picking, the shape timeline
# Third-party code: external/ (headers, imgui, Windows GLFW), third_party/glad, see below.
set(MODULES app core io mesh transport morph splats particles director)

set(
    SOURCES
    src/app/main.cpp
    src/app/Window.cpp
    src/app/Camera.cpp
    src/app/PerfModel.cpp
    src/core/Shader.cpp
    src/io/OBJLoader.cpp
    src/io/PlyLoader.cpp
    src/io/MeshData.cpp
    src/io/Texture.cpp
    src/io/GltfLoader.cpp
    src/io/PlyMeshLoader.cpp
    src/mesh/MeshTopology.cpp
    src/mesh/MeshSampler.cpp
    src/mesh/SurfaceQuery.cpp
    src/mesh/SurfaceColor.cpp
    src/mesh/TexturedMeshRenderer.cpp
    src/mesh/MeshFitter.cpp
    src/mesh/MeshRenderer.cpp
    src/transport/Gaussian.cpp
    src/transport/GaussianMixture.cpp
    src/transport/OptimalTransport.cpp
    src/transport/TransportAnimator.cpp
    src/morph/SDFMorph.cpp
    src/morph/GpuSDFMorph.cpp
    src/morph/PiecewiseMorph.cpp
    src/splats/SplatGenerator.cpp
    src/splats/SplatRenderer.cpp
    src/splats/SplatHeat.cpp
    src/mesh/MorphTexturedRenderer.cpp
    src/particles/Particle.cpp
    src/particles/ParticleSystem.cpp
    src/director/MorphCurve.cpp
    src/director/MorphStyle.cpp
    src/director/MorphSpring.cpp
    src/director/MorphTimeline.cpp
    src/director/MorphDirector.cpp
    src/director/Picking.cpp
    src/director/DirectorWidgets.cpp
)

# ---- Third-party code ----
# external/: header-only libraries and imgui. third_party/glad: the OpenGL loader.
# GLFW: the prebuilt external/lib/glfw3.lib with MSVC; elsewhere an installed
# one (GC_SYSTEM_GLFW=ON) or 3.3.8 (the version of external/GLFW) built from source.
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

add_library(glad STATIC third_party/glad/src/gl.c)
target_include_directories(glad PUBLIC third_party/glad/include)

option(GC_SYSTEM_GLFW "Use an installed GLFW (find_package) instead of downloading it" OFF)
if(MSVC AND EXISTS "${PROJECT_SOURCE_DIR}/external/lib/glfw3.lib")
    add_library(glfw STATIC IMPORTED)
    set_target_properties(glfw PROPERTIES
        IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/external/lib/glfw3.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/external")
elseif(GC_SYSTEM_GLFW)
    find_package(glfw3 3.3 REQUIRED)
else()
    include(FetchContent)
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.8
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(glfw)
endif()

set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/external/imgui)
set(
    IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/imgui_impl_opengl3.cpp
)

# Headers too, so Visual Studio lists them, grouped by folder as on disk
file(GLOB HEADERS src/*/*.h)
source_group(TREE ${PROJECT_SOURCE_DIR}/src FILES ${SOURCES} ${HEADERS})

# Settings shared by the app and the tests
add_library(gc_settings INTERFACE)
target_compile_features(gc_settings INTERFACE cxx_std_17)
# Every module folder is on the include path, so "#include \"Window.h\"" works anywhere
target_include_directories(gc_settings INTERFACE external external/Eigen ${IMGUI_DIR})
foreach(module ${MODULES})
    target_include_directories(gc_settings INTERFACE src/${module})
endforeach()
target_compile_definitions(gc_settings INTERFACE GLM_ENABLE_EXPERIMENTAL _USE_MATH_DEFINES NOMINMAX)
if(MSVC)
    target_compile_options(gc_settings INTERFACE /bigobj /utf-8)
endif()
target_link_libraries(gc_settings INTERFACE glad glfw OpenGL::GL Threads::Threads ${CMAKE_DL_LIBS})

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${IMGUI_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE gc_settings)

# Move assets to .exe
add_custom_target(CopyShaders ALL
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${PROJECT_SOURCE_DIR}/shaders"
	"${CMAKE_BINARY_DIR}/shaders"
)
add_dependencies(menv CopyShaders)

if(EXISTS "${CMAKE_SOURCE_DIR}/meshes")
    add_custom_target(CopyMeshes ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/meshes"
        "${CMAKE_BINARY_DIR}/meshes"
    )
    add_dependencies(${PROJECT_NAME} CopyMeshes)
endif()

# ---- Tests (no window or GPU needed): ctest --test-dir build ----
enable_testing()
# everything but the app itself (window, GUI)
set(TEST_SOURCES ${SOURCES})
list(FILTER TEST_SOURCES EXCLUDE REGEX "src/app/")
add_executable(gc_tests tests/tests.cpp ${TEST_SOURCES} ${IMGUI_SOURCES})
target_link_libraries(gc_tests PRIVATE gc_settings)
add_test(NAME gc_tests COMMAND gc_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
