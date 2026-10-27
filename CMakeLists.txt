cmake_minimum_required(VERSION 3.16)

project(menv)

# Our code lives in src/<module>/, each header next to its .cpp:
#   app        window, GUI, camera, entry point, performance model
#   core       OpenGL setup, shaders, GPU timers, threading helpers
#   io         .obj meshes and .ply splat captures
#   mesh       topology, sampling, closest-point queries, template fitting, drawing
#   transport  Gaussians, Gaussian mixtures, optimal transport, the morph timeline
#   morph      signed-distance-field morph (CPU and GPU)
#   splats     Gaussian splats: generation and rendering
#   particles  the particle system
# Third-party code (imgui, Eigen, glm, GLEW, GLFW and their .lib files) is in external/.
set(MODULES app core io mesh transport morph splats particles)

set(
    SOURCES
    src/app/main.cpp
    src/app/Window.cpp
    src/app/Camera.cpp
    src/app/PerfModel.cpp
    src/core/Shader.cpp
    src/io/OBJLoader.cpp
    src/io/PlyLoader.cpp
    src/mesh/MeshTopology.cpp
    src/mesh/MeshSampler.cpp
    src/mesh/SurfaceQuery.cpp
    src/mesh/MeshFitter.cpp
    src/mesh/MeshRenderer.cpp
    src/transport/Gaussian.cpp
    src/transport/GaussianMixture.cpp
    src/transport/OptimalTransport.cpp
    src/transport/TransportAnimator.cpp
    src/morph/SDFMorph.cpp
    src/morph/GpuSDFMorph.cpp
    src/splats/SplatGenerator.cpp
    src/splats/SplatRenderer.cpp
    src/particles/Particle.cpp
    src/particles/ParticleSystem.cpp
)

set(
    IMGUI_SOURCES
    external/imgui/imgui.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_impl_glfw.cpp
    external/imgui/imgui_impl_opengl3.cpp
)

# Require GL
find_package(OpenGL REQUIRED)

# Every module folder is on the include path, so "#include \"Window.h\"" works anywhere
include_directories(external external/Eigen external/imgui)
foreach(module ${MODULES})
    include_directories(src/${module})
endforeach()

# Add library directories
link_directories(
    external/lib
)

# Headers too, so Visual Studio lists them, grouped by folder as on disk
file(GLOB HEADERS src/*/*.h)
source_group(TREE ${PROJECT_SOURCE_DIR}/src FILES ${SOURCES} ${HEADERS})

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${IMGUI_SOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} glew32s.lib glfw3)

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
