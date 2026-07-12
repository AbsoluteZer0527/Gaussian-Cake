cmake_minimum_required(VERSION 3.16)

project(menv)

# Add source files
set(
    SOURCES
    main.cpp
    src/Camera.cpp
    src/Cube.cpp
    src/Shader.cpp
    src/Window.cpp
    src/imgui_widgets.cpp
    src/imgui.cpp
    src/imgui_demo.cpp
    src/imgui_draw.cpp
    src/imgui_tables.cpp
    src/imgui_impl_glfw.cpp
    src/imgui_impl_opengl3.cpp
    src/Particle.cpp
    src/ParticleSystem.cpp
    src/Gaussian.cpp
    src/OBJLoader.cpp
    src/MeshSampler.cpp
    src/OptimalTransport.cpp
    src/TransportAnimator.cpp
    src/MeshRenderer.cpp
)

# Add header files
set(
    HEADERS
    include/core.h
    include/Camera.h
    include/Cube.h
    include/Shader.h
    include/Window.h
)

# Require GL
find_package(OpenGL REQUIRED)

# Add include directories
include_directories(
    include
    include/Eigen
)

# Add library directories
link_directories(
    lib
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
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

add_dependencies(menv CopyMeshes)