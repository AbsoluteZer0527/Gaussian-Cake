#pragma once

#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "core.h"

GLuint LoadShaders(const char* vertex_file_path, const char* fragment_file_path);

// Compute program from one .comp file. commonFilePath, if given, is inserted
// after its #version line, so shaders can share helper functions.
GLuint LoadComputeShader(const char* computeFilePath, const char* commonFilePath = nullptr);

// Compute shaders, storage buffers and indirect draws need OpenGL 4.3.
bool GpuComputeAvailable();
