#version 330 core
// Particles as round screen-space points (replaces the old fixed-function
// glBegin(GL_POINTS) path, which a core profile, e.g. on macOS, doesn't have).
layout (location = 0) in vec3 position;

uniform mat4  viewProj;
uniform float pointSize;   // pixels

void main() {
    gl_Position  = viewProj * vec4(position, 1.0);
    gl_PointSize = pointSize;
}
