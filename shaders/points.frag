#version 330 core
uniform vec3 color;
out vec4 fragColor;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    if (dot(d, d) > 1.0) discard;   // round dots instead of squares
    fragColor = vec4(color, 1.0);
}
