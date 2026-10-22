#version 330 core

// Morph targets (blend shapes): both end shapes live on the GPU and every
// vertex blends them itself, so the CPU only sends morphT each frame.

layout (location = 0) in vec3 positionStart;
layout (location = 1) in vec3 normalStart;
layout (location = 2) in vec3 positionEnd;
layout (location = 3) in vec3 normalEnd;

uniform mat4 viewProj;
uniform mat4 model;
uniform float morphT;

out vec3 fragNormal;

void main()
{
    vec3 position = mix(positionStart, positionEnd, morphT);

    // Lerp of the two end normals: not the exact normal of the in-between
    // surface, but close, and it needs no neighbouring vertices. The fragment
    // shader renormalizes; the fallback covers opposite normals cancelling.
    vec3 normal = mix(normalStart, normalEnd, morphT);
    if (dot(normal, normal) < 1e-8) normal = normalEnd;

    gl_Position = viewProj * model * vec4(position, 1.0);
    fragNormal = vec3(model * vec4(normal, 0));
}
