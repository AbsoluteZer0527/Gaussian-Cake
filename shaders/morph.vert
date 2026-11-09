#version 330 core

// Morph targets (blend shapes): both end shapes live on the GPU and every
// vertex blends them itself, so the CPU only sends morphT each frame.

layout (location = 0) in vec3 positionStart;
layout (location = 1) in vec3 normalStart;
layout (location = 2) in vec3 positionEnd;
layout (location = 3) in vec3 normalEnd;
layout (location = 4) in vec3 colorStart;   // per-vertex colours (used if useVertexColor is on)
layout (location = 5) in vec3 colorEnd;
layout (location = 6) in vec2 uvA;         // texture coords on A and on B
layout (location = 7) in vec2 uvB;         // (MorphTexturedRenderer only; 0 otherwise)
layout (location = 8) in float vertexT;    // Morph Director: this vertex's own time

uniform mat4 viewProj;
uniform mat4 model;
uniform float morphT;
uniform int   usePerVertexT = 0;   // 1: each vertex runs its own clock (vertexT)
uniform float sag = 0.0;           // mid-flight droop (Melt), 0 at both ends

out vec3 fragNormal;
out vec3 fragAlbedo;
out vec2 fragUV;             // A's uv
out vec2 fragUVB;            // B's uv
out vec3 fragAlbedoA;        // the two end colours, unmixed (for useMorphTexture)
out vec3 fragAlbedoB;
out float fragT;             // the time this vertex is at (blends the textures)

void main()
{
    float t = (usePerVertexT == 1) ? vertexT : morphT;
    vec3 position = mix(positionStart, positionEnd, t);
    float tc = clamp(t, 0.0, 1.0);
    position.y -= sag * 4.0 * tc * (1.0 - tc);

    // Lerp of the two end normals: not the exact normal of the in-between
    // surface, but close, and it needs no neighbouring vertices. The fragment
    // shader renormalizes; the fallback covers opposite normals cancelling.
    vec3 normal = mix(normalStart, normalEnd, t);
    if (dot(normal, normal) < 1e-8) normal = normalEnd;

    gl_Position = viewProj * model * vec4(position, 1.0);
    fragNormal = vec3(model * vec4(normal, 0));
    fragAlbedo = mix(colorStart, colorEnd, tc);
    fragT = t;
    fragAlbedoA = colorStart;
    fragAlbedoB = colorEnd;
    fragUV = uvA;
    fragUVB = uvB;
}
