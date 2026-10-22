#version 430

// Splats prepared by splat_prep.comp and sorted on the GPU: this shader only
// looks up its splat (in back-to-front order) and places the quad corner.

layout (location = 0) in vec2 corner;     // quad corner in [-1, 1]^2

struct Prepared {
    vec4 clip;
    vec4 conicRadius;
    vec4 color;
};
layout(std430, binding = 0) readonly buffer Prep  { Prepared prep[]; };
layout(std430, binding = 1) readonly buffer Order { uint order[]; };

uniform vec2 viewport;                    // pixels

out vec2 pixelOffset;
out vec3 conic;
out vec4 splatColor;

void main()
{
    Prepared p = prep[order[gl_InstanceID]];
    pixelOffset = corner * p.conicRadius.w;
    conic       = p.conicRadius.xyz;
    splatColor  = p.color;
    if (p.conicRadius.w <= 0.0) {         // culled: a degenerate quad outside the clip volume
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    vec4 clip = p.clip;
    clip.xy += (pixelOffset * 2.0 / viewport) * clip.w;
    gl_Position = clip;
}
