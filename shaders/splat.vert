#version 330 core

// EWA splatting: project a 3D Gaussian to a 2D screen-space Gaussian.
// Sigma2D = J * W * Sigma * W^T * J^T, where W is the view rotation and
// J is the Jacobian of the perspective projection at the splat center.

layout (location = 0) in vec2 corner;     // quad corner in [-1, 1]^2
layout (location = 1) in vec3 center;     // per-instance
layout (location = 2) in vec3 covA;       // xx, xy, xz
layout (location = 3) in vec3 covB;       // yy, yz, zz
layout (location = 4) in vec4 rgba;

uniform mat4 view;
uniform mat4 proj;
uniform vec2 viewport;                    // pixels

out vec2 pixelOffset;                     // offset from splat center, pixels
out vec3 conic;                           // inverse 2D covariance (xx, xy, yy)
out vec4 splatColor;

void main()
{
    vec4 t = view * vec4(center, 1.0);
    float tz = -t.z;                      // positive depth in front of camera

    // Cull splats behind / too close to the camera
    if (tz < 0.05) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    // Focal lengths in pixels
    float fx = proj[0][0] * viewport.x * 0.5;
    float fy = proj[1][1] * viewport.y * 0.5;

    // Clamp to slightly outside the frustum for numerical stability (as in 3DGS)
    float limX = 1.3 / proj[0][0];
    float limY = 1.3 / proj[1][1];
    float tx = clamp(t.x / tz, -limX, limX) * tz;
    float ty = clamp(t.y / tz, -limY, limY) * tz;

    // Jacobian w.r.t. view-space (x, y, z); third row unused
    mat3 J = mat3(
        fx / tz, 0.0, 0.0,                         // column 0
        0.0, fy / tz, 0.0,                         // column 1
        fx * tx / (tz * tz), fy * ty / (tz * tz), 0.0); // column 2

    mat3 W = mat3(view);
    mat3 Sigma = mat3(
        covA.x, covA.y, covA.z,
        covA.y, covB.x, covB.y,
        covA.z, covB.y, covB.z);

    mat3 T = J * W;
    mat3 cov2d = T * Sigma * transpose(T);

    // Low-pass filter: every splat covers at least ~1 pixel
    float a = cov2d[0][0] + 0.3;
    float b = cov2d[0][1];
    float c = cov2d[1][1] + 0.3;

    float det = a * c - b * b;
    if (det <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    conic = vec3(c, -b, a) / det;

    // Quad half-size: 3 sigma along the major axis
    float mid = 0.5 * (a + c);
    float lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    float radius = 3.0 * sqrt(lambda1);

    pixelOffset = corner * radius;
    splatColor = rgba;

    vec4 clip = proj * t;
    clip.xy += (pixelOffset * 2.0 / viewport) * clip.w;
    gl_Position = clip;
}
