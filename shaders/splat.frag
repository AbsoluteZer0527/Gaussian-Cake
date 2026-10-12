#version 330 core

in vec2 pixelOffset;
in vec3 conic;
in vec4 splatColor;

out vec4 fragColor;

void main()
{
    vec2 d = pixelOffset;
    float power = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);
    if (power > 0.0) discard;

    float alpha = min(0.99, splatColor.a * exp(power));
    if (alpha < 1.0 / 255.0) discard;

    fragColor = vec4(splatColor.rgb, alpha);
}
