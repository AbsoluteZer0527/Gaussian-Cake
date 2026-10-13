#version 330 core

// Inputs to the fragment shader are the outputs of the same name from the vertex shader.
// Note that you do not have access to the vertex shader's default output, gl_Position.

in vec3 fragNormal;

// uniforms used for lighting
uniform vec3 AmbientColor = vec3(0.2);
uniform vec3 LightDirection = normalize(vec3(1, 5, 2));
uniform vec3 LightColor = vec3(1);
uniform vec3  DiffuseColor;
uniform float alpha = 1.0;

out vec4 fragColor;

void main()
{
	// Two-sided lighting: back faces use the flipped normal, so open sheets
	// and meshes with inverted winding still shade on the side being viewed.
	vec3 n = normalize(fragNormal);
	if (!gl_FrontFacing) n = -n;

	vec3 irradiance  = AmbientColor + LightColor * max(0, dot(LightDirection, n));
	vec3 reflectance = irradiance * DiffuseColor;
	fragColor = vec4(sqrt(reflectance), alpha);
}