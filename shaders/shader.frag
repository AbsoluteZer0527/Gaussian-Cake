#version 330 core

// Inputs to the fragment shader are the outputs of the same name from the vertex shader.
// Note that you do not have access to the vertex shader's default output, gl_Position.

in vec3 fragNormal;
in vec3 fragAlbedo;          // display colour per vertex
in vec2 fragUV;
in vec3 fragAlbedoA;         // morph.vert: the start and end colours, unmixed
in vec3 fragAlbedoB;
in vec2 fragUVB;             // morph.vert: B's uv

// uniforms used for lighting
uniform vec3 AmbientColor = vec3(0.2);
uniform vec3 LightDirection = normalize(vec3(1, 5, 2));
uniform vec3 LightColor = vec3(1);
uniform vec3  DiffuseColor;
uniform float alpha = 1.0;
uniform int   useVertexColor = 0;   // 1: fragAlbedo instead of DiffuseColor
uniform vec3  MaterialColor = vec3(1.0);   // display colour, multiplies fragAlbedo
uniform int   useTexture = 0;       // 1: also multiply by albedoTex at fragUV
uniform sampler2D albedoTex;
// Textured morph (MorphTexturedRenderer): each side is its own colour x its
// own texture, blended per pixel by morphT
uniform int   useMorphTexture = 0;
uniform int   useTextureB = 0;     // 1: side B multiplies by albedoTexB at fragUVB
uniform sampler2D albedoTexB;
uniform float morphT;

out vec4 fragColor;

void main()
{
	// Two-sided lighting: back faces use the flipped normal, so open sheets
	// and meshes with inverted winding still shade on the side being viewed.
	vec3 n = normalize(fragNormal);
	if (!gl_FrontFacing) n = -n;

	vec3 irradiance  = AmbientColor + LightColor * max(0, dot(LightDirection, n));
	// Display colours (vertex x material x texture, the same product as
	// MeshData::Albedo); squared they are (roughly) linear like DiffuseColor
	vec3 albedo = DiffuseColor;
	if (useMorphTexture != 0) {
		vec3 a = fragAlbedoA;
		if (useTexture != 0) a *= texture(albedoTex, fragUV).rgb;
		vec3 b = fragAlbedoB;
		if (useTextureB != 0) b *= texture(albedoTexB, fragUVB).rgb;
		vec3 c = mix(a, b, clamp(morphT, 0.0, 1.0));   // easing can overshoot
		albedo = c * c;
	} else if (useVertexColor != 0) {
		vec3 c = fragAlbedo * MaterialColor;
		if (useTexture != 0) c *= texture(albedoTex, fragUV).rgb;
		albedo = c * c;
	}
	vec3 reflectance = irradiance * albedo;
	fragColor = vec4(sqrt(reflectance), alpha);
}