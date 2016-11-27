#version 330 core

#if defined(DIFFUSE_MAP) || defined(NORMAL_MAP) || defined(SPECULAR_MAP)
#define TEXTURED
#endif

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 normalMatrix;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
#ifdef TEXTURED
layout(location = 2) in vec2 texel;
#endif
#ifdef NORMAL_MAP
layout(location = 3) in vec3 tangent;
#endif

out VertexData {
  vec3 position;
  vec3 normal;
#ifdef TEXTURED
  vec2 texel;
#endif
#ifdef NORMAL_MAP
  vec3 tangent;
#endif
} outputVertex;

void main(void)
{
  vec4 pos = modelViewMatrix * vec4(position, 1.0);

  outputVertex.position = pos.xyz;
  outputVertex.normal = vec3(normalMatrix * vec4(normal, 0.0));
#ifdef TEXTURED
  outputVertex.texel = texel;
#endif
#ifdef NORMAL_MAP
  outputVertex.tangent = vec3(normalMatrix * vec4(tangent, 0.0));
#endif

  gl_Position = projectionMatrix * pos;
}
