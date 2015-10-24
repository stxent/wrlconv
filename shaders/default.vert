#version 330 core

#if defined(DIFFUSE_MAP) || defined(NORMAL_MAP) || defined(SPECULAR_MAP)
#define TEXTURED
#endif

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 normalMatrix;

uniform vec3 lightPosition[LIGHT_COUNT];

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
#ifdef TEXTURED
layout(location = 2) in vec2 texel;
#endif
#ifdef NORMAL_MAP
layout(location = 3) in vec3 tangent;
#endif

out vec3 calcPosition;
out vec3 calcNormal;
out vec3 calcLightPosition[LIGHT_COUNT];
#ifdef TEXTURED
out vec2 calcTexel;
#endif
#ifdef NORMAL_MAP
out vec3 calcTangent;
#endif

void main(void)
{
  vec4 pos = modelViewMatrix * vec4(position, 1.0);

  for (int i = 0; i < LIGHT_COUNT; ++i)
    calcLightPosition[i] = vec3(modelViewMatrix * vec4(lightPosition[i], 1.0));

  calcPosition = pos.xyz;
  calcNormal = vec3(normalMatrix * vec4(normal, 0.0));
#ifdef TEXTURED
  calcTexel = texel;
#endif
#ifdef NORMAL_MAP
  calcTangent = vec3(normalMatrix * vec4(tangent, 0.0));
#endif

  gl_Position = projectionMatrix * pos;
}
