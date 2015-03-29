#version 410 core

#if defined(DIFFUSE_MAP) || defined(NORMAL_MAP) || defined(SPECULAR_MAP)
#define TEXTURED
#endif

#ifdef DIFFUSE_MAP
uniform sampler2D diffuseTexture;
#endif
#ifdef NORMAL_MAP
uniform sampler2D normalTexture;
#endif
#ifdef SPECULAR_MAP
uniform sampler2D specularTexture;
#endif

uniform vec3 lightDiffuseColor[LIGHT_COUNT];
uniform float lightAmbientIntensity;

uniform vec4 materialDiffuseColor;
uniform vec3 materialSpecularColor;
uniform vec3 materialEmissiveColor;
uniform float materialShininess;

in vec3 calcPosition;
in vec3 calcNormal;
in vec3 calcLightPosition[LIGHT_COUNT];
#ifdef TEXTURED
in vec2 calcTexel;
#endif
#ifdef NORMAL_MAP
in vec3 calcTangent;
#endif

layout(location = 0) out vec4 color;

void main(void)
{
  vec3 view = normalize(-calcPosition.xyz);
  color = vec4(materialEmissiveColor, materialDiffuseColor.a);

#ifdef NORMAL_MAP
  vec3 binormal = cross(calcTangent, calcNormal);
  vec3 normal = texture2D(normalTexture, calcTexel.st).rgb;

  normal = normalize(normal * 2.0 - 1.0);
  mat3 tbnMatrix = mat3(
      calcTangent.x, binormal.x, calcNormal.x,
      calcTangent.y, binormal.y, calcNormal.y,
      calcTangent.z, binormal.z, calcNormal.z);
  view = tbnMatrix * view;
#else
  vec3 normal = calcNormal;
#endif

  for (int i = 0; i < LIGHT_COUNT; i++)
  {
    vec3 light = normalize(calcLightPosition[i] - calcPosition);
#ifdef NORMAL_MAP
    light = tbnMatrix * light;
#endif

    color.rgb += materialDiffuseColor.rgb * lightDiffuseColor[i] * max(0.0, dot(normal, light));
    color.rgb += materialDiffuseColor.rgb * lightAmbientIntensity;
    if (materialShininess != 0.0)
    {
      vec3 reflection = normalize(-reflect(light, normal));
      color.rgb += materialSpecularColor * pow(max(0.0, dot(reflection, view)), materialShininess);
    }
  }

#ifdef DIFFUSE_MAP
  color *= texture2D(diffuseTexture, calcTexel.st);
#endif
  color = clamp(color, 0.0, 1.0);
}
