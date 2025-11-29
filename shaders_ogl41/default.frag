#version 300 es
precision mediump float;

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

uniform vec3 lightPosition[LIGHT_COUNT];
uniform vec3 lightDiffuseColor[LIGHT_COUNT];
uniform float lightAmbientIntensity;

uniform vec4 materialDiffuseColor;
uniform vec3 materialSpecularColor;
uniform vec3 materialEmissiveColor;
uniform float materialShininess;

in vec3 outputPosition;
in vec3 outputNormal;
#ifdef TEXTURED
in vec2 outputTexel;
#endif
#ifdef NORMAL_MAP
in vec3 outputTangent;
#endif

layout(location = 0) out vec4 color;

void main(void)
{
  vec3 view = normalize(-outputPosition.xyz);
  color = vec4(materialEmissiveColor, materialDiffuseColor.a);

#ifdef NORMAL_MAP
  vec3 binormal = cross(outputTangent, outputNormal);
  vec3 normal = texture2D(normalTexture, outputTexel.st).rgb;

  normal = normalize(normal * 2.0 - 1.0);
  mat3 tbnMatrix = mat3(
      outputTangent.x, binormal.x, outputNormal.x,
      outputTangent.y, binormal.y, outputNormal.y,
      outputTangent.z, binormal.z, outputNormal.z);
  view = tbnMatrix * view;
#else
  vec3 normal = outputNormal;
#endif

  for (int i = 0; i < LIGHT_COUNT; i++)
  {
    vec3 light = normalize(lightPosition[i] - outputPosition);
#ifdef NORMAL_MAP
    light = tbnMatrix * light;
#endif

    color.rgb += materialDiffuseColor.rgb * lightDiffuseColor[i] * max(0.0, dot(normal, light));
    color.rgb += materialDiffuseColor.rgb * lightAmbientIntensity;
    if (materialShininess != 0.0)
    {
      vec3 reflection = normalize(-reflect(light, normal));
      vec3 specularPart = materialSpecularColor * pow(max(0.0, dot(reflection, view)), materialShininess);
#ifdef SPECULAR_MAP
      specularPart *= texture2D(specularTexture, outputTexel.st).rgb;
#endif
      color.rgb += specularPart;
    }
  }

#ifdef DIFFUSE_MAP
  color *= texture2D(diffuseTexture, outputTexel.st);
#endif
  color = clamp(color, 0.0, 1.0);
}
