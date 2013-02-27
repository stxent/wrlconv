// Fragment program
#define LIGHTS 2

varying vec3 normal, pos, tangent;
varying vec3 ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
uniform sampler2D diffuseTexture, normalTexture;

void main()
{
  vec4 color = vec4(ambientGlobal, gl_FrontMaterial.diffuse.a);
  vec3 reflection, light, view = normalize(-pos.xyz);
  vec3 bumpNormal = vec3(texture2D(normalTexture, gl_TexCoord[0].st));
  vec3 binormal = cross(tangent, normal);
  mat3 tbnMatrix;
  float diffuseIntensity, specularModifier;

  bumpNormal = normalize((bumpNormal) * 2.0 - 1.0);
  tbnMatrix = mat3(tangent.x, binormal.x, normal.x,
                   tangent.y, binormal.y, normal.y,
                   tangent.z, binormal.z, normal.z);
  view = tbnMatrix * view;

  for (int i = 0; i < LIGHTS; i++)
  {
    light = tbnMatrix * normalize(gl_LightSource[i].position.xyz - pos);
    reflection = normalize(-reflect(light, bumpNormal));
    diffuseIntensity = max(0.0, dot(bumpNormal, light));
    color.rgb += ambient[i];
    color.rgb += diffuse[i].rgb * diffuseIntensity;
    if (diffuseIntensity > 0.0)
      color.rgb += specular[i] * pow(max(0.0, dot(reflection, view)), gl_FrontMaterial.shininess);
  }

  color *= texture2D(diffuseTexture, gl_TexCoord[0].st);
  gl_FragColor = clamp(color, 0.0, 1.0);
}
