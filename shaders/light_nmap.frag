// Fragment program
#define LIGHTS 2
varying vec3 normal, pos, ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
varying vec3 light[LIGHTS], view;
// varying vec3 ta, bi;
varying float attenuation[LIGHTS];
uniform sampler2D diffuseTexture, normalTexture;

void main()
{
  vec4 texel;
  vec3 reflection;
  float diffuseIntensity, specularModifier;
  vec4 color = vec4(ambientGlobal, gl_FrontMaterial.diffuse.a);
  vec3 bumpNormal = vec3(texture2D(normalTexture, gl_TexCoord[0].st));
  bumpNormal = normalize((bumpNormal) * 2.0 - 1.0);

  for (int i = 0; i < LIGHTS; i++)
  {
    reflection = normalize(-reflect(light[i], bumpNormal));
    diffuseIntensity = max(0.0, dot(bumpNormal, light[i]));
    color.rgb += ambient[i] * attenuation[i];
    color.rgb += diffuse[i].rgb * diffuseIntensity * attenuation[i];
    if (diffuseIntensity > 0.0)
    {
      specularModifier = max(0.0, dot(reflection, view));
      color.rgb += specular[i] * pow(specularModifier, gl_FrontMaterial.shininess) * attenuation[i];
    }
  }
  texel = texture2D(diffuseTexture, gl_TexCoord[0].st);
  color *= texel;
//   color.rgb = ta;

  gl_FragColor = clamp(color, 0.0, 1.0);
}
