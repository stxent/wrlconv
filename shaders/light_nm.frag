// Fragment program
#define LIGHTS 2
varying vec3 normal, pos, ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
varying vec3 light[LIGHTS], view;
uniform sampler2D normalTexture;

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
    color.rgb += ambient[i];
    color.rgb += diffuse[i].rgb * diffuseIntensity;
    if (diffuseIntensity > 0.0)
    {
      specularModifier = max(0.0, dot(reflection, view));
      color.rgb += specular[i] * pow(specularModifier, gl_FrontMaterial.shininess);
    }
  }

  gl_FragColor = clamp(color, 0.0, 1.0);
}
