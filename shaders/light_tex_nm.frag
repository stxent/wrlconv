// Fragment program
#define LIGHTS 2
varying vec3 ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
varying vec3 view;
varying vec3 light[LIGHTS];
uniform sampler2D diffuseTexture, normalTexture;

void main()
{
  vec3 reflection;
  float diffuseIntensity, specularModifier;
  vec4 color = vec4(ambientGlobal, gl_FrontMaterial.diffuse.a);
  vec3 bumpNormal = vec3(texture2D(normalTexture, gl_TexCoord[0].st));
  bumpNormal = normalize((bumpNormal - 0.5) * 2.0);

  for (int i = 0; i < LIGHTS; i++)
  {
    reflection = normalize(-reflect(light[i], bumpNormal));
    diffuseIntensity = max(0.0, dot(bumpNormal, light[i]));
    color.rgb += ambient[i];
    color.rgb += diffuse[i].rgb * diffuseIntensity;
    if ((diffuseIntensity > 0.0) && (gl_FrontMaterial.shininess > 0.0))
    {
      specularModifier = max(0.0, dot(reflection, view));
      color.rgb += specular[i] * pow(specularModifier, gl_FrontMaterial.shininess);
    }
  }
  color *= texture2D(diffuseTexture, gl_TexCoord[0].st);

  gl_FragColor = clamp(color, 0.0, 1.0);
}
