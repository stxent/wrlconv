// Fragment program
varying vec3 normal, pos, ambientGlobal, ambient, specular;
varying vec4 diffuse;
uniform sampler2D diffuseTexture, normalTexture;
varying vec3 light, view;

void main()
{
  vec4 texel;
  vec4 color = vec4(ambientGlobal, diffuse.a);

  vec3 bumpNormal = vec3(texture2D(normalTexture, gl_TexCoord[0].st));
  bumpNormal = normalize((bumpNormal) * 2.0 - 1.0);
  vec3 reflection = normalize(-reflect(light, bumpNormal));

  float diffuseIntensity = max(0.0, dot(bumpNormal, light));
  color.rgb += ambient;
  color.rgb += diffuse.rgb * diffuseIntensity;
  texel = texture2D(diffuseTexture, gl_TexCoord[0].st);
  color *= texel;
  if (diffuseIntensity > 0.0)
  {
    float specularModifier = max(0.0, dot(reflection, view));
    color.rgb += specular * pow(specularModifier, gl_FrontMaterial.shininess);
  }

  gl_FragColor = clamp(color, 0.0, 1.0);
}
