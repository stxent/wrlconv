// Fragment program
#define LIGHTS 2
varying vec3 normal, pos, ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
varying vec3 light[LIGHTS];
varying float attenuation[LIGHTS];
uniform sampler2D diffuseTexture;

void main()
{
  vec4 color = vec4(ambientGlobal, gl_FrontMaterial.diffuse.a);
  vec4 texel;
  vec3 reflection;
  vec3 view = normalize(-pos.xyz);

  for (int i = 0; i < LIGHTS; i++)
  {
    reflection = normalize(-reflect(light[i], normal));
    color.rgb += ambient[i] * attenuation[i];
    color.rgb += diffuse[i].rgb * max(0.0, dot(normal, light[i])) * attenuation[i];
    if (gl_FrontMaterial.shininess != 0.0)
      color.rgb += specular[i] * pow(max(0.0, dot(reflection, view)), gl_FrontMaterial.shininess) * attenuation[i];
  }

  texel = texture2D(diffuseTexture, gl_TexCoord[0].st);
  color *= texel;

  gl_FragColor = clamp(color, 0.0, 1.0);
}