// Fragment program
#define LIGHTS 2

varying vec3 normal, pos;
varying vec3 ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
uniform sampler2D diffuseTexture;

void main()
{
  vec4 color = vec4(ambientGlobal, gl_FrontMaterial.diffuse.a);
  vec3 reflection, light, view = normalize(-pos.xyz);

  for (int i = 0; i < LIGHTS; i++)
  {
    light = normalize(gl_LightSource[i].position.xyz - pos);
    reflection = normalize(-reflect(light, normal));
    color.rgb += ambient[i];
    color.rgb += diffuse[i].rgb * max(0.0, dot(normal, light));
    if (gl_FrontMaterial.shininess != 0.0)
      color.rgb += specular[i] * pow(max(0.0, dot(reflection, view)), gl_FrontMaterial.shininess);
  }

  color *= texture2D(diffuseTexture, gl_TexCoord[0].st);
  gl_FragColor = clamp(color, 0.0, 1.0);
}
