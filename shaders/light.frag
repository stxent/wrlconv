// Fragment program
#define LIGHTS 2
varying vec3 ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
varying vec3 normal, pos;
varying vec3 light[LIGHTS];

void main()
{
  vec4 color = vec4(ambientGlobal, gl_FrontMaterial.diffuse.a);
  vec3 reflection;
  vec3 view = normalize(-pos.xyz);

  for (int i = 0; i < LIGHTS; i++)
  {
//     reflection = normalize(-reflect(light[i], normal));
    color.rgb += ambient[i];
    color.rgb += diffuse[i].rgb * max(0.0, dot(normal, light[i]));
    if (gl_FrontMaterial.shininess != 0.0)
      color.rgb += specular[i] * pow(max(0.0, dot(-reflect(light[i], normal), view)), gl_FrontMaterial.shininess);
  }

  gl_FragColor = clamp(color, 0.0, 1.0);
}
