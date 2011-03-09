// Fragment program
varying vec3 normal, pos, ambientGlobal, ambient, specular;
varying vec4 diffuse;

void main()
{
  vec4 color = vec4(ambientGlobal, 0.0);
  vec3 light = -normalize(pos - gl_LightSource[0].position.xyz);
  vec3 reflection = normalize(-reflect(light, normal));
  vec3 view = normalize(-pos.xyz);

  color.rgb += ambient;
  color = vec4(diffuse.rgb * max(0.0, dot(normal, light)), diffuse.a);
  if (gl_FrontMaterial.shininess != 0.0)
    color.rgb += specular * pow(max(0.0, dot(reflection, view)), gl_FrontMaterial.shininess);

  gl_FragColor = color;
}