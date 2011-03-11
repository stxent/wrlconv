// Fragment program
varying vec3 normal, pos, ambientGlobal, ambient, specular;
varying vec4 diffuse;
uniform sampler2D diffuseTexture;

void main()
{
  vec4 texel;
  vec4 color = vec4(ambientGlobal, diffuse.a);
  vec3 light = -normalize(pos - gl_LightSource[0].position.xyz);
  vec3 view = normalize(-pos.xyz);
  vec3 reflection = normalize(-reflect(light, normal));

  color.rgb += ambient;
  color.rgb += (diffuse.rgb + color.rgb) * max(0.0, dot(normal, light));

  texel = texture2D(diffuseTexture, gl_TexCoord[0].st);
  color *= texel;

  if (gl_FrontMaterial.shininess != 0.0)
    color.rgb += specular * pow(max(0.0, dot(reflection, view)), gl_FrontMaterial.shininess);

  gl_FragColor = clamp(color, 0.0, 1.0);
}
