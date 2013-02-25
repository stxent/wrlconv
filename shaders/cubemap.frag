// Fragment program
uniform samplerCube cubeTexture;
varying vec3 view, normal;

void main()
{
  vec3 nview = normalize(view);
  vec3 reflection = normalize(reflect(view, normal));
//   if (abs(reflection.x - reflection.y) < 0.01)
//     reflection.x += 0.01;
  vec4 cube_color = vec4(textureCube(cubeTexture, reflection.xzy).rgb, 1.0);
//   vec4 cube_color = vec4(textureCube(cubeTexture, view.xyz).rgb, 1.0);

  gl_FragColor = cube_color;
//   gl_FragColor.rgb = nview;;
}
