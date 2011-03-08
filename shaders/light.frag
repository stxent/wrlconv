// Fragment program
varying vec3 normal, pos;
varying vec4 diffuse, ambientGlobal, ambient;

void main() {
  vec4 color = ambientGlobal;
  vec3 light = -normalize(pos - gl_LightSource[0].position.xyz);
  vec3 reflection = normalize(-reflect(light, normal));
  vec3 view = normalize(-pos.xyz);

  color += diffuse * max(0.0, dot(normal, light));
  if (gl_FrontMaterial.shininess != 0.0) {
    color += gl_LightSource[0].specular * gl_FrontMaterial.specular * pow(max(0.0, dot(reflection, view)), gl_FrontMaterial.shininess);
  }

  gl_FragColor = color;
}