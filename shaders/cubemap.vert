// Vertex program
// uniform vec3 cameraPosition;
varying vec3 view, normal;

void main()
{
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
  vec4 eye = vec4(0.0, 0.0, 0.0, 1.0);
  mat4 inv = inverse(gl_ModelViewMatrix);
//   mat3 rotMatrix = mat3(1.0,  0.0,  0.0,
//                         0.0,  0.0,  1.0,
//                         0.0, -1.0,  0.0);
  view = normalize(vec3(inv * eye));
//   view = normalize(cameraPosition);
//   view = refract(cameraPosition, gl_Normal, 1.0);
//   normal = normalize(gl_NormalMatrix * gl_Normal);
//   normal = rotMatrix * gl_Normal;
  normal = gl_Normal;
}
