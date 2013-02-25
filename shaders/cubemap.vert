// Vertex program
// uniform vec3 cameraPosition;
varying vec3 view, normal;

void main()
{
//   vec3 pos = vec3(gl_ModelViewMatrix * gl_Vertex);
// //   vec4 eye = vec4(0.0, 0.0, 0.0, 1.0);
// //   mat4 inv = inverse(gl_ModelViewMatrix);
//   mat4 inv = inverse(gl_ModelViewProjectionMatrix);
//   mat3 rotMatrix = mat3(1.0,  0.0,  0.0,
//                         0.0,  0.0,  1.0,
//                         0.0, -1.0,  0.0);
  //view = vec3(inv * eye) - vec3(gl_Vertex);
  view = inverse(gl_ModelViewMatrix)[3].xyz - vec3(gl_Vertex);
//   view = -vec3(gl_ModelViewProjectionMatrix * gl_Vertex);
//   view = normalize(cameraPosition);
//   view = refract(cameraPosition, gl_Normal, 1.0);
//   normal = normalize(gl_NormalMatrix * gl_Normal);
//   normal = rotMatrix * gl_Normal;
  normal = gl_Normal;
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
