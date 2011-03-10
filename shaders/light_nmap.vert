// Vertex program
varying vec3 normal, pos, ambientGlobal, ambient, specular;
varying vec4 diffuse;
varying vec3 light, view;
attribute vec3 tangent;

void main()
{
  normal = normalize(gl_NormalMatrix * gl_Normal);
  pos = vec3(gl_ModelViewMatrix * gl_Vertex);
  //gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
  gl_TexCoord[0] = gl_MultiTexCoord0;
  /* Compute the diffuse, ambient and globalAmbient terms */
  diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
  ambient = vec3(gl_FrontMaterial.ambient * gl_LightSource[0].ambient);
  specular = vec3(gl_FrontMaterial.specular * gl_LightSource[0].specular);
  ambientGlobal = vec3(gl_FrontMaterial.ambient * gl_LightModel.ambient);

  light = -normalize(pos - gl_LightSource[0].position.xyz);
  view = normalize(-pos.xyz);
  vec3 tang = normalize(gl_NormalMatrix * tangent);
  vec3 binormal = cross(normal, tang);

  mat3 tbnMatrix = mat3(tang.x, binormal.x, normal.x,
                        tang.y, binormal.y, normal.y,
                        tang.z, binormal.z, normal.z);
  light = tbnMatrix * light;
  view = tbnMatrix * view;

  gl_Position = ftransform();
}
