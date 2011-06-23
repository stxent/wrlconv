// Vertex program
#define LIGHTS 2
varying vec3 ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
varying vec3 view;
varying vec3 light[LIGHTS];
attribute vec3 tangent;

void main()
{
  vec3 pos = vec3(gl_ModelViewMatrix * gl_Vertex);
  vec3 normal = normalize(gl_NormalMatrix * gl_Normal);
  gl_TexCoord[0] = gl_MultiTexCoord0;

  vec3 tang = normalize(gl_NormalMatrix * tangent);
  vec3 binormal = cross(tang, normal);
  mat3 tbnMatrix = inverse(mat3(tang, binormal, normal));
  view = normalize(-pos.xyz);
  view = tbnMatrix * view;

  ambientGlobal = vec3(gl_FrontMaterial.ambient * gl_LightModel.ambient);
  for (int i = 0; i < LIGHTS; i++)
  {
    light[i] = normalize(gl_LightSource[i].position.xyz - pos);
    light[i] = tbnMatrix * light[i];
    diffuse[i] = vec3(gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse);
    ambient[i] = vec3(gl_FrontMaterial.ambient * gl_LightSource[i].ambient);
    specular[i] = vec3(gl_FrontMaterial.specular * gl_LightSource[i].specular);
  }

  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
